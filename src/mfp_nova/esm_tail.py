"""Task-adapt the ESM tail while reusing a frozen-prefix cache."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen pretrained projection plus a zero-initialized low-rank update."""

    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


class ESMTail(nn.Module):
    """Continue after a cached intermediate layer, retaining CLS and EOS context.

    Only LoRA Q/K/V/output projections learn. Frozen tail weights are reloaded
    from the pinned pretrained revision when opening a compact checkpoint.
    """

    def __init__(self, name, revision, representation_layer=29, rank=8, alpha=16, dropout=.05):
        super().__init__()
        from transformers import AutoModel

        model = AutoModel.from_pretrained(name, revision=revision, add_pooling_layer=False,
                                          attn_implementation="eager")
        if not 0 < representation_layer < model.config.num_hidden_layers:
            raise ValueError("ESM prefix layer must leave at least one trainable tail layer")
        model.requires_grad_(False)
        self.layers = nn.ModuleList(list(model.encoder.layer)[representation_layer:])
        self.final_norm = model.encoder.emb_layer_norm_after
        for layer in self.layers:
            attention = layer.attention.self
            for key in ("query", "key", "value"):
                setattr(attention, key, LoRALinear(getattr(attention, key), rank, alpha, dropout))
            output = layer.attention.output
            output.dense = LoRALinear(output.dense, rank, alpha, dropout)

    def forward(self, batch):
        residues, mask = batch["esm"], batch["mask"]
        length = residues.shape[1]
        lengths = mask.sum(1)
        # Padding in the residue cache is not an EOS token. Restore each EOS at
        # its own sequence boundary, rather than after the batch's longest row.
        hidden = F.pad(residues, (0, 0, 1, 1))
        hidden[:, 0] = batch["esm_cls"]
        hidden[torch.arange(len(lengths), device=mask.device), lengths + 1] = batch["esm_eos"]
        valid = torch.arange(length + 2, device=mask.device)[None] < lengths[:, None] + 2
        attention_mask = (~valid[:, None, None]).to(hidden.dtype) * torch.finfo(hidden.dtype).min
        for layer in self.layers:
            rotary = layer.attention.self.rotary_embeddings
            # Validation caches made under inference_mode cannot be saved for
            # backward. Refresh them when gradient-enabled execution resumes.
            if (torch.is_grad_enabled() and rotary is not None and rotary._cos_cached is not None
                    and rotary._cos_cached.is_inference()):
                rotary._seq_len_cached = None
            output = layer(hidden, attention_mask=attention_mask, head_mask=None, output_attentions=False)
            # Transformers 4.44 returns a tuple; newer ESM layers return a tensor.
            hidden = output[0] if isinstance(output, tuple) else output
        return self.final_norm(hidden)[:, 1:length + 1] * mask[..., None]
