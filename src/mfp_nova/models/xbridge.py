"""Fine39 label evidence with cross-cluster positive supervision."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..esm_tail import LoRALinear
from ..multimodal import suppress_generic_antimicrobial
from ..research_layers import observed_label_loss


class PeptideLoRA(nn.Module):
    def __init__(self, name, revision, max_length=80, layers=4, rank=8, alpha=16, dropout=.05):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        residues = tokenizer.convert_tokens_to_ids(list(alphabet))
        if (tokenizer(alphabet)["input_ids"] != [tokenizer.cls_token_id, *residues, tokenizer.eos_token_id]
                or tokenizer.unk_token_id in residues or len(set(residues)) != len(alphabet)):
            raise ValueError("Peptide ESM tokenizer must map each canonical residue to one token")
        self.register_buffer("residue_ids", torch.tensor([tokenizer.pad_token_id, *residues]), persistent=False)
        self.pad, self.cls, self.eos = tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.eos_token_id
        self.max_length = max_length
        self.encoder = AutoModel.from_pretrained(name, revision=revision, add_pooling_layer=False,
                                                 attn_implementation="eager").requires_grad_(False).eval()
        self.dim = self.encoder.config.hidden_size
        if not 1 <= layers <= len(self.encoder.encoder.layer):
            raise ValueError("Invalid number of adapted peptide ESM layers")
        for layer in self.encoder.encoder.layer[-layers:]:
            attention = layer.attention.self
            for key in ("query", "value"):
                projection = LoRALinear(getattr(attention, key), rank, alpha, dropout)
                setattr(attention, key, projection)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        for layer in self.encoder.encoder.layer:
            attention = layer.attention.self
            for projection in (attention.query, attention.value):
                if isinstance(projection, LoRALinear):
                    projection.dropout.train(mode)
        return self

    def forward(self, aa, mask):
        if aa.shape[1] + 2 > self.max_length:
            raise ValueError("Sequence exceeds the peptide ESM token limit")
        lengths = mask.sum(1)
        ids = F.pad(self.residue_ids[aa.masked_fill(~mask, 0)], (1, 1), value=self.pad)
        ids[:, 0] = self.cls
        ids.scatter_(1, (lengths + 1)[:, None], self.eos)
        valid = torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None] + 2
        for layer in self.encoder.encoder.layer:
            rotary = layer.attention.self.rotary_embeddings
            if (torch.is_grad_enabled() and rotary is not None and rotary._cos_cached is not None
                    and rotary._cos_cached.is_inference()):
                rotary._seq_len_cached = None
        return self.encoder(input_ids=ids, attention_mask=valid.long()).last_hidden_state[:, 1:-1]


def axis_coverage(logits, targets, projection, temperature):
    positive = targets.bool()[:, :, None] & projection[None]
    counts = positive.sum(1)
    valid = suppress_generic_antimicrobial(counts > 0)
    scaled = logits.float()[:, :, None] / temperature
    masked = scaled.masked_fill(~positive, -torch.inf)
    masked = torch.where(valid[:, None], masked, 0.)
    score = temperature * (masked.logsumexp(1) - counts.clamp_min(1).float().log())
    per_row = (F.softplus(-score) * valid).sum(1) / valid.sum(1).clamp_min(1)
    return per_row.sum() / valid.any(1).sum().clamp_min(1)


def positive_bridge(evidence, targets, clusters, margin):
    """All within-batch pairs with observed shared labels and distinct clusters."""
    normalized = F.normalize(evidence.float(), dim=-1, eps=1e-6).transpose(0, 1)
    cosine = normalized @ normalized.transpose(1, 2)
    recorded = targets.bool().T
    valid = recorded[:, :, None] & recorded[:, None, :] & (clusters[:, None] != clusters[None, :])
    count = valid.sum((1, 2))
    per_label = (F.relu(margin - cosine).square() * valid).sum((1, 2)) / count.clamp_min(1)
    return (per_label.sum() / (count > 0).sum().clamp_min(1), count.sum() / 2,
            ((cosine < margin) & valid).sum() / count.sum().clamp_min(1))


class XBridge(nn.Module):
    def __init__(self, num_labels, statistics, input_dim, peptide_input_dim, vocab_size,
                 function_projection, dim=192, dropout=.15, online_encoder=None, reference_args=None):
        super().__init__()
        self.statistics = statistics
        self.register_buffer("function_projection", torch.as_tensor(function_projection, dtype=torch.bool),
                             persistent=False)
        self.peptide_lora = PeptideLoRA(**online_encoder) if online_encoder else None
        if self.peptide_lora is not None and self.peptide_lora.dim != peptide_input_dim:
            raise ValueError("Online and cached peptide ESM dimensions differ")
        self.chemistry = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.layer = nn.Sequential(nn.Linear(input_dim + peptide_input_dim + 32, dim), nn.LayerNorm(dim),
                                   nn.GELU(), nn.Dropout(dropout))
        self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.queries = nn.Parameter(torch.randn(num_labels, dim))
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Parameter(torch.empty(num_labels, dim))
        nn.init.uniform_(self.classifier, -1 / math.sqrt(dim), 1 / math.sqrt(dim))
        self.bias = nn.Parameter(torch.full((num_labels,), -1.))
        self.reference = None
        if reference_args is not None:
            from .theraset import TheraSet

            self.reference = TheraSet(**reference_args).requires_grad_(False).eval()
            self.residual = nn.Sequential(nn.Linear(dim + 1, dim), nn.GELU(), nn.Linear(dim, 1))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        if self.reference is not None:
            self.reference.eval()
        return self

    def forward(self, batch):
        mask = batch["mask"].bool()
        peptide = (self.peptide_lora(batch["aa"], mask) if self.peptide_lora is not None
                   else batch["esm_peptide"])
        primary = batch["esm"].masked_fill(~mask[..., None], 0.)
        peptide = peptide.masked_fill(~mask[..., None], 0.)
        residues = self.layer(torch.cat((F.layer_norm(primary, (primary.shape[-1],)),
                                         F.layer_norm(peptide, (peptide.shape[-1],)),
                                         self.chemistry(batch["residue_chem"].masked_fill(~mask, 0))), dim=-1))
        scores = (self.queries @ self.key(residues).transpose(1, 2)) / math.sqrt(residues.shape[-1])
        weights = scores.float().masked_fill(~mask[:, None], torch.finfo(torch.float32).min).softmax(-1)
        evidence = self.norm((weights * mask[:, None]).to(residues.dtype) @ self.value(residues))
        auxiliary = (evidence * self.classifier).sum(-1) + self.bias
        output = {"logits": auxiliary, "auxiliary_logits": auxiliary, "evidence": evidence}
        if self.reference is not None:
            with torch.no_grad(), torch.autocast(mask.device.type, enabled=False):
                reference = batch.get("base_logits") if self.training else None
                if reference is None:
                    reference = self.reference(batch)["logits"]
            residual = self.residual(torch.cat((evidence, auxiliary[..., None]), -1)).squeeze(-1)
            output.update(logits=reference.float() + residual.float(), reference_logits=reference, residual=residual)
        return output

    def compute_loss(self, batch, settings, return_output=False):
        output = self(batch)
        targets = batch["y"]
        weak = settings.get("unlabeled_weight", .3)
        main = observed_label_loss(output["logits"], targets, self.statistics, weak)
        terms = {"classification": main}
        loss = main
        if self.reference is not None:
            auxiliary = observed_label_loss(output["auxiliary_logits"], targets, self.statistics, weak)
            terms["evidence_classification"] = auxiliary
            loss = loss + settings.get("evidence_classification", .3) * auxiliary
        if settings.get("axis_coverage", 0):
            coverage = axis_coverage(output["logits"], targets, self.function_projection,
                                     settings.get("axis_temperature", .2))
            terms["axis_coverage"] = coverage
            loss = loss + settings["axis_coverage"] * coverage
        if settings.get("bridge", 0):
            bridge, pairs, active = positive_bridge(output["evidence"], targets, batch["xbridge_cluster"],
                                                     settings.get("bridge_margin", .5))
            terms.update(bridge=bridge, bridge_pairs=pairs, bridge_active=active,
                         evidence_std=output["evidence"].float().std(0, unbiased=False).mean().detach())
            loss = loss + settings["bridge"] * bridge
        return (loss, terms, output) if return_output else (loss, terms)
