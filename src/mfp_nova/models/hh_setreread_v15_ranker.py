"""Independent Fine39 scores with optional cluster-held-out pair supervision."""

import torch
from torch import nn
from torch.nn import functional as F

from .sequence_features import prior_bias
from .termiscale import endpoint_positions
from ..multimodal import function_projection
from ..research_layers import masked_mean, observed_label_loss


def select_hard_pair(reference, targets):
    """Compete the highest unrecorded call against the closest missed annotation."""
    missed = (targets > 0) & (reference < 0)
    called = (targets == 0) & (reference >= 0)
    add = reference.masked_fill(~missed, -torch.inf).argmax(dim=1, keepdim=True)
    drop = reference.masked_fill(~called, -torch.inf).argmax(dim=1, keepdim=True)
    return add, drop, missed.any(1) & called.any(1)


class HHSetRereadV15Ranker(nn.Module):
    def __init__(self, input_dim, peptide_input_dim, num_labels, vocab_size, statistics,
                 dim=128, heads=4, dropout=.15):
        super().__init__()
        self.statistics = statistics
        self.primary = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
        self.peptide = nn.Sequential(nn.LayerNorm(peptide_input_dim), nn.Linear(peptide_input_dim, dim))
        self.fusion = nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
        self.position = nn.Linear(4, dim, bias=False)
        self.local = nn.ModuleList(nn.Conv1d(dim, dim // 2, width, padding=width // 2)
                                   for width in (3, 7))
        self.local_dropout = nn.Dropout(dropout)
        self.context = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim,
                                                  dropout=dropout, activation="gelu",
                                                  batch_first=True, norm_first=True)
        self.keys = nn.Linear(dim, num_labels, bias=False)
        self.readout = nn.Parameter(torch.empty(num_labels, dim))
        nn.init.normal_(self.readout, std=dim ** -.5)
        self.global_readout = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, num_labels))
        self.bias = nn.Parameter(prior_bias(statistics))

        labels = statistics["labels"]
        axes = torch.tensor(function_projection(labels, "fine39"))
        unknown = torch.tensor([label.endswith("__unknown") for label in labels])
        same_axis = axes @ axes.T > 0
        specific_am = axes[:, :4].any(dim=1)
        generic_am = axes[:, 4].bool()
        ambiguity = ((same_axis & (unknown[:, None] | unknown[None, :]))
                     | (specific_am[:, None] & generic_am[None, :])
                     | (generic_am[:, None] & specific_am[None, :]))
        self.register_buffer("ambiguous_pair", ambiguity)

    def forward(self, batch):
        mask = batch["mask"]
        padding = ~mask[..., None]
        primary = self.primary(batch["esm"].masked_fill(padding, 0))
        peptide = self.peptide(batch["esm_peptide"].masked_fill(padding, 0))
        residues = self.fusion(torch.cat((primary, peptide), dim=-1))
        residues = (residues + self.position(endpoint_positions(mask))).masked_fill(padding, 0)
        local = torch.cat([conv(residues.transpose(1, 2)) for conv in self.local], dim=1)
        residues = residues + self.local_dropout(F.gelu(local.transpose(1, 2))).masked_fill(padding, 0)
        residues = self.context(residues, src_key_padding_mask=~mask).masked_fill(padding, 0)
        attention = self.keys(residues).transpose(1, 2).float().masked_fill(~mask[:, None], -torch.inf)
        evidence = torch.bmm(attention.softmax(-1).to(residues.dtype), residues)
        local_logits = (evidence * self.readout).sum(-1)
        summary = torch.cat((masked_mean(residues, mask),
                             residues.masked_fill(padding, -torch.inf).amax(1)), dim=-1)
        return {"logits": local_logits + self.global_readout(summary) + self.bias}

    def compute_loss(self, batch, settings, return_output=False):
        output = self(batch)
        logits, targets = output["logits"].float(), batch["y"]
        weak_weight = settings.get("unlabeled_weight", .3)
        classification = observed_label_loss(logits, targets, self.statistics, weak_weight)
        loss = classification
        terms = {"classification": classification}
        if settings.get("hard_pair", 0):
            add, drop, valid = select_hard_pair(batch["base_logits"], targets)
            weight = torch.where(self.ambiguous_pair[add, drop].squeeze(1), .25, 1.) * weak_weight
            margin = (logits.gather(1, add) - logits.gather(1, drop)).squeeze(1)
            pair_loss = (F.softplus(-margin) * weight * valid).sum() / valid.sum().clamp_min(1)
            loss = loss + settings["hard_pair"] * pair_loss
            terms.update(hard_pair=pair_loss, hard_pair_fraction=valid.float().mean())
        return (loss, terms, output) if return_output else (loss, terms)
