"""Direct Fine39 prediction with chemical residue maps and family supervision."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..multimodal import function_projection
from ..research_layers import masked_mean, observed_label_loss
from . import HH_V20_CHEMICAL_MODES
from .sequence_features import prior_bias
from .termiscale import endpoint_positions
from .theraset import MotifCNN, motif_classifier


class HHSetRereadV20(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, family_statistics=None,
                 dim=96, cnn_channels=64, dropout=.15, peptide_plm=True, mode="full"):
        super().__init__()
        self.statistics = statistics
        self.family_statistics = family_statistics
        self.mode = mode
        self.cnn = MotifCNN(vocab_size, cnn_channels, dropout, chemical_evidence=True)
        if mode in HH_V20_CHEMICAL_MODES:
            self.direct = motif_classifier(self.cnn.output_channels, num_labels, dropout)
            with torch.no_grad():
                self.direct[-1].bias.copy_(prior_bias(statistics))
            if mode == "chemical_cnn_fusion":
                self.peptide = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim))
                self.position = nn.Linear(4, dim, bias=False)
                self.norm = nn.LayerNorm(dim)
                self.context_summary = nn.Linear(2 * dim, self.cnn.output_channels, bias=False)
            return
        self.terminal = nn.Linear(2, 64, bias=False)
        if mode != "theraset_direct":
            self.local = nn.Linear(self.cnn.output_channels, dim)
        self.peptide = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim)) if peptide_plm else None
        self.position = nn.Linear(4, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        if mode == "theraset_direct":
            self.context_summary = nn.Linear(2 * dim, self.cnn.output_channels, bias=False)
            self.direct = motif_classifier(self.cnn.output_channels, num_labels, dropout)
        else:
            self.direct = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, num_labels))
        with torch.no_grad():
            self.direct[-1].bias.copy_(prior_bias(statistics))
        if mode in ("direct", "theraset_direct"):
            return

        labels = statistics["labels"]
        projection = torch.tensor(function_projection(labels, "fine39"), dtype=torch.float32)
        self.register_buffer("projection", projection)
        self.register_buffer("family_of_label", projection.argmax(1))
        unspecified = torch.tensor([label.endswith("__unknown") for label in labels])
        concrete = (~unspecified).nonzero().flatten()
        self.register_buffer("concrete", concrete)
        self.register_buffer("mixed_family", projection[unspecified].bool().any(0) & projection[concrete].bool().any(0))
        families = projection.shape[1]
        self.keys = nn.Linear(dim, dim, bias=False)
        self.family_queries = nn.Parameter(torch.randn(families, dim) * .02)
        self.family_weights = nn.Parameter(torch.randn(families, dim) / math.sqrt(dim))
        self.family_bias = nn.Parameter(prior_bias(family_statistics))
        self.detail = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        if mode == "full":
            self.fine_queries = nn.Parameter(torch.randn(len(concrete), dim) * .02)
            self.condition = nn.Linear(dim, dim, bias=False)
            self.fine_evidence = nn.Sequential(nn.Linear(5 * dim + 1, dim), nn.GELU(), nn.Dropout(dropout))
            self.fine_weights = nn.Parameter(torch.zeros(num_labels, dim))

    def encode(self, batch):
        mask = batch["mask"]
        width = mask.shape[1]
        position = torch.arange(width, device=mask.device)
        terminal = batch.get("terminal_chemical")
        x = self.cnn.residue_inputs(batch)
        if terminal is not None:
            last = (mask.sum(1, keepdim=True) - 1) == position
            endpoints = torch.stack((mask & (position == 0), mask & last), -1)
            x = x + self.terminal((endpoints & terminal[:, None].bool()).to(x.dtype))
        maps = self.cnn.residue_maps(batch, x)
        positions = self.position(endpoint_positions(mask))
        residues = positions if self.mode == "theraset_direct" else self.local(torch.cat(maps, -1)) + positions
        if self.peptide is not None:
            residues = residues + self.peptide(batch["esm"])
        residues, global_state = self.summarize(residues, mask)
        if self.mode == "theraset_direct":
            global_state = self.cnn.pool_maps(maps, mask) + self.context_summary(global_state)
        return residues, global_state

    def summarize(self, residues, mask):
        residues = self.norm(residues).masked_fill(~mask[..., None], 0.)
        summary = torch.cat((masked_mean(residues, mask),
                             residues.masked_fill(~mask[..., None], -torch.inf).amax(1)), -1)
        return residues, summary

    def read(self, queries, keys, residues, mask):
        scores = (queries @ keys.transpose(-1, -2)).float() / math.sqrt(keys.shape[-1])
        scores = scores.masked_fill(~mask[:, None], -torch.inf)
        return scores.softmax(-1).to(residues.dtype) @ residues

    def forward(self, batch):
        if self.mode in HH_V20_CHEMICAL_MODES:
            _, pooled = self.cnn(batch)
            if self.mode == "chemical_cnn_fusion":
                mask = batch["mask"]
                residues = self.peptide(batch["esm"]) + self.position(endpoint_positions(mask))
                _, summary = self.summarize(residues, mask)
                pooled = pooled + self.context_summary(summary)
            return {"logits": self.direct(pooled)}
        residues, global_state = self.encode(batch)
        logits = self.direct(global_state)
        if self.mode in ("direct", "theraset_direct"):
            return {"logits": logits}
        mask = batch["mask"]
        keys = self.keys(residues)
        family = self.read(self.family_queries, keys, residues, mask)
        family_logits = (family * self.family_weights).sum(-1) + self.family_bias
        detail = self.detail(torch.cat((global_state[:, None].expand(-1, family.shape[1], -1), family), -1)).squeeze(-1)
        if self.mode == "full":
            family_of_label = family[:, self.family_of_label]
            chosen = family_of_label[:, self.concrete]
            query = self.fine_queries + self.condition(chosen)
            local = self.read(query, keys, residues, mask)
            # Broad labels read family evidence; concrete labels can refine it.
            fine = family_of_label.index_copy(1, self.concrete, local)
            detail_signal = (detail * self.mixed_family)[:, self.family_of_label, None]
            features = torch.cat((global_state[:, None].expand(-1, logits.shape[-1], -1),
                                  family_of_label, fine, fine - family_of_label, detail_signal), -1)
            logits = logits + (self.fine_evidence(features) * self.fine_weights).sum(-1)
        return {"logits": logits, "family_logits": family_logits, "detail_logits": detail}

    def supervision(self, target, major, major_mask):
        fine_families = (target @ self.projection) > 0
        family_targets = fine_families | (major.bool() & major_mask[:, None])
        specific = (target[:, self.concrete] @ self.projection[self.concrete]) > 0
        return family_targets, specific, fine_families & self.mixed_family

    def compute_loss(self, batch, settings, return_output=False):
        output = self(batch)
        target = batch["y"]
        weak = settings.get("unlabeled_weight", .3)
        classification = observed_label_loss(output["logits"], target, self.statistics, weak)
        terms = {"classification": classification}
        loss = classification
        if self.mode in ("auxiliary", "full"):
            family_targets, specific, mask = self.supervision(target, batch["aux_y"], batch["aux_mask"])
            family_loss = observed_label_loss(output["family_logits"], family_targets.float(), self.family_statistics, weak)
            detail = F.binary_cross_entropy_with_logits(output["detail_logits"].float(), specific.float(), reduction="none")
            detail = (detail * mask).sum() / mask.sum().clamp_min(1)
            terms.update(family=family_loss, annotation_detail=detail)
            loss = loss + settings["family_weight"] * family_loss + settings["detail_weight"] * detail
        return (loss, terms, output) if return_output else (loss, terms)
