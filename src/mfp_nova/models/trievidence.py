"""TriEvidence-MFP: independent Fine39 readers with fixed late logit fusion."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..research_layers import residue_distances
from .helixset import HelixSet
from .sequence_features import ResidueCNN, prior_bias
from .theraset import TheraSet, motif_classifier


LOGIT_LIMIT = math.log((1 - 1e-6) / 1e-6)
READER_HEADS = ("global", "local", "contrast", "dual")


class LocalLabelReadout(nn.Module):
    """Contrast the strongest residue neighborhoods with the peptide's own background."""

    def __init__(self, input_dim, num_labels, subtract_background):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 64), nn.GELU())
        self.labels = nn.Linear(64, num_labels, bias=False)
        self.subtract_background = subtract_background

    def forward(self, maps, mask):
        scores = self.labels(self.project(maps)).float()
        valid = mask[..., None]
        length = valid.sum(1).clamp_min(1)
        top = scores.masked_fill(~valid, -torch.inf).topk(min(3, scores.shape[1]), dim=1).values
        count = ((length + 1) // 2).clamp_max(3)
        rank = torch.arange(top.shape[1], device=maps.device)[None, :, None]
        peak = top.masked_fill(rank >= count[:, None], 0).sum(1) / count
        if self.subtract_background:
            background = scores.masked_fill(~valid, 0).sum(1) / length
            return peak - background
        return peak


class AA34Reader(nn.Module):
    def __init__(self, statistics, channels=128, dropout=.2, reader_head="global"):
        super().__init__()
        if reader_head not in READER_HEADS:
            raise ValueError(f"Unknown AA34 reader head: {reader_head}")
        self.motifs = ResidueCNN(dropout, channels)
        num_labels = len(statistics["labels"])
        self.classifier = motif_classifier(4 * channels, num_labels, dropout)
        with torch.no_grad():
            self.classifier[-1].bias.copy_(prior_bias(statistics))
        if reader_head == "dual":
            self.review_classifier = motif_classifier(4 * channels, num_labels, dropout)
            with torch.no_grad():
                self.review_classifier[-1].bias.copy_(self.classifier[-1].bias)
        elif reader_head != "global":
            with torch.random.fork_rng(devices=[]):
                self.local = LocalLabelReadout(4 * channels, num_labels, reader_head == "contrast")
            self.local_gain = nn.Parameter(torch.full((num_labels,), .1))

    def forward(self, batch, return_heads=False):
        maps = self.motifs(batch)
        pooled = maps.masked_fill(~batch["mask"][..., None], -torch.inf).amax(1)
        logits = self.classifier(pooled)
        if hasattr(self, "review_classifier"):
            review = self.review_classifier(pooled)
            if return_heads:
                return logits, review
            return (logits + review) * .5
        if hasattr(self, "local"):
            logits = logits + self.local_gain * self.local(maps, batch["mask"])
        return logits


def center_evidence(logits, offsets):
    # Use the historical probability clipping in the fixed-fusion experiment.
    return logits.float().clamp(-LOGIT_LIMIT, LOGIT_LIMIT) - offsets[:, None, None]


def fuse_evidence(theraset, aa34, latent, offsets):
    return center_evidence(torch.stack((theraset, aa34, latent)), offsets).mean(0)


def empty_top10(logits, theraset_logits):
    raw = logits >= 0
    candidates = theraset_logits.argsort(dim=-1, descending=True, stable=True)[:, :10]
    top = logits.gather(1, candidates)
    # Historical ties favor ranks 1–5, then the earlier label in the label table.
    num_labels = logits.shape[1]
    low_rank = torch.arange(candidates.shape[1], device=logits.device) >= 5
    priority = candidates + num_labels * low_rank
    label = priority.masked_fill(top != top.amax(1, keepdim=True), 2 * num_labels).amin(1, keepdim=True) % num_labels
    return raw | torch.zeros_like(raw).scatter_(1, label, ~raw.any(1, keepdim=True))


class TriEvidence(nn.Module):
    def __init__(self, theraset_args, helixset_args, cnn_channels=128, cnn_dropout=.2, reader_head="global"):
        super().__init__()
        # The frozen branches are loaded from checkpoints. Their initialization
        # should not affect the independently trained CNN's random stream.
        with torch.random.fork_rng(devices=[]):
            self.theraset = TheraSet(**theraset_args).requires_grad_(False).eval()
            self.helixset = HelixSet(**helixset_args, direct_only=True).requires_grad_(False).eval()
        self.aa34 = AA34Reader(theraset_args["statistics"], cnn_channels, cnn_dropout, reader_head)
        self.register_buffer("offsets", torch.zeros(3))
        self.reuse_input_compute = False
        self.register_buffer("_packed_esm_weight", None, persistent=False)
        self.register_buffer("_packed_esm_bias", None, persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.theraset.eval()
        self.helixset.eval()
        return self

    def _can_share_esm(self):
        first = getattr(self.theraset, "sequence", None)
        second = self.helixset.sequence
        return (hasattr(first, "peptide_norm") and hasattr(second, "peptide_norm")
                and self.helixset.esm_tail is None
                and first.plm_norm.normalized_shape == second.plm_norm.normalized_shape
                and first.peptide_norm.normalized_shape == second.peptide_norm.normalized_shape)

    @torch.no_grad()
    def pack_esm_projections(self):
        if not self._can_share_esm():
            raise ValueError("Packing requires the same frozen dual-ESM inputs for both encoders")
        first = self.theraset.sequence.plm
        second = self.helixset.sequence.plm
        self._packed_esm_weight = torch.cat((first.weight, second.weight))
        self._packed_esm_bias = torch.cat((first.bias, second.bias))
        self.reuse_input_compute = True

    def anchor_logits(self, batch):
        if self.reuse_input_compute:
            batch = dict(batch)
            if self._can_share_esm():
                batch["_trievidence_dual_esm"] = self.theraset.sequence.normalized_dual_esm(batch)
            if self.theraset.structure_evidence and self.helixset.structure_evidence:
                structural = batch["struct_mask"] & batch["mask"]
                batch["_trievidence_distances"] = residue_distances(batch, structural)
            if self._packed_esm_weight is not None and not self.training:
                projected = F.linear(batch["_trievidence_dual_esm"], self._packed_esm_weight, self._packed_esm_bias)
                first, second = projected.split((self.theraset.sequence.plm.out_features,
                                                 self.helixset.sequence.plm.out_features), dim=-1)
                return (self.theraset(dict(batch, _trievidence_esm_projection=first))["logits"],
                        self.helixset(dict(batch, _trievidence_esm_projection=second))["logits"])
        return self.theraset(batch)["logits"], self.helixset(batch)["logits"]

    def forward(self, batch):
        theraset, latent = self.anchor_logits(batch)
        aa34 = self.aa34(batch)
        logits = fuse_evidence(theraset, aa34, latent, self.offsets)
        return {"logits": logits, "theraset_logits": theraset, "aa34_logits": aa34, "latent_logits": latent,
                "predicted": empty_top10(logits, theraset)}
