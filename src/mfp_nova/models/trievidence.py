"""TriEvidence-MFP: independent Fine39 readers with fixed late logit fusion."""

import math

import torch
from torch import nn

from .helixset import HelixSet
from .sequence_features import ResidueCNN, prior_bias
from .theraset import TheraSet, motif_classifier


LOGIT_LIMIT = math.log((1 - 1e-6) / 1e-6)


class AA34Reader(nn.Module):
    def __init__(self, statistics, channels=128, dropout=.2):
        super().__init__()
        self.motifs = ResidueCNN(dropout, channels)
        self.classifier = motif_classifier(4 * channels, len(statistics["labels"]), dropout)
        with torch.no_grad():
            self.classifier[-1].bias.copy_(prior_bias(statistics))

    def forward(self, batch):
        maps = self.motifs(batch)
        pooled = maps.masked_fill(~batch["mask"][..., None], -torch.inf).amax(1)
        return self.classifier(pooled)


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
    def __init__(self, theraset_args, helixset_args, cnn_channels=128, cnn_dropout=.2):
        super().__init__()
        # The frozen branches are loaded from checkpoints. Their initialization
        # should not affect the independently trained CNN's random stream.
        with torch.random.fork_rng(devices=[]):
            self.theraset = TheraSet(**theraset_args).requires_grad_(False).eval()
            self.helixset = HelixSet(**helixset_args, direct_only=True).requires_grad_(False).eval()
        self.aa34 = AA34Reader(theraset_args["statistics"], cnn_channels, cnn_dropout)
        self.register_buffer("offsets", torch.zeros(3))

    def train(self, mode=True):
        super().train(mode)
        self.theraset.eval()
        self.helixset.eval()
        return self

    def anchor_logits(self, batch):
        return self.theraset(batch)["logits"], self.helixset(batch)["logits"]

    def forward(self, batch):
        theraset, latent = self.anchor_logits(batch)
        aa34 = self.aa34(batch)
        logits = fuse_evidence(theraset, aa34, latent, self.offsets)
        return {"logits": logits, "theraset_logits": theraset, "aa34_logits": aa34, "latent_logits": latent,
                "predicted": empty_top10(logits, theraset)}
