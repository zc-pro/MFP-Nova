"""Estimate the value of adding one Fine39 label to a fixed prediction."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..multimodal import function_projection
from .theraset import TheraSet


def addition_targets(base, targets):
    truth = targets.bool()
    hits = (base & truth).sum(-1, keepdim=True)
    union = (base | truth).sum(-1, keepdim=True).clamp_min(1).float()
    return torch.where(truth, union.reciprocal(), -hits / (union * (union + 1)))


def edit_scores(unary, gains):
    base = unary >= 0
    ranked = gains.masked_fill(base, -torch.inf)
    value, label = ranked.max(-1)
    return base, value, label


def apply_edits(unary, gains, threshold):
    base, value, label = edit_scores(unary, gains)
    selected = value > (math.inf if threshold is None else threshold)
    addition = F.one_hot(label, unary.shape[-1]).bool() & selected[:, None]
    return base | addition, torch.where(selected, label, -1)


class HHSetRereadV13(nn.Module):
    def __init__(self, statistics, input_dim=1280, peptide_input_dim=640, num_labels=39,
                 readout="conditional", backbone_args=None):
        super().__init__()
        if (num_labels, input_dim, peptide_input_dim) != (39, 1280, 640):
            raise ValueError("HH V1.3 requires Fine39 and both ESM650M/peptide150M caches")
        if readout not in ("scores", "unconditional", "conditional"):
            raise ValueError(f"Unknown HH V1.3 readout: {readout}")
        self.statistics = statistics
        self.readout = readout
        self.backbone = None
        if backbone_args is not None:
            self.backbone = TheraSet(**backbone_args).requires_grad_(False).eval()
        projection = torch.tensor(function_projection(statistics["labels"], "fine39"))
        axis_index = projection.argmax(-1)
        axis = torch.arange(projection.shape[1])
        related = axis_index[:, None] == axis
        related |= ((axis_index[:, None] == 4) & (axis < 4)) | ((axis_index[:, None] < 4) & (axis == 4))
        self.register_buffer("membership", projection.T.bool())
        self.register_buffer("axis_index", axis_index)
        self.register_buffer("background_axes", ~related)

        dim = 96
        self.fine_queries = nn.Parameter(torch.randn(num_labels, dim) * .02)
        self.axis_queries = nn.Parameter(torch.randn(len(axis), dim) * .02)
        if readout != "scores":
            self.projection = nn.Linear(input_dim + peptide_input_dim, dim, bias=False)
            self.keys = nn.Linear(dim, dim, bias=False)
            self.null = nn.Parameter(torch.zeros(num_labels))
        if readout == "conditional":
            self.context = nn.Linear(dim, dim, bias=False)
        self.readout_head = nn.Sequential(nn.Linear(5*dim+6, 64), nn.GELU(), nn.Dropout(.1),
                                          nn.Linear(64, 1), nn.Tanh())
        nn.init.zeros_(self.readout_head[-2].weight)
        nn.init.zeros_(self.readout_head[-2].bias)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    def attention(self, queries, keys, mask):
        logits = (queries @ keys.transpose(-1, -2) / math.sqrt(keys.shape[-1])).float()
        logits = logits.masked_fill(~mask[:, None], -torch.inf)
        null = mask.sum(-1, keepdim=True).clamp_min(1).float().log() + self.null
        return torch.cat((logits, null[..., None]), -1).softmax(-1)[..., :-1]

    def axes(self, base):
        active = (base[:, None] & self.membership).any(-1)
        active[:, 4] &= ~active[:, :4].any(-1)
        return active

    @torch.no_grad()
    def background(self, unary, active, evidence, attention):
        representative = unary[:, None].masked_fill(~self.membership, -torch.inf).argmax(-1)
        rows = torch.arange(len(unary), device=unary.device)[:, None]
        available = self.background_axes & active[:, None]
        weights = available.float() / available.sum(-1, keepdim=True).clamp_min(1)
        return weights @ evidence[rows, representative], weights @ attention[rows, representative]

    def forward(self, batch, without_background=False):
        if "base_logits" in batch:
            unary = batch["base_logits"].float()
        else:
            with torch.no_grad():
                unary = self.backbone(batch)["logits"].float()
        mask = batch["mask"]
        base = unary >= 0
        active = self.axes(base)
        order = unary.masked_fill(base, -torch.inf).argsort(descending=True, stable=True)
        rank = order.argsort().float() / ((~base).sum(-1, keepdim=True)-1).clamp_min(1)
        scaled = unary.sign() * unary.abs().log1p()
        shared_features = torch.stack((base.sum(-1)/39, active.sum(-1)/15,
                                       scaled.amax(-1), mask.sum(-1)/80), -1)
        features = torch.cat((scaled[..., None], rank[..., None],
                              shared_features[:, None].expand(-1, unary.shape[-1], -1)), -1)
        queries = self.axis_queries[self.axis_index] + self.fine_queries
        queries = queries.expand(len(unary), -1, -1)
        zero = torch.zeros_like(queries)
        first, shared, remaining, context = zero, zero, zero, zero
        if self.readout != "scores":
            residues = torch.cat([F.layer_norm(batch[key].float(), (batch[key].shape[-1],))
                                  for key in ("esm", "esm_peptide")], -1)
            residues = self.projection(residues).float().masked_fill(~mask[..., None], 0.)
            keys = self.keys(residues)
            attention = self.attention(queries, keys, mask)
            first = remaining = attention @ residues
            if self.readout == "conditional" and not without_background:
                context, occupancy = self.background(unary, active, first, attention)
                reread = self.attention(queries + self.context(context), keys, mask)
                overlap = torch.minimum(reread, occupancy)
                shared = overlap @ residues
                remaining = (reread - overlap) @ residues
        gains = self.readout_head(torch.cat((queries, first, shared, remaining, context, features), -1))
        return {"logits": unary, "edit_gain": gains.squeeze(-1).float()}

    def compute_loss(self, batch):
        output = self(batch)
        base = output["logits"] >= 0
        error = (output["edit_gain"] - addition_targets(base, batch["y"])).square()
        return ((error * ~base).sum(-1) / (~base).sum(-1).clamp_min(1)).mean()
