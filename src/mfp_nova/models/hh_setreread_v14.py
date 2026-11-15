"""Separate the decision to add a second axis from Fine39 candidate ranking."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..multimodal import function_projection
from .theraset import TheraSet


def apply_second_axis(unary, candidates, row_scores, label_scores, threshold):
    base = unary >= 0
    best = label_scores.masked_fill(~candidates, -torch.inf).argmax(-1)
    chosen = candidates.any(-1) & (row_scores > (math.inf if threshold is None else threshold))
    added = F.one_hot(best, unary.shape[-1]).bool() & chosen[:, None]
    return base | added, best.masked_fill(~chosen, -1)


def signed_log(value):
    return value.sign() * value.abs().log1p()


class ResidueQueries(nn.Module):
    def __init__(self, labels, dim):
        super().__init__()
        self.embedding = nn.Linear(1280+640, dim, bias=False)
        self.keys = nn.Linear(dim, dim, bias=False)
        self.query = nn.Parameter(torch.randn(labels, dim) * .02)
        self.null = nn.Parameter(torch.zeros(labels))

    def encode(self, batch):
        views = [F.layer_norm(batch[key].float(), (batch[key].shape[-1],)) for key in ("esm", "esm_peptide")]
        residues = self.embedding(torch.cat(views, -1)).masked_fill(~batch["mask"][..., None], 0.)
        return residues, self.keys(residues)

    def attention(self, queries, keys, mask):
        scores = (queries @ keys.transpose(-1, -2)) / math.sqrt(keys.shape[-1])
        scores = scores.masked_fill(~mask[:, None], -torch.inf)
        null = mask.sum(-1, keepdim=True).clamp_min(1).float().log() + self.null
        return torch.cat((scores, null[..., None]), -1).softmax(-1)[..., :-1]


class RowHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.reader = ResidueQueries(15, 64) if mode == "evidence" else None
        if self.reader is not None:
            self.fuse = nn.Sequential(nn.Linear(3*64+2, 32), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(114, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, batch, unary, base, anchor, anchor_score, membership, candidates):
        per_axis = (candidates[:, None] & membership).any(-1)
        best = unary[:, None].masked_fill(~(candidates[:, None] & membership), -torch.inf).amax(-1)
        best = best.masked_fill(~per_axis, 0.)
        strongest = unary.masked_fill(~candidates, -torch.inf).amax(-1)
        strongest = strongest.masked_fill(~candidates.any(-1), 0.)
        scalars = torch.stack((signed_log(anchor_score), signed_log(strongest), signed_log(anchor_score-strongest),
                               base.sum(-1)/39, batch["mask"].sum(-1)/80), -1)
        features = torch.cat((signed_log(best), per_axis.float(), F.one_hot(anchor, 15).float(), scalars), -1)
        if self.reader is None:
            evidence = unary.new_zeros((len(unary), 64))
        else:
            residues, keys = self.reader.encode(batch)
            pooled = self.reader.attention(self.reader.query, keys, batch["mask"]) @ residues
            anchor_evidence = pooled[torch.arange(len(unary), device=unary.device), anchor]
            anchor_evidence = anchor_evidence[:, None].expand_as(pooled)
            z = self.fuse(torch.cat((anchor_evidence, pooled, anchor_evidence*pooled,
                                     signed_log(anchor_score)[:, None, None].expand(-1, 15, 1),
                                     signed_log(best)[..., None]), -1))
            mask = per_axis[..., None]
            largest = z.masked_fill(~mask, -torch.inf).amax(1).masked_fill(~per_axis.any(1, keepdim=True), 0.)
            average = (z*mask).sum(1) / per_axis.sum(1, keepdim=True).clamp_min(1)
            evidence = torch.cat((largest, average), -1)
        return self.mlp(torch.cat((features, evidence), -1)).squeeze(-1)


class LabelHead(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.reader = ResidueQueries(39, 96)
        self.axis_query = nn.Parameter(torch.randn(15, 96) * .02)
        self.context = nn.Linear(96, 96, bias=False) if mode == "conditional" else None
        self.mlp = nn.Sequential(nn.Linear(483, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, batch, unary, axis, anchor_label, without_background=False):
        residues, keys = self.reader.encode(batch)
        queries = self.reader.query + self.axis_query[axis]
        attention = self.reader.attention(queries, keys, batch["mask"])
        first = attention @ residues
        shared = context = torch.zeros_like(first)
        remaining = first
        rows = torch.arange(len(unary), device=unary.device)
        if self.context is not None and not without_background:
            anchor_attention = attention[rows, anchor_label].detach()[:, None]
            context = first[rows, anchor_label].detach()[:, None]
            reread = self.reader.attention(queries + self.context(context), keys, batch["mask"])
            overlap = torch.minimum(reread, anchor_attention)
            shared, remaining = overlap @ residues, (reread-overlap) @ residues
            context = context.expand_as(first)
        anchor_score = unary[rows, anchor_label, None].expand_as(unary)
        scalars = signed_log(torch.stack((unary, anchor_score, unary-anchor_score), -1))
        features = torch.cat((queries.expand_as(first), first, shared, remaining, context, scalars), -1)
        return self.mlp(features).squeeze(-1)


class HHSetRereadV14(nn.Module):
    def __init__(self, statistics, row_mode="evidence", label_mode="conditional", backbone_args=None):
        super().__init__()
        if len(statistics["labels"]) != 39:
            raise ValueError("HH V1.4 requires Fine39")
        if row_mode not in ("scores", "evidence") or label_mode not in ("raw", "unconditional", "conditional"):
            raise ValueError("Unknown HH V1.4 row or label mode")
        self.statistics = statistics
        projection = torch.tensor(function_projection(statistics["labels"], "fine39"))
        self.register_buffer("membership", projection.T.bool())
        self.register_buffer("axis_index", projection.argmax(-1))
        self.backbone = None if backbone_args is None else TheraSet(**backbone_args).requires_grad_(False).eval()
        self.row_head = RowHead(row_mode)
        self.label_head = None if label_mode == "raw" else LabelHead(label_mode)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone is not None:
            self.backbone.eval()
        return self

    @staticmethod
    def effective(raw):
        generic = raw[..., 4:5] & ~raw[..., :4].any(-1, keepdim=True)
        return torch.cat((raw[..., :4], generic, raw[..., 5:]), -1)

    def raw_axes(self, labels):
        return (labels[:, None] & self.membership).any(-1)

    def candidates(self, unary):
        base = unary >= 0
        raw = self.raw_axes(base)
        axes = self.effective(raw)
        after = self.effective(raw[:, None] | self.membership.T)
        mask = (~base & (axes.sum(-1, keepdim=True) == 1) & (after.sum(-1) == 2)
                & ~(axes[:, None] & ~after).any(-1))
        return mask, axes, after

    def supervision(self, unary, targets):
        mask, axes, after = self.candidates(unary)
        truth_axes = self.effective(self.raw_axes(targets.bool()))
        exact = mask & targets.bool() & ~(after & ~truth_axes[:, None]).any(-1)
        axis_hit = mask & ~(after & ~truth_axes[:, None]).any(-1)
        return mask, exact, axis_hit, (axes & truth_axes).any(-1), truth_axes.sum(-1) >= 2

    def forward(self, batch, branch=None, without_background=False):
        if "base_logits" in batch:
            unary = batch["base_logits"].float()
        else:
            with torch.no_grad():
                unary = self.backbone(batch)["logits"].float()
        candidates, axes, _ = self.candidates(unary)
        base = unary >= 0
        anchor = axes.long().argmax(-1)
        support = base & self.membership[anchor]
        anchor_label = unary.masked_fill(~support, -torch.inf).argmax(-1)
        anchor_score = unary.gather(1, anchor_label[:, None]).squeeze(-1)
        output = {"logits": unary, "candidate_mask": candidates}
        if branch != "label":
            output["row_logit"] = self.row_head(batch, unary, base, anchor, anchor_score,
                                                self.membership, candidates)
        if branch != "row":
            output["label_logit"] = (unary if self.label_head is None else self.label_head(
                batch, unary, self.axis_index, anchor_label, without_background=without_background))
        return output

    def branch_loss(self, batch, branch):
        out = self(batch, branch=branch)
        mask, exact, _, _, _ = self.supervision(out["logits"], batch["y"])
        if branch == "row":
            return F.binary_cross_entropy_with_logits(out["row_logit"], exact.any(-1).float())
        loss = F.binary_cross_entropy_with_logits(out["label_logit"], exact.float(), reduction="none")
        return ((loss * mask).sum(-1) / mask.sum(-1).clamp_min(1)).mean()
