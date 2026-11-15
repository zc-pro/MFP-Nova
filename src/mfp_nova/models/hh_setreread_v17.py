"""Fine39 label evidence and reference-relative set decisions."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..neighbor_evidence import neighbor_tensors
from .hh_setreread_v14 import ResidueQueries, signed_log
from .theraset import TheraSet


ACTIONS = ("add", "drop", "swap")


def propose(base, probabilities, coexistence=None, coexistence_bonus=.05):
    """Best single action of each type using a plug-in estimate of Jaccard."""
    hits = (probabilities * base).sum(-1, keepdim=True)
    size = base.sum(-1, keepdim=True)
    union = (size + (probabilities * ~base).sum(-1, keepdim=True)).clamp_min(1e-6)
    old = hits / union
    gain_add = (hits + probabilities) / (union + 1 - probabilities) - old
    if coexistence is not None:
        conditional, eligible = coexistence
        gain_add = gain_add + coexistence_bonus * probabilities * (2 * conditional - 1) * eligible
    gain_add, incoming = gain_add.masked_fill(base | (size == 0), -torch.inf).max(-1)
    least, outgoing = probabilities.masked_fill(~base, torch.inf).min(-1)
    least = least.masked_fill(size[:, 0] == 0, 0.)
    gain_drop = ((hits[:, 0] - least) / (union[:, 0] - 1 + least).clamp_min(1e-6)
                 - old[:, 0]).masked_fill(size[:, 0] <= 1, -torch.inf)
    # For a fixed size the swap utility is strictly increasing in p(in) - p(out).
    most, replacement = probabilities.masked_fill(base, -torch.inf).max(-1)
    valid = (size[:, 0] > 0) & (size[:, 0] < base.shape[1])
    difference = (most - least).masked_fill(~valid, 0.)
    gain_swap = ((hits[:, 0] + difference) / (union[:, 0] - difference).clamp_min(1e-6)
                 - old[:, 0]).masked_fill(~valid, -torch.inf)
    return torch.stack((gain_add, gain_drop, gain_swap), -1), incoming, outgoing, replacement


def decisions(base, proposal, threshold, actions="all"):
    gains, incoming, outgoing, replacement = proposal
    if actions == "all":
        gain, kind = gains.max(-1)
    else:
        index = ACTIONS.index(actions)
        gain = gains[:, index]
        kind = torch.full_like(incoming, index)
    kind = kind.masked_fill(gain <= (math.inf if threshold is None else threshold), -1)
    added = torch.where(kind == 0, incoming, replacement).masked_fill((kind != 0) & (kind != 2), -1)
    removed = outgoing.masked_fill((kind != 1) & (kind != 2), -1)
    columns = torch.arange(base.shape[-1], device=base.device)
    final = (base & (columns != removed[:, None])) | (columns == added[:, None])
    return final, kind, added, removed


class LocalLabelReader(ResidueQueries):
    def __init__(self, labels, dim):
        super().__init__(labels, dim)
        self.residue = nn.Embedding(21, dim, padding_idx=0)
        self.windows = nn.ModuleList(
            nn.Conv1d(dim, dim, width, padding=width // 2, groups=dim) for width in (3, 7)
        )
        self.fuse = nn.Linear(dim * 3, dim)
        self.positions = nn.Linear(4, dim, bias=False)

    def forward(self, batch):
        mask = batch["mask"]
        views = [F.layer_norm(batch[key].float(), (batch[key].shape[-1],)) for key in ("esm", "esm_peptide")]
        h = (self.embedding(torch.cat(views, -1)) + self.residue(batch["aa"])).masked_fill(~mask[..., None], 0.)
        local = [layer(h.transpose(1, 2)).transpose(1, 2) for layer in self.windows]
        pos = torch.arange(h.shape[1], device=h.device)[None].expand_as(mask)
        length = mask.sum(-1, keepdim=True).clamp_min(1)
        end = length - 1 - pos
        positions = torch.stack((pos / length, end / length, pos < 3, end < 3), -1)
        tokens = F.gelu(self.fuse(torch.cat((h, *local), -1)) + self.positions(positions))
        tokens = tokens.masked_fill(~mask[..., None], 0.)
        return self.attention(self.query, self.keys(tokens), mask) @ tokens


class HHSetRereadV17(nn.Module):
    def __init__(self, function_projection, dim=64, evidence=True, neighbors=True, coexistence=True,
                 coexistence_bonus=.05, backbone_args=None):
        super().__init__()
        projection = torch.as_tensor(function_projection, dtype=torch.float32)
        self.register_buffer("projection", projection)
        self.register_buffer("neighbor_vectors", None, persistent=False)
        self.register_buffer("neighbor_labels", None, persistent=False)
        labels = len(projection)
        self.backbone = TheraSet(**backbone_args).requires_grad_(False) if backbone_args is not None else None
        self.use_neighbors = neighbors
        self.bonus = coexistence_bonus
        self.reader = LocalLabelReader(labels, dim) if evidence else None
        self.label_embedding = nn.Parameter(torch.randn(labels, dim) * .02)
        self.scale = nn.Parameter(torch.zeros(labels))
        self.bias = nn.Parameter(torch.zeros(labels))
        self.identity = nn.Sequential(nn.Linear(2 * dim + 5, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.identity[-1].weight)
        nn.init.zeros_(self.identity[-1].bias)
        self.coexist = (nn.Sequential(nn.Linear(4 * dim + 5, dim), nn.GELU(), nn.Linear(dim, 1))
                        if coexistence else None)

    @staticmethod
    def effective(raw_axes):
        return torch.cat((raw_axes[..., :4], raw_axes[..., 4:5] & ~raw_axes[..., :4].any(-1, keepdim=True),
                          raw_axes[..., 5:]), -1)

    def addition_axes(self, base):
        raw = (base.float() @ self.projection) > 0
        before = self.effective(raw)
        after = self.effective(raw[:, None] | self.projection.bool())
        new = after & ~before[:, None]
        eligible = (~base & before.any(-1, keepdim=True) & new.any(-1)
                    & ~(before[:, None] & ~after).any(-1))
        return before, new, eligible

    def neighbor_features(self, batch, unary):
        if not self.use_neighbors:
            return unary.new_zeros((*unary.shape, 3))
        if "neighbor_features" in batch:
            return batch["neighbor_features"]
        mask = batch["mask"][..., None]
        pooled = (batch["esm_peptide"].float() * mask).sum(1) / mask.sum(1).clamp_min(1)
        votes, positive, nearest = neighbor_tensors(F.normalize(pooled), self.neighbor_vectors, self.neighbor_labels)
        return torch.stack((votes, positive, nearest[:, None].expand_as(votes)), -1)

    def forward(self, batch):
        if "base_logits" in batch:
            unary = batch["base_logits"].float()
        else:
            with torch.no_grad():
                unary = self.backbone(batch)["logits"].float()
        base = unary >= 0
        evidence = self.reader(batch) if self.reader is not None else unary.new_zeros(
            (*unary.shape, self.label_embedding.shape[-1]))
        embed = self.label_embedding.expand_as(evidence)
        features = torch.cat((evidence, embed, signed_log(unary)[..., None], base[..., None].float(),
                              self.neighbor_features(batch, unary)), -1)
        logits = 2 * self.scale.sigmoid() * unary + self.bias + self.identity(features).squeeze(-1)
        output = {"base_logits": unary, "logits": logits}
        if self.coexist is not None:
            before, new, eligible = self.addition_axes(base)
            anchor = ((evidence + embed) * base[..., None]).sum(1, keepdim=True)
            anchor = anchor / base.sum(-1, keepdim=True).clamp_min(1)[..., None]
            anchor = anchor.expand_as(evidence)
            coexist = self.coexist(torch.cat((features, anchor, (evidence + embed) * anchor), -1)).squeeze(-1)
            output.update(coexist_logits=coexist, eligible=eligible, before_axes=before, new_axes=new)
        return output

    def loss(self, batch, output, coexistence_weight):
        target = batch["y"].float()
        loss = F.binary_cross_entropy_with_logits(output["logits"], target)
        if self.coexist is None:
            return loss
        axes = self.effective((target @ self.projection) > 0)
        anchor = (output["before_axes"] & axes).any(-1, keepdim=True)
        label = (output["new_axes"] & axes[:, None]).any(-1) & anchor
        mask = output["eligible"] & target.bool()
        conditional = F.binary_cross_entropy_with_logits(output["coexist_logits"], label.float(), reduction="none")
        return loss + coexistence_weight * (conditional * mask).sum() / mask.sum().clamp_min(1)

    def proposals(self, output):
        coexistence = None
        if self.coexist is not None:
            coexistence = output["coexist_logits"].sigmoid(), output["eligible"]
        return propose(output["base_logits"] >= 0, output["logits"].sigmoid(), coexistence, self.bonus)
