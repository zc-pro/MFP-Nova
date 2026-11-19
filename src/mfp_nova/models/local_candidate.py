"""An independent Fine39 candidate head over cached peptide PLM residues."""

import math

import torch
from torch import nn

from ..research_layers import FeedForward
from .helixset import MotifPyramid, QualityRead


class LocalCandidateHead(nn.Module):
    def __init__(self, input_dim, num_labels, mode="span", dim=96, dropout=.1):
        super().__init__()
        if mode not in ("pool", "span"):
            raise ValueError("Expected a pooled or multiscale candidate head")
        self.mode = mode
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU())
        if mode == "pool":
            self.classify = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(dim, num_labels))
        else:
            self.position = nn.Linear(2, dim, bias=False)
            self.pyramid = MotifPyramid(dim, dropout)
            self.queries = nn.Parameter(torch.randn(num_labels, dim) * .02)
            self.read = QualityRead(dim, 4, dropout)
            self.update = FeedForward(dim, 2, dropout)
            self.norm = nn.LayerNorm(dim)
            self.output = nn.Parameter(torch.randn(num_labels, dim) / math.sqrt(dim))
            self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, residues, mask):
        length = mask.sum(1, keepdim=True)
        h = self.project(residues)
        if self.mode == "pool":
            maximum = h.masked_fill(~mask[..., None], float("-inf")).max(1).values
            average = (h * mask[..., None]).sum(1) / length
            return self.classify(torch.cat((maximum, average), dim=-1))

        # Relative position lets the same window contribute differently near each terminus.
        position = torch.arange(h.size(1), device=h.device)[None] / (length - 1).clamp_min(1)
        h = (h + self.position(torch.stack((position, 1 - position), dim=-1))) * mask[..., None]
        memory, valid, scale_weight = self.pyramid(h, mask)
        query = self.queries[None].expand(h.size(0), -1, -1)
        query_mask = torch.ones(query.shape[:2], device=h.device, dtype=torch.bool)
        evidence = self.read(query, memory, query_mask, valid, scale_weight)
        state = self.update(query + evidence, query_mask)
        return (self.norm(state) * self.output).sum(-1) + self.bias
