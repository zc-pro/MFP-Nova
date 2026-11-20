"""Fine39 classification with a single label readout over four evidence memories."""

import math

import torch
from torch import nn

from ..research_layers import Attention, training_views
from .helixset import HelixSet
from .sequence_features import ResidueCNN, prior_bias, supervised_loss


class UnifiedEvidence(HelixSet):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=384,
                 heads=8, dropout=.2, **backbone):
        super().__init__(input_dim, num_labels, vocab_size, statistics, dim=dim,
                         heads=heads, dropout=dropout, latent_memory_only=True, **backbone)
        self.local = ResidueCNN(dropout, channels=128)
        self.local_project = nn.Linear(512, dim)
        self.queries = nn.Parameter(torch.randn(num_labels, dim) * .02)
        self.readouts = nn.ModuleList(Attention(dim, heads, dropout) for _ in range(4))
        self.output_norm = nn.LayerNorm(dim)
        self.label_weights = nn.Parameter(torch.randn(num_labels, dim) / math.sqrt(dim))
        self.label_bias = nn.Parameter(prior_bias(statistics))

    def forward(self, batch):
        residues, properties, property_mask, latent, latent_mask = super().forward(batch)
        mask = batch["mask"]
        local = self.local(batch)
        present = mask.any(dim=1, keepdim=True)
        pooled = local.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        pooled = pooled.masked_fill(~present, 0)
        local = self.local_project(torch.cat((local, pooled[:, None]), dim=1))
        memories = (residues, local, properties, latent)
        masks = (mask, torch.cat((mask, present), dim=1), property_mask, latent_mask)
        queries = self.queries[None].expand(mask.shape[0], -1, -1)
        query_mask = torch.ones(queries.shape[:2], dtype=torch.bool, device=queries.device)
        evidence = sum(read(queries, memory, query_mask, valid)
                       for read, memory, valid in zip(self.readouts, memories, masks))
        logits = (self.output_norm(evidence) * self.label_weights).sum(dim=-1) + self.label_bias
        return {"logits": logits}

    def compute_loss(self, batch, settings, return_output=False):
        view, _ = training_views(batch, settings.get("modality_dropout", .1), 0.)
        return supervised_loss(self, view, settings, return_output)
