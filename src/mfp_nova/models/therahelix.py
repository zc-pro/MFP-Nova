"""One multimodal encoder with local and relational reads for each label."""

import math

import torch
from torch import nn

from ..research_layers import Attention, training_views
from .helixset import HelixSet, MotifPyramid
from .sequence_features import supervised_loss


class TheraHelix(HelixSet):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=384,
                 heads=8, dropout=.2, **backbone):
        super().__init__(input_dim, num_labels, vocab_size, statistics, dim=dim,
                         heads=heads, dropout=dropout, direct_only=True, **backbone)
        del self.latent_classifier
        self.local_windows = MotifPyramid(dim, dropout)
        self.queries = nn.Parameter(torch.randn(num_labels, dim) * .02)
        self.read_local = Attention(dim, heads, dropout)
        self.read_relations = Attention(dim, heads, dropout)
        self.output_norm = nn.LayerNorm(3 * dim)
        self.label_weights = nn.Parameter(torch.randn(num_labels, 3 * dim) / math.sqrt(3 * dim))
        self.label_bias = nn.Parameter(torch.tensor(statistics["observed_prior"]).clamp(1e-4, 1 - 1e-4).logit())

    def read_latent(self, latent, sequence, sequence_mask):
        # Use residues before geometry mixing and pair updates for the local read.
        local, local_mask, scale_mass = self.local_windows(sequence[:, :-1], sequence_mask[:, :-1])
        local = torch.cat([local, sequence[:, -1:]], dim=1)
        local_mask = torch.cat([local_mask, sequence_mask[:, -1:]], dim=1)
        scale_mass = torch.cat([scale_mass, sequence_mask[:, -1:].float()], dim=1)
        query = self.queries[None].expand(sequence.size(0), -1, -1)
        query_mask = torch.ones(query.shape[:2], device=query.device, dtype=torch.bool)
        latent_mask = torch.ones(latent.shape[:2], device=query.device, dtype=torch.bool)
        local_bias = scale_mass.clamp_min(1e-8).log()[:, None, None, :]
        local_bias = local_bias.expand(-1, self.read_local.heads, query.size(1), -1)
        local_read = self.read_local(query, local, query_mask, local_mask, bias=local_bias)
        relation_read = self.read_relations(query, latent, query_mask, latent_mask)
        combined = self.output_norm(torch.cat([query, local_read, relation_read], dim=-1))
        return (combined * self.label_weights).sum(-1) + self.label_bias

    def compute_loss(self, batch, settings, return_output=False):
        view, _ = training_views(batch, settings.get("modality_dropout", .1), 0.)
        return supervised_loss(self, view, settings, return_output)
