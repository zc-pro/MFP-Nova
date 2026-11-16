"""V3.0 sequence CNN with directed interactions over residue distance bands."""

import torch
from torch import nn
from torch.nn import functional as F

from . import V3_MODES
from .sequence_features import prior_bias, supervised_loss
from .theraset import MotifCNN, motif_classifier


DISTANCE_BANDS = ((1, 2), (3, 5), (6, 9), (10, 16))


def interval_sum(prefix, positions, start, stop):
    last = prefix.shape[1] - 1
    return prefix[:, (positions + stop).clamp(0, last)] - prefix[:, (positions + start).clamp(0, last)]


class OrderedInteractions(nn.Module):
    def __init__(self, local_dim, output_dim, rank, mode):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(local_dim)
        self.left = nn.Linear(local_dim, rank, bias=False)
        self.right = nn.Linear(local_dim, rank, bias=False)
        self.band_weight = nn.Parameter(torch.ones(2, len(DISTANCE_BANDS), rank))
        self.write_left = nn.Linear(rank, output_dim, bias=False)
        self.write_right = nn.Linear(rank, output_dim, bias=False)
        nn.init.zeros_(self.write_left.weight)
        nn.init.zeros_(self.write_right.weight)

    def messages(self, left, right, mask):
        """Return endpoint messages in FP32, using only valid residues and edges."""
        valid = mask.unsqueeze(-1)
        positions = torch.arange(mask.shape[1], device=mask.device)
        left, right = left.float(), right.float()
        left_prefix = F.pad(torch.cat((left, valid), -1).masked_fill(~valid, 0).cumsum(1), (0, 0, 1, 0))
        right_prefix = F.pad(torch.cat((right, valid), -1).masked_fill(~valid, 0).cumsum(1), (0, 0, 1, 0))
        left_messages, right_messages = torch.zeros_like(left), torch.zeros_like(right)
        left_bands, right_bands = torch.zeros_like(left[..., :1]), torch.zeros_like(right[..., :1])

        for b, (low, high) in enumerate(DISTANCE_BANDS):
            right_neighbors = interval_sum(right_prefix, positions, low, high + 1)
            left_neighbors = interval_sum(left_prefix, positions, -high, -low + 1)
            out_degree = right_neighbors[..., -1:] * valid
            in_degree = left_neighbors[..., -1:] * valid
            right_mean = right_neighbors[..., :-1] / out_degree.clamp_min(1)
            left_mean = left_neighbors[..., :-1] / in_degree.clamp_min(1)
            if self.mode != "product":
                edges = out_degree.sum(dim=1, keepdim=True).clamp_min(1)
                left_center = (left * out_degree).sum(dim=1, keepdim=True) / edges
                right_center = (right * in_degree).sum(dim=1, keepdim=True) / edges
            if self.mode == "additive":
                at_left = (left - left_center) + (right_mean - right_center)
                at_right = (left_mean - left_center) + (right - right_center)
            elif self.mode == "centered":
                at_left = (left - left_center) * (right_mean - right_center)
                at_right = (left_mean - left_center) * (right - right_center)
            else:
                at_left = left * right_mean
                at_right = left_mean * right
            left_available = out_degree > 0
            right_available = in_degree > 0
            left_messages = left_messages + at_left.masked_fill(~left_available, 0) * self.band_weight[0, b]
            right_messages = right_messages + at_right.masked_fill(~right_available, 0) * self.band_weight[1, b]
            left_bands = left_bands + left_available
            right_bands = right_bands + right_available

        return left_messages / left_bands.clamp_min(1), right_messages / right_bands.clamp_min(1)

    def forward(self, features, mask):
        local = self.norm(features)
        left, right = self.messages(self.left(local).tanh(), self.right(local).tanh(), mask)
        return self.write_left(left) + self.write_right(right)


class PeptideV3(nn.Module):
    def __init__(self, num_labels, statistics, cnn_channels=128, relation_rank=32, dropout=.2, mode="centered"):
        super().__init__()
        if mode not in V3_MODES:
            raise ValueError(f"Unknown V3 mode: {mode}")
        self.statistics = statistics
        self.cnn = MotifCNN(2, cnn_channels, dropout, chemical_evidence=False)
        self.cnn.d.requires_grad_(False)
        self.cnn.chem.requires_grad_(False)
        self.direct = motif_classifier(self.cnn.output_channels, num_labels, dropout)
        with torch.no_grad():
            self.direct[-1].bias.copy_(prior_bias(statistics))
        self.interaction = (None if mode == "cnn" else
                            OrderedInteractions(2 * cnn_channels, self.cnn.output_channels, relation_rank, mode))

    def forward(self, batch):
        mask = batch["mask"]
        first = self.cnn.first_maps(batch)
        maps = self.cnn.second_maps(first, mask)
        if self.interaction is not None:
            local = torch.cat(first[:2], -1)
            update = self.interaction(local, mask)
            maps = [h + delta for h, delta in zip(maps, update.split(maps[0].shape[-1], dim=-1))]
        return {"logits": self.direct(self.cnn.pool_maps(maps, mask))}

    def compute_loss(self, batch, settings, return_output=False):
        return supervised_loss(self, batch, settings, return_output)
