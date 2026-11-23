"""Compact contextual reader for the second branch of DuoEvidence."""

import torch
from torch import nn
from torch.nn import functional as F

from .sequence_features import FrozenFeatures, ResidueCNN, prior_bias


class DuoContextReader(nn.Module):
    def __init__(self, input_dim, peptide_input_dim, statistics, mode):
        super().__init__()
        width = 128
        self.mode = mode
        self.motifs = ResidueCNN(.2, channels=64)
        self.motif_projection = nn.Linear(256, width)
        self.esm = FrozenFeatures(input_dim, width, peptide_input_dim)
        self.position = nn.Embedding(80, width)
        self.input_norm = nn.LayerNorm(width)
        self.latents = nn.Parameter(torch.randn(16, width) * .02)
        self.read = nn.MultiheadAttention(width, 4, dropout=.1, batch_first=True)
        self.process = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=.1,
                                                  activation="gelu", batch_first=True, norm_first=True)
        self.classifier = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(statistics["labels"])))
        with torch.no_grad():
            self.classifier[-1].bias.copy_(prior_bias(statistics))

    def forward(self, batch):
        mask = batch["mask"]
        local = self.motif_projection(self.motifs(batch))
        positions = torch.arange(mask.size(1), device=mask.device)
        residues = self.input_norm(local + self.esm(batch) + self.position(positions))
        if self.mode == "pooled":
            # Fixed four-residue windows retain coarse position and reduce access to individual tokens.
            pad = (-mask.size(1)) % 4
            windows = F.pad(residues.masked_fill(~mask[..., None], 0), (0, 0, 0, pad))
            counts = F.pad(mask, (0, pad)).reshape(mask.size(0), -1, 4).sum(2)
            residues = windows.reshape(mask.size(0), -1, 4, residues.size(-1)).sum(2)
            residues = residues / counts.clamp_min(1)[..., None]
            mask = counts > 0
        queries = self.latents[None].expand(mask.size(0), -1, -1)
        context, _ = self.read(queries, residues, residues, key_padding_mask=~mask, need_weights=False)
        return self.classifier(self.process(queries + context).mean(1))
