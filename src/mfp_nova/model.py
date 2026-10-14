import torch
from torch import nn


class NovaSequenceModel(nn.Module):
    """Shared compact residue projection, global/local concatenation, binary heads."""
    def __init__(self, input_dim, num_labels, hidden_dim=128, kernels=(3, 5),
                 channels=64, dropout=0.1, use_local=True):
        super().__init__()
        if not kernels or any(k <= 0 or k % 2 == 0 for k in kernels):
            raise ValueError("Convolution kernels must be positive odd integers")
        self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.use_local = use_local
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, channels, k, padding=k // 2) for k in kernels
        ] if use_local else [])
        fusion_dim = hidden_dim + (channels * len(kernels) if use_local else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_dim, num_labels))

    def forward(self, residues, mask):
        x = self.project(residues).masked_fill(~mask.unsqueeze(-1), 0)
        global_features = x.sum(1) / mask.sum(1, keepdim=True)
        features = [global_features]
        for conv in self.convs:
            local = torch.nn.functional.gelu(conv(x.transpose(1, 2)))
            features.append(local.masked_fill(~mask.unsqueeze(1), -torch.inf).amax(-1))
        return self.head(torch.cat(features, dim=-1))
