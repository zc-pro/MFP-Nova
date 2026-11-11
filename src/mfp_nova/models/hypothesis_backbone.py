"""The shared raw-sequence B0 backbone for the H1–H8 experiments."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


RESIDUES = {residue: index for index, residue in enumerate("ACDEFGHIKLMNPQRSTVWY", 2)}


def tokenize(sequence):
    sequence = sequence.strip()
    if not sequence:
        raise ValueError("An amino acid sequence cannot be empty")
    return torch.tensor([RESIDUES.get(residue.upper(), 1) for residue in sequence], dtype=torch.long)


def pad_tokens(tokens):
    padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True)
    return padded, padded.ne(0)


@dataclass
class BackboneOutput:
    local: torch.Tensor
    contextual: torch.Tensor
    evidence: torch.Tensor
    summary: torch.Tensor
    logits: torch.Tensor
    attention: tuple[torch.Tensor, torch.Tensor] | None = None


class RelativeEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout, relative_clip):
        super().__init__()
        if dim % heads:
            raise ValueError("Backbone dimension must be divisible by attention heads")
        self.heads = heads
        self.clip = relative_clip
        self.dropout = dropout
        self.relative_bias = nn.Embedding(2 * relative_clip + 1, heads)
        nn.init.zeros_(self.relative_bias.weight)
        self.attention_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, mask, distance):
        batch, length, dim = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).reshape(batch, length, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        bias = self.relative_bias(distance.clamp(-self.clip, self.clip) + self.clip).permute(2, 0, 1)
        bias = bias.to(q.dtype).unsqueeze(0).masked_fill(~mask[:, None, None, :], -torch.inf)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                    dropout_p=self.dropout if self.training else 0.)
        x = x + self.residual_dropout(self.output(attention.transpose(1, 2).reshape(batch, length, dim)))
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x.masked_fill(~mask[..., None], 0.)


class FunctionalQuery(nn.Module):
    def __init__(self, num_labels, dim, heads):
        super().__init__()
        self.query = nn.Parameter(torch.empty(num_labels, dim))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask, return_attention):
        query = self.query.unsqueeze(0).expand(x.shape[0], -1, -1)
        value, weights = self.attention(query, x, x, key_padding_mask=~mask, need_weights=return_attention)
        return self.norm(query + value), weights


class SequenceBackbone(nn.Module):
    def __init__(self, num_labels, dim=192, heads=6, depth=4, dropout=.1, relative_clip=64,
                 residue_features=None):
        super().__init__()
        self.num_labels = num_labels
        self.dim = dim
        if residue_features is None:
            self.embedding = nn.Embedding(22, dim, padding_idx=0)
        else:
            self.embedding = nn.Sequential(nn.Embedding.from_pretrained(residue_features, padding_idx=0),
                                           nn.Linear(residue_features.shape[1], dim))
        self.local_norm = nn.LayerNorm(dim)
        self.local_paths = nn.ModuleList(nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
                                         for kernel in (3, 7))
        self.local_mix = nn.Linear(2 * dim, dim)
        self.local_gate = nn.Linear(dim, dim)
        self.encoder = nn.ModuleList(RelativeEncoderLayer(dim, heads, dropout, relative_clip) for _ in range(depth))
        self.context_norm = nn.LayerNorm(dim)
        self.local_query = FunctionalQuery(num_labels, dim, heads)
        self.global_query = FunctionalQuery(num_labels, dim, heads)
        self.evidence = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.label_weight = nn.Parameter(torch.empty(num_labels, dim))
        self.label_bias = nn.Parameter(torch.zeros(num_labels))
        nn.init.uniform_(self.label_weight, -dim ** -.5, dim ** -.5)

    def forward(self, tokens, mask, return_attention=False):
        # Masked positions keep their coordinates for H6 but cannot inject token content.
        mask3 = mask[..., None]
        embeddings = self.embedding(tokens).masked_fill(~mask3, 0.)
        normalized = self.local_norm(embeddings).masked_fill(~mask3, 0.)
        branches = [F.gelu(path(normalized.transpose(1, 2)).transpose(1, 2)) for path in self.local_paths]
        local = embeddings + self.local_gate(normalized).sigmoid() * self.local_mix(torch.cat(branches, dim=-1))
        local = local.masked_fill(~mask3, 0.)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        distance = positions[None, :] - positions[:, None]
        contextual = local
        for layer in self.encoder:
            contextual = layer(contextual, mask, distance)
        contextual = self.context_norm(contextual).masked_fill(~mask3, 0.)
        left, local_attention = self.local_query(local, mask, return_attention)
        right, global_attention = self.global_query(contextual, mask, return_attention)
        evidence = self.evidence(torch.cat([left, right], dim=-1))
        logits = (evidence * self.label_weight).sum(dim=-1) + self.label_bias
        mean = contextual.sum(1) / mask.sum(1, keepdim=True)
        maximum = contextual.masked_fill(~mask3, -torch.inf).amax(1)
        summary = F.layer_norm((mean + maximum) * .5, (self.dim,))
        attention = (local_attention, global_attention) if return_attention else None
        return BackboneOutput(local, contextual, evidence, summary, logits, attention)
