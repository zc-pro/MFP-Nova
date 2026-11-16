"""Independent local, contextual, and pair readouts with optional semantic queries."""

import math

import numpy as np
import torch
from torch import nn

from ..multimodal import function_projection
from ..research_layers import Attention, FeedForward
from .helixset import LabelPairRead, QualityRead, structure_strength


V5_MODES = ("baseline", "shared", "split", "semantic", "shuffled")
V5_VIEWS = ("local", "context", "pair")


def function_adjacency(labels, shuffled=False):
    axes = np.asarray(function_projection(labels, "fine39")).argmax(1)
    specific = np.array([not label.endswith("__unknown") for label in labels])
    graph = (axes[:, None] == axes[None, :]) & specific[:, None] & specific[None, :]
    np.fill_diagonal(graph, False)
    if not shuffled:
        return graph
    # Undirected edge swaps preserve each label's degree, including isolated labels.
    edges = list(zip(*np.where(np.triu(graph, 1))))
    if len(edges) < 2:
        return graph
    rng = np.random.default_rng(42)
    for _ in range(20 * len(edges)):
        i, j = rng.choice(len(edges), 2, replace=False)
        a, b = edges[i]
        c, d = edges[j]
        if rng.integers(2):
            c, d = d, c
        if len({a, b, c, d}) != 4 or graph[a, d] or graph[c, b]:
            continue
        graph[a, b] = graph[b, a] = graph[c, d] = graph[d, c] = False
        graph[a, d] = graph[d, a] = graph[c, b] = graph[b, c] = True
        edges[i], edges[j] = (a, d), (c, b)
    return graph


class V5Readout(nn.Module):
    def __init__(self, labels, prior, dim, heads, pair_dim, dropout, mode):
        super().__init__()
        self.mode = mode
        self.queries = nn.Parameter(torch.randn(3, len(labels), dim) * .02)
        self.scale_embeddings = nn.Parameter(torch.randn(4, dim) * .02)
        self.local = Attention(dim, heads, dropout)
        self.context = Attention(dim, heads, dropout)
        self.properties = Attention(dim, heads, dropout)
        self.structure = QualityRead(dim, heads, dropout)
        self.pair = LabelPairRead(dim, pair_dim, dropout)
        self.source_norms = nn.ModuleList(nn.LayerNorm(dim, elementwise_affine=False) for _ in range(5))
        self.evidence = nn.ModuleList(FeedForward(dim, 2, dropout) for _ in V5_VIEWS)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in V5_VIEWS)
        self.weights = nn.Parameter(torch.randn(3, len(labels), dim) / math.sqrt(dim))
        self.bias = nn.Parameter(prior.expand(3, -1).clone())
        if mode in ("semantic", "shuffled"):
            graph = function_adjacency(labels, shuffled=mode == "shuffled")
            adjacency = torch.from_numpy(graph).float()
            self.register_buffer("relation", adjacency / adjacency.sum(1, keepdim=True).clamp_min(1.))
            self.relation_query = nn.ModuleList(nn.Linear(2 * dim, dim, bias=False) for _ in V5_VIEWS)
            self.relation_write = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in V5_VIEWS)
            for write in self.relation_write:
                nn.init.zeros_(write.weight)
        else:
            self.relation = None

    def read(self, route, query, active, memory):
        (motifs, context, context_mask, properties, property_mask, geometry, geometry_mask,
         reliability, residues, pairs, geometric_pairs, pair_quality, mask, present) = memory
        if route == 0:
            keys = torch.cat([h + scale for h, scale in zip(motifs, self.scale_embeddings)], dim=1)
            return self.source_norms[0](self.local(query, keys, active, mask.repeat(1, 4)))
        if route == 1:
            h = self.source_norms[1](self.context(query, context, active, context_mask))
            props = self.source_norms[2](self.properties(query, properties, active, property_mask))
            return (h + props) / present[:, 1:3].sum(1).clamp_min(1)[:, None, None]
        pair, _ = self.pair(query, residues, pairs, mask, geometric_pairs, pair_quality)
        pair = self.source_norms[4](pair) * present[:, 4, None, None]
        geometry = self.structure(query, geometry, active, geometry_mask, reliability)
        geometry = self.source_norms[3](geometry) * structure_strength(reliability, geometry_mask)
        return (geometry + pair) / present[:, 3:].sum(1).clamp_min(1)[:, None, None]

    def forward(self, memory, global_state, use_relation=True, without_view=None):
        present = memory[-1]
        available = torch.stack((present[:, 0], present[:, 1:3].any(1), present[:, 3:].any(1)), 1)
        reads = []
        for route, query in enumerate(self.queries):
            offset = global_state[:, None] if route == 0 else 0.
            active = available[:, route, None].expand(-1, len(query))
            raw_query = query[None].expand(len(active), -1, -1) + offset
            if self.relation is not None and use_relation:
                neighbor = self.relation.to(query.dtype) @ query
                related = query + self.relation_query[route](torch.cat((neighbor, query - neighbor), -1))
                # Both queries use the same source. Batch the queries to reuse the keys.
                both = self.read(route, torch.cat((raw_query, related[None].expand_as(raw_query) + offset), 1),
                                 active.repeat(1, 2), memory)
                raw, aux = both.chunk(2, dim=1)
                raw = raw + self.relation_write[route](aux - raw) * self.relation.any(1)[None, :, None]
            else:
                raw = self.read(route, raw_query, active, memory)
            reads.append(self.evidence[route](raw + raw_query, active))

        states = torch.stack(reads, dim=1)
        if self.mode == "shared":
            shared = states.sum(1, keepdim=True) / available.sum(1)[:, None, None, None].clamp_min(1)
            states = shared.expand_as(states)
            available = torch.ones_like(available)
        states = torch.stack([norm(h) for norm, h in zip(self.norms, states.unbind(1))], dim=1)
        logits = ((states * self.weights).sum(-1) + self.bias).float()
        if without_view is not None:
            available = available.clone()
            available[:, V5_VIEWS.index(without_view)] = False
            if not available.any(1).all():
                raise ValueError("View ablation leaves some peptides without an available view")
        probability = (logits.sigmoid() * available[..., None]).sum(1) / available.sum(1)[:, None].clamp_min(1)
        return {"logits": probability.clamp(1e-7, 1 - 1e-7).logit(),
                "view_logits": logits, "view_available": available}
