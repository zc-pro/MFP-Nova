"""GeoFactor: typed heterogeneous graph and differentiable high-order factors."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .research_layers import (SequenceEncoder, PropertyEncoder, GeometricLayer, Attention, FeedForward,
                              TailClassifier, Reconstruction, masked_mean, training_views,
                              reconstruction_loss, observed_label_loss)


def fragment_membership(batch):
    mask = batch["mask"]
    length = mask.shape[1]
    position = torch.arange(length, device=mask.device)
    continuous = []
    for width in (4, 8):
        groups = position//width
        membership = F.one_hot(groups, int(groups.max())+1).T.float()[None]*mask[:, None]
        continuous.append(membership)
    anchors = position[::4]
    distances = torch.cdist(batch["coords"][:, anchors], batch["coords"])
    spatial = (distances<8)&batch["struct_mask"][:, anchors, None]&batch["struct_mask"][:, None]
    membership = torch.cat([*continuous, spatial.float()], 1)
    kinds = torch.cat([torch.zeros(continuous[0].shape[1], device=mask.device, dtype=torch.long),
                       torch.ones(continuous[1].shape[1], device=mask.device, dtype=torch.long),
                       torch.full((len(anchors),), 2, device=mask.device, dtype=torch.long)])
    return membership, membership.sum(-1)>0, kinds


class HeterogeneousLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.residue = GeometricLayer(dim, heads, dropout)
        self.member_up, self.member_down = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.property_fragment = Attention(dim, heads, dropout)
        self.fragment_property = Attention(dim, heads, dropout)
        self.fragment_label = Attention(dim, heads, dropout)
        self.label_fragment = Attention(dim, heads, dropout)
        self.property_label = Attention(dim, heads, dropout)
        self.ff = nn.ModuleList([FeedForward(dim, 2, dropout) for _ in range(3)])

    def forward(self, residue, fragment, properties, labels, masks, membership, batch, quality_aware=True):
        rmask, fmask, pmask, lmask = masks
        graph = dict(batch)
        kind = batch["edge_type"].clone()
        length = kind.shape[1]
        adjacent = torch.eye(length, device=kind.device, dtype=torch.bool).roll(1, 0)
        adjacent[0, -1] = False
        adjacent |= adjacent.T.clone()
        valid = rmask[:, :, None]&rmask[:, None]
        kind = torch.where(adjacent[None]&valid, 1, kind)
        graph["edge_type"] = kind
        residue = self.residue(residue, rmask, graph, masked_mean(properties, pmask), quality_aware)
        member_up = membership/membership.sum(-1, keepdim=True).clamp_min(1)
        fragment = fragment+self.member_up(torch.einsum("bfl,bld->bfd", member_up, residue))
        # These are learned semantic associations, never physical property measurements.
        fragment = fragment+self.fragment_property(fragment, properties, fmask, pmask)
        properties = self.ff[0](properties+self.property_fragment(properties, fragment, pmask, fmask), pmask)
        labels = self.ff[1](labels+self.label_fragment(labels, fragment, lmask, fmask)
                           +self.property_label(labels, properties, lmask, pmask), lmask)
        fragment = self.ff[2](fragment+self.fragment_label(fragment, labels, fmask, lmask), fmask)
        member_down = membership/membership.sum(1, keepdim=True).clamp_min(1)
        residue = (residue+self.member_down(torch.einsum("bfl,bfd->bld", member_down, fragment)))*rmask[..., None]
        return residue, fragment, properties, labels


class StructuredFactors(nn.Module):
    def __init__(self, labels, dim, statistics, steps=4, latent_factors=8):
        super().__init__()
        self.steps = steps
        self.register_buffer("pair_prior", torch.tensor(statistics["pair_prior"], dtype=torch.float32))
        empirical = torch.tensor(statistics["hyperedges"], dtype=torch.float32).reshape(-1, labels)
        self.register_buffer("incidence", empirical)
        n = torch.tensor(statistics["hyperedge_clusters"], dtype=torch.float32)
        self.register_buffer("shrink", n/(n+20))
        self.pair_query, self.pair_key = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.pair_scale = nn.Parameter(torch.tensor(.1))
        self.shared_factor = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, 1))
        self.factor_residual = nn.Parameter(torch.randn(len(empirical), dim)*.02)
        self.latent_incidence = nn.Parameter(torch.randn(latent_factors, labels)*.02)
        self.latent_factor = nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, 1))
        self.damping = nn.Parameter(torch.zeros(steps))

    def forward(self, unary, nodes, pairwise=True, high_order=True):
        pair = (self.pair_query(nodes)@self.pair_key(nodes).transpose(1, 2))/math.sqrt(nodes.shape[-1])
        pair = (pair+pair.transpose(1, 2))/2
        pair = torch.tanh(pair+self.pair_prior[None])*self.pair_scale.tanh()
        diagonal = torch.eye(nodes.shape[1], device=nodes.device, dtype=torch.bool)
        pair = pair.masked_fill(diagonal, 0)
        normalized = self.incidence/self.incidence.sum(-1, keepdim=True).clamp_min(1)
        joint = torch.einsum("kc,bcd->bkd", normalized, nodes)
        coefficient = self.shared_factor(joint+self.factor_residual[None]*self.shrink[None, :, None]).squeeze(-1).tanh()*self.shrink
        latent = self.latent_incidence.softmax(-1)
        latent_coefficient = self.latent_factor(torch.einsum("kc,bcd->bkd", latent, nodes)).squeeze(-1).tanh()
        q = unary.sigmoid()
        for damping in self.damping:
            message = torch.zeros_like(unary)
            if pairwise:
                message = message+torch.einsum("bcd,bd->bc", pair, q)/max(1, nodes.shape[1]-1)
            if high_order:
                # Gradient of b_H * product(y_c), one message per member. No labels
                # enter inference; unsupported factors use a shared nonlinear potential.
                logq = q.clamp_min(1e-5).log()
                log_product = torch.einsum("kc,bc->bk", self.incidence, logq)
                excluding = torch.exp(log_product[:, :, None]-logq[:, None])*self.incidence[None]
                empirical_message = (excluding*coefficient[:, :, None]).sum(1)
                empirical_message /= self.incidence.sum(0).clamp_min(1)
                summary = torch.einsum("kc,bc->bk", latent, q)
                latent_message = torch.einsum("kc,bk->bc", latent, latent_coefficient*torch.tanh(summary))
                message = message+empirical_message+latent_message
            step = torch.sigmoid(damping)
            q = (1-step)*q+step*torch.sigmoid(unary+message)
        return torch.logit(q.clamp(1e-6, 1-1e-6)), pair, coefficient


class GeoFactor(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=256, heads=8, dropout=.15,
                 sequence_depth=2, property_depth=2, graph_depth=6, inference_steps=4,
                 pairwise=True, high_order=True, spatial=True, structured=True, quality_aware=True):
        super().__init__()
        self.statistics = statistics
        self.pairwise, self.high_order, self.spatial, self.structured = pairwise, high_order, spatial, structured
        self.quality_aware = quality_aware
        self.sequence = SequenceEncoder(input_dim, vocab_size, dim, heads, dropout, sequence_depth)
        self.properties = PropertyEncoder(dim, heads, dropout, property_depth)
        self.geometry_input = nn.Linear(16, dim)
        self.dssp = nn.Embedding(10, dim)
        self.fragment_type = nn.Embedding(3, dim)
        self.label_query = nn.Parameter(torch.randn(num_labels, dim)*.02)
        self.global_geometry = nn.Linear(50, dim)
        self.layers = nn.ModuleList([HeterogeneousLayer(dim, heads, dropout) for _ in range(graph_depth)])
        self.unary = TailClassifier(num_labels, dim, statistics)
        self.factors = StructuredFactors(num_labels, dim, statistics, inference_steps)
        self.reconstruction = Reconstruction(dim)

    def forward(self, batch):
        if not self.spatial:
            batch = dict(batch, struct_mask=torch.zeros_like(batch["struct_mask"]),
                         edge_type=torch.where(batch["edge_type"]==2, 0, batch["edge_type"]))
        sequence, _ = self.sequence(batch)
        residue = sequence[:, :-1]+(self.geometry_input(batch["geometry"])+self.dssp(batch["dssp"]))*batch["struct_mask"][..., None]
        properties, pmask = self.properties(batch)
        membership, fmask, kinds = fragment_membership(batch)
        normalized = membership/membership.sum(-1, keepdim=True).clamp_min(1)
        fragment = torch.einsum("bfl,bld->bfd", normalized, residue)+self.fragment_type(kinds)
        fragment *= fmask[..., None]
        labels = self.label_query[None].expand(residue.shape[0], -1, -1)+sequence[:, -1, None]
        global_geometry = self.global_geometry(torch.cat([batch["struct_global"], batch["struct_global_mask"].float()], -1))
        labels = labels+global_geometry[:, None]*batch["struct_mask"].any(1)[:, None, None]
        lmask = torch.ones(labels.shape[:2], dtype=torch.bool, device=labels.device)
        for layer in self.layers:
            residue, fragment, properties, labels = layer(residue, fragment, properties, labels,
                                                          [batch["mask"], fmask, pmask, lmask], membership, batch, self.quality_aware)
        unary = self.unary(labels)
        logits, pair, factor = self.factors(unary, labels, self.pairwise, self.high_order) if self.structured else (unary, unary[..., None]*0, unary*0)
        return {"logits": logits, "direct": unary, "pair": pair, "factor": factor,
                "reconstruction": self.reconstruction(residue, properties)}

    def compute_loss(self, batch, settings, return_output=False):
        view, hidden = training_views(batch, settings.get("modality_dropout", .1), settings.get("mask_ratio", .15))
        output = self(view)
        loss = lambda x: observed_label_loss(x, batch["y"], self.statistics, settings.get("unlabeled_weight", .3))
        terms = {"classification": loss(output["logits"]), "direct": loss(output["direct"]),
                 "reconstruction": reconstruction_loss(output["reconstruction"], batch, hidden),
                 "factor": output["pair"].square().mean()+output["factor"].square().sum()/max(1, output["factor"].numel())}
        total = terms["classification"]+settings.get("direct", .2)*terms["direct"]+settings.get("reconstruction", .05)*terms["reconstruction"]+settings.get("factor", .001)*terms["factor"]
        return (total, terms, output) if return_output else (total, terms)
