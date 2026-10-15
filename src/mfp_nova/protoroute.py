"""ProtoRoute: label-conditioned four-expert routing with adaptive prototypes."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .research_layers import (SequenceEncoder, GeometryEncoder, PropertyEncoder, MultimodalInteraction,
                              Attention, TailClassifier, LabelRelations, Reconstruction, masked_mean,
                              training_views, reconstruction_loss, observed_label_loss)


class AdaptivePrototypes(nn.Module):
    def __init__(self, labels, dim, statistics, bank_size=32, max_prototypes=4, shared=True):
        super().__init__()
        self.shared = shared
        counts = statistics["prototype_counts"]
        geometric = statistics.get("geometry_prototype_counts", counts)
        capacity = torch.tensor([counts, geometric, counts, counts]).T
        self.register_buffer("active", torch.arange(max_prototypes)[None, None]<capacity[..., None])
        clusters = torch.tensor([statistics["positive_clusters"], statistics.get("geometry_positive_clusters", statistics["positive_clusters"]),
                                 statistics["positive_clusters"], statistics["positive_clusters"]], dtype=torch.float32).T
        self.register_buffer("shrink", clusters/(clusters+100))
        self.bank = nn.Parameter(torch.randn(4, bank_size, dim)*.1)
        self.assignment = nn.Parameter(torch.randn(labels, 4, max_prototypes, bank_size)*.1)
        self.residual = nn.Parameter(torch.randn(labels, 4, max_prototypes, dim)*.1)
        self.mixture = nn.Parameter(torch.zeros(labels, 4, max_prototypes))
        self.log_temperature = nn.Parameter(torch.full((4,), math.log(10)))

    def forward(self, evidence):
        shared = torch.einsum("ceks,esd->cekd", self.assignment.softmax(-1), self.bank)
        prototypes = shared+self.shrink[..., None, None]*self.residual if self.shared else self.residual
        prototypes = F.normalize(prototypes, dim=-1)
        cosine = torch.einsum("bced,cekd->bcek", F.normalize(evidence, dim=-1), prototypes)
        log_mix = self.mixture.masked_fill(~self.active, -1e4).log_softmax(-1)
        temperature = self.log_temperature.exp().clamp(1, 30)
        scores = cosine*temperature[None, None, :, None]+log_mix
        scores = scores.masked_fill(~self.active[None], -1e4)
        logits = torch.logsumexp(scores, -1)
        logits = torch.where(self.active.any(-1)[None], logits, 0)
        return logits, cosine, scores.softmax(-1), prototypes


class ProtoRoute(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=256, heads=8, dropout=.15,
                 sequence_depth=4, geometry_depth=4, property_depth=3, collaboration_depth=3,
                 bank_size=32, max_prototypes=4, shared_prototypes=True, relations=True,
                 high_order=True, quality_aware=True, dynamic_routing=True):
        super().__init__()
        self.statistics = statistics
        self.relations_enabled, self.high_order = relations, high_order
        self.quality_aware, self.dynamic_routing = quality_aware, dynamic_routing
        self.sequence = SequenceEncoder(input_dim, vocab_size, dim, heads, dropout, sequence_depth)
        self.geometry = GeometryEncoder(dim, heads, dropout, geometry_depth)
        self.properties = PropertyEncoder(dim, heads, dropout, property_depth)
        self.collaboration = nn.ModuleList([MultimodalInteraction(dim, heads, dropout) for _ in range(collaboration_depth)])
        self.query = nn.Parameter(torch.randn(num_labels, dim)*.02)
        self.readers = nn.ModuleList([Attention(dim, heads, dropout) for _ in range(4)])
        self.expert_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.prototypes = AdaptivePrototypes(num_labels, dim, statistics, bank_size, max_prototypes, shared_prototypes)
        self.quality = nn.Linear(6, dim)
        self.route = nn.Sequential(nn.Linear(4*dim+1, 2*dim), nn.GELU(), nn.Linear(2*dim, 1))
        self.prototype_gate = nn.Parameter(torch.zeros(4))
        self.classifier = TailClassifier(num_labels, dim, statistics)
        self.relations = LabelRelations(num_labels, dim, heads, dropout, statistics)
        self.relation_gate = nn.Linear(2*dim, 1)
        self.reconstruction = Reconstruction(dim)

    def forward(self, batch):
        seq, smask = self.sequence(batch)
        prop, pmask = self.properties(batch)
        geo, gmask = self.geometry(batch, masked_mean(prop, pmask), self.quality_aware)
        streams, masks = [seq, geo, prop], [smask, gmask, pmask]
        cooperative = streams
        for interaction in self.collaboration:
            cooperative = interaction(cooperative, masks, batch["quality"], self.quality_aware)
        joint = torch.cat(cooperative, 1)
        jmask = torch.cat(masks, 1)
        query = self.query[None].expand(seq.shape[0], -1, -1)
        qmask = torch.ones(query.shape[:2], device=query.device, dtype=torch.bool)
        evidence = torch.stack([norm(query+reader(query, x, qmask, mask)) for reader, norm, x, mask
                                in zip(self.readers, self.expert_norms, [seq, geo, prop, joint], [smask, gmask, pmask, jmask])], 2)
        proto_logits, cosine, assignments, prototypes = self.prototypes(evidence)
        branch_logits = torch.stack([self.classifier(evidence[:, :, e]) for e in range(4)], -1)
        branch_logits = branch_logits+torch.sigmoid(self.prototype_gate)[None, None]*proto_logits
        quality = torch.cat([batch["quality"], batch["quality"].mean(1, keepdim=True)], 1)
        encoded_quality = self.quality(quality)[:, None].expand(-1, query.shape[1], -1, -1)
        if not self.quality_aware:
            encoded_quality = torch.zeros_like(encoded_quality)
        sample_context = masked_mean(seq, smask)[:, None, None].expand_as(evidence)
        routing_input = torch.cat([evidence, query[:, :, None].expand_as(evidence), sample_context,
                                   encoded_quality, proto_logits[..., None]], -1)
        score = self.route(routing_input).squeeze(-1) if self.dynamic_routing else torch.zeros_like(proto_logits)
        availability = torch.stack([m.any(1) for m in [smask, gmask, pmask, jmask]], -1)
        label_available = availability[:, None]&self.prototypes.active.any(-1)[None]
        route = score.masked_fill(~label_available, -1e4).softmax(-1)
        fused = (evidence*route[..., None]).sum(2)
        direct = (branch_logits*route).sum(-1)
        relation = self.relations(fused, self.high_order) if self.relations_enabled else fused
        residual = self.classifier(relation)-self.classifier(fused)
        logits = direct+torch.sigmoid(self.relation_gate(torch.cat([fused, relation], -1))).squeeze(-1)*residual
        return {"logits": logits, "direct": direct, "branch": branch_logits, "routing": route,
                "label_expert_available": label_available, "cosine": cosine, "assignments": assignments,
                "prototypes": prototypes, "reconstruction": self.reconstruction(cooperative[0][:, :-1], cooperative[2])}

    def compute_loss(self, batch, settings, return_output=False):
        view, hidden = training_views(batch, settings.get("modality_dropout", .1), settings.get("mask_ratio", .15))
        output = self(view)
        loss = lambda logits, mask=None: observed_label_loss(logits, batch["y"], self.statistics, settings.get("unlabeled_weight", .3), mask)
        terms = {"classification": loss(output["logits"]), "direct": loss(output["direct"])}
        terms["branch"] = torch.stack([loss(output["branch"][..., e], output["label_expert_available"][..., e]) for e in range(4)]).mean()
        positive = batch["y"].bool()[..., None]&output["label_expert_available"]
        attraction = (1-(output["assignments"]*output["cosine"]).sum(-1))*positive
        terms["prototype"] = attraction.sum()/positive.sum().clamp_min(1)
        prototypes = output["prototypes"]
        similarity = torch.einsum("cekd,celd->cekl", prototypes, prototypes)
        active = self.prototypes.active
        pairs = active[..., :, None]&active[..., None, :]
        pairs &= ~torch.eye(active.shape[-1], device=active.device, dtype=torch.bool)
        terms["diversity"] = (F.relu(similarity-.8).square()*pairs).sum()/pairs.sum().clamp_min(1)
        # Balance only available experts, label by label, rather than forcing missing structures.
        availability = output["label_expert_available"].float().mean(0)
        target = availability/availability.sum(-1, keepdim=True).clamp_min(1e-6)
        usage = output["routing"].mean(0)
        terms["balance"] = (usage*(usage.clamp_min(1e-8).log()-target.clamp_min(1e-8).log())).sum(-1).mean()
        terms["reconstruction"] = reconstruction_loss(output["reconstruction"], batch, hidden)
        total = terms["classification"]
        for key, default in [("direct", .2), ("branch", .1), ("prototype", .1), ("diversity", .01), ("balance", .01), ("reconstruction", .05)]:
            total = total+settings.get(key, default)*terms[key]
        return (total, terms, output) if return_output else (total, terms)
