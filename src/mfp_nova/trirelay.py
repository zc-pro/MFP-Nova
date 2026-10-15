"""TriRelay: bidirectional multimodal evidence and label-conditioned feedback."""
import torch
from torch import nn

from .research_layers import (SequenceEncoder, GeometryEncoder, PropertyEncoder, MultimodalInteraction,
                              EvidenceReader, TailClassifier, LabelRelations, Attention, Reconstruction,
                              masked_mean, training_views, reconstruction_loss, observed_label_loss)


class TriRelay(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=256, heads=8,
                 dropout=.15, sequence_depth=4, geometry_depth=4, property_depth=2,
                 interaction_depth=3, feedback=True, relations=True, high_order=True,
                 quality_aware=True, property_geometry=True):
        super().__init__()
        self.statistics = statistics
        self.feedback_enabled, self.relations_enabled = feedback, relations
        self.high_order, self.quality_aware = high_order, quality_aware
        self.property_geometry = property_geometry
        self.sequence = SequenceEncoder(input_dim, vocab_size, dim, heads, dropout, sequence_depth)
        self.geometry = GeometryEncoder(dim, heads, dropout, geometry_depth)
        self.properties = PropertyEncoder(dim, heads, dropout, property_depth)
        self.interactions = nn.ModuleList([MultimodalInteraction(dim, heads, dropout) for _ in range(interaction_depth)])
        self.reader = EvidenceReader(num_labels, dim, heads, dropout)
        self.relations = LabelRelations(num_labels, dim, heads, dropout, statistics)
        self.feedback = nn.ModuleList([Attention(dim, heads, dropout) for _ in range(3)])
        self.feedback_gate = nn.ModuleList([nn.Linear(2*dim, dim) for _ in range(3)])
        self.refine = MultimodalInteraction(dim, heads, dropout)
        self.classifier = TailClassifier(num_labels, dim, statistics)
        self.relation_gate = nn.Linear(2*dim, 1)
        self.reconstruction = Reconstruction(dim)

    def forward(self, batch):
        prop, pmask = self.properties(batch)
        seq, smask = self.sequence(batch)
        geo, gmask = self.geometry(batch, masked_mean(prop, pmask) if self.property_geometry else None, self.quality_aware)
        streams, masks = [seq, geo, prop], [smask, gmask, pmask]
        for interaction in self.interactions:
            streams = interaction(streams, masks, batch["quality"], self.quality_aware)
        direct, evidence, fusion = self.reader(streams, masks, batch["quality"], quality_aware=self.quality_aware)
        relation = self.relations(direct, self.high_order) if self.relations_enabled else direct
        if self.feedback_enabled:
            label_mask = torch.ones(relation.shape[:2], dtype=torch.bool, device=relation.device)
            refined = []
            for x, mask, attention, gate in zip(streams, masks, self.feedback, self.feedback_gate):
                delta = attention(x, relation, mask, label_mask)
                refined.append((x+torch.sigmoid(gate(torch.cat([x, delta], -1)))*delta)*mask[..., None])
            streams = self.refine(refined, masks, batch["quality"], self.quality_aware)
            final, evidence, fusion = self.reader(streams, masks, batch["quality"], context=relation, quality_aware=self.quality_aware)
            relation = self.relations(final, self.high_order) if self.relations_enabled else final
        else:
            final = direct
        direct_logits = self.classifier(direct)
        evidence_logits = self.classifier(final)
        relation_logits = self.classifier(relation)-evidence_logits
        logits = evidence_logits+torch.sigmoid(self.relation_gate(torch.cat([final, relation], -1))).squeeze(-1)*relation_logits
        branch = torch.stack([self.classifier(evidence[:, :, m]) for m in range(3)], 1)
        return {"logits": logits, "direct": direct_logits, "branch": branch, "fusion": fusion,
                "available": torch.stack([m.any(1) for m in masks], 1),
                "reconstruction": self.reconstruction(streams[0][:, :-1], streams[2])}

    def compute_loss(self, batch, settings, return_output=False):
        view, hidden = training_views(batch, settings.get("modality_dropout", .1), settings.get("mask_ratio", .15))
        output = self(view)
        loss = lambda logits, target: observed_label_loss(logits, target, self.statistics, settings.get("unlabeled_weight", .3))
        terms = {"classification": loss(output["logits"], batch["y"]), "direct": loss(output["direct"], batch["y"])}
        branches = [loss(output["branch"][output["available"][:, m], m], batch["y"][output["available"][:, m]])
                    for m in range(3) if output["available"][:, m].any()]
        terms["branch"] = torch.stack(branches).mean()
        terms["reconstruction"] = reconstruction_loss(output["reconstruction"], batch, hidden)
        total = terms["classification"]+settings.get("direct", .2)*terms["direct"]+settings.get("branch", .1)*terms["branch"]+settings.get("reconstruction", .05)*terms["reconstruction"]
        return (total, terms, output) if return_output else (total, terms)
