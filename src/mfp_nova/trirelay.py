"""TriRelay: bidirectional multimodal evidence and label-conditioned feedback."""
import torch
from torch import nn

from .research_layers import (SequenceEncoder, GeometryEncoder, PropertyEncoder, MultimodalInteraction,
                              EvidenceReader, TailClassifier, LabelRelations, Attention, Reconstruction,
                              FeedForward, masked_mean, training_views, reconstruction_loss, observed_label_loss)


class SharedPrivateEvidence(nn.Module):
    """Preserve pre-relay evidence, without claiming statistical disentanglement."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.shared = nn.ModuleList([FeedForward(dim, 2, dropout) for _ in range(3)])
        self.private = nn.ModuleList([nn.ModuleList([FeedForward(dim, 2, dropout) for _ in range(2)])
                                      for _ in range(3)])
        self.readers = nn.ModuleList([Attention(dim, heads, dropout) for _ in range(3)])
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(4*dim, dim), nn.GELU(), nn.Linear(dim, 1))
                                    for _ in range(3)])

    def split(self, streams, masks):
        shared, private = [], []
        for x, mask, shared_layer, private_layers in zip(streams, masks, self.shared, self.private):
            shared.append(shared_layer(x, mask))
            for layer in private_layers:
                x = layer(x, mask)
            private.append(x)
        return shared, private

    def read(self, reader, streams, private, masks, quality, context=None, quality_aware=True, private_enabled=True):
        query, shared_evidence, qualities = reader.read(streams, masks, quality, context, quality_aware)
        qmask = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device)
        private_evidence = torch.stack([attention(query, x, qmask, mask)
                                       for attention, x, mask in zip(self.readers, private, masks)], 2)
        gates = torch.stack([torch.sigmoid(gate(torch.cat([query, shared_evidence[:, :, m],
                                                          private_evidence[:, :, m], qualities[:, :, m]], -1)))
                             for m, gate in enumerate(self.gates)], 2)
        present = torch.stack([mask.any(1) for mask in masks], 1)
        gates = gates*present[:, None, :, None]
        if not private_enabled:
            gates = torch.zeros_like(gates)
        evidence = shared_evidence+gates*private_evidence
        # Recompute modality weights using both evidence paths, including missing masks.
        fused, evidence, weights = reader.fuse(query, evidence, qualities, masks)
        return fused, evidence, weights, private_evidence, gates.squeeze(-1), shared_evidence


class TriRelay(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=256, heads=8,
                 dropout=.15, sequence_depth=4, geometry_depth=4, property_depth=2,
                 interaction_depth=3, feedback=True, relations=True, high_order=True,
                 quality_aware=True, property_geometry=True, function_aux=False,
                 function_aux_statistics=None, shared_private=False):
        super().__init__()
        self.statistics = statistics
        self.feedback_enabled, self.relations_enabled = feedback, relations
        self.high_order, self.quality_aware = high_order, quality_aware
        self.property_geometry = property_geometry
        # Runtime inference intervention; not a learned parameter/checkpoint setting.
        self.private_evidence_enabled = True
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
        if function_aux:
            from .function_auxiliary import FunctionAuxiliary
            self.function_auxiliary = FunctionAuxiliary(statistics["labels"], function_aux_statistics, dim, heads, dropout)
        if shared_private:
            self.evidence_partition = SharedPrivateEvidence(dim, heads, dropout)

    def forward(self, batch, unmixed_branches=False, first_readouts=False):
        prop, pmask = self.properties(batch)
        seq, smask = self.sequence(batch)
        geo, gmask = self.geometry(batch, masked_mean(prop, pmask) if self.property_geometry else None, self.quality_aware)
        streams, masks = [seq, geo, prop], [smask, gmask, pmask]
        partitioned = hasattr(self, "evidence_partition")
        if partitioned:
            streams, private = self.evidence_partition.split(streams, masks)
        for interaction in self.interactions:
            streams = interaction(streams, masks, batch["quality"], self.quality_aware)
        if partitioned:
            direct, evidence, fusion, private_evidence, private_gate, shared_evidence = self.evidence_partition.read(
                self.reader, streams, private, masks, batch["quality"], quality_aware=self.quality_aware,
                private_enabled=self.private_evidence_enabled)
            if unmixed_branches or first_readouts:
                # First read has no relation context derived from mixed evidence.
                # Reuse these same reads: no extra attention or learned classifier.
                shared_branch = torch.stack([self.classifier(shared_evidence[:, :, m]) for m in range(3)], 1)
                unmixed_private_branch = torch.stack([self.classifier(private_evidence[:, :, m]) for m in range(3)], 1)
            if first_readouts:
                first_mixed_branch = torch.stack([self.classifier(evidence[:, :, m]) for m in range(3)], 1)
                first_fusion, first_private_gate = fusion, private_gate
        else:
            direct, evidence, fusion = self.reader(streams, masks, batch["quality"], quality_aware=self.quality_aware)
        relation = self.relations(direct, self.high_order) if self.relations_enabled else direct
        if self.feedback_enabled:
            label_mask = torch.ones(relation.shape[:2], dtype=torch.bool, device=relation.device)
            refined = []
            for x, mask, attention, gate in zip(streams, masks, self.feedback, self.feedback_gate):
                delta = attention(x, relation, mask, label_mask)
                refined.append((x+torch.sigmoid(gate(torch.cat([x, delta], -1)))*delta)*mask[..., None])
            streams = self.refine(refined, masks, batch["quality"], self.quality_aware)
            if partitioned:
                final, evidence, fusion, private_evidence, private_gate, shared_evidence = self.evidence_partition.read(
                    self.reader, streams, private, masks, batch["quality"], context=relation, quality_aware=self.quality_aware,
                    private_enabled=self.private_evidence_enabled)
            else:
                final, evidence, fusion = self.reader(streams, masks, batch["quality"], context=relation, quality_aware=self.quality_aware)
            relation = self.relations(final, self.high_order) if self.relations_enabled else final
        else:
            final = direct
        direct_logits = self.classifier(direct)
        evidence_logits = self.classifier(final)
        relation_logits = self.classifier(relation)-evidence_logits
        relation_weight = torch.sigmoid(self.relation_gate(torch.cat([final, relation], -1)))
        logits = evidence_logits+relation_weight.squeeze(-1)*relation_logits
        branch = torch.stack([self.classifier(evidence[:, :, m]) for m in range(3)], 1)
        output = {"logits": logits, "direct": direct_logits, "branch": branch, "fusion": fusion,
                "available": torch.stack([m.any(1) for m in masks], 1),
                "reconstruction": self.reconstruction(streams[0][:, :-1], streams[2])}
        if partitioned:
            output["private_branch"] = torch.stack([self.classifier(private_evidence[:, :, m]) for m in range(3)], 1)
            output["private_gate"] = private_gate
            if unmixed_branches or first_readouts:
                output["shared_branch"] = shared_branch
                output["unmixed_private_branch"] = unmixed_private_branch
            if first_readouts:
                output["first_mixed_branch"] = first_mixed_branch
                output["first_fusion"] = first_fusion
                output["first_private_gate"] = first_private_gate
        if hasattr(self, "function_auxiliary"):
            output["function_aux"] = self.function_auxiliary(final+relation_weight*(relation-final))
        return output

    def compute_loss(self, batch, settings, return_output=False):
        symmetric = settings.get("branch_supervision", "mixed_private") == "shared_private"
        if symmetric and not hasattr(self, "evidence_partition"):
            raise ValueError("Shared/private branch supervision requires shared-private TriRelay")
        view, hidden = training_views(batch, settings.get("modality_dropout", .1), settings.get("mask_ratio", .15))
        output = self(view, unmixed_branches=symmetric)
        loss = lambda logits, target: observed_label_loss(logits, target, self.statistics, settings.get("unlabeled_weight", .3))
        terms = {"classification": loss(output["logits"], batch["y"]), "direct": loss(output["direct"], batch["y"])}
        def branch_loss(branch):
            return torch.stack([loss(branch[output["available"][:, m], m], batch["y"][output["available"][:, m]])
                                for m in range(3) if output["available"][:, m].any()]).mean()
        if symmetric:
            terms["branch_shared"] = branch_loss(output["shared_branch"])
            terms["branch_private"] = branch_loss(output["unmixed_private_branch"])
            terms["branch"] = .5*(terms["branch_shared"]+terms["branch_private"])
        else:
            # Preserve the original reduction order and default supervision.
            branch_outputs = [output["branch"]]
            if "private_branch" in output:
                branch_outputs.append(output["private_branch"])
            branches = [loss(branch[output["available"][:, m], m], batch["y"][output["available"][:, m]])
                        for branch in branch_outputs for m in range(3) if output["available"][:, m].any()]
            terms["branch"] = torch.stack(branches).mean()
        terms["reconstruction"] = reconstruction_loss(output["reconstruction"], batch, hidden)
        total = terms["classification"]+settings.get("direct", .2)*terms["direct"]+settings.get("branch", .1)*terms["branch"]+settings.get("reconstruction", .05)*terms["reconstruction"]
        if hasattr(self, "function_auxiliary"):
            auxiliary = self.function_auxiliary.loss(output["logits"], output["function_aux"], batch, settings)
            terms.update(auxiliary)
            total = total+sum(settings.get(key, 0)*value for key, value in auxiliary.items())
        return (total, terms, output) if return_output else (total, terms)
