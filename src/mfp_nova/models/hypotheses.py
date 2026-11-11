"""Sequence-only deployment for B0 and H1–H8."""

import torch
from torch import nn

from .hypothesis_backbone import SequenceBackbone


MODELS = ("b0", *(f"h{i}" for i in range(1, 9)))


class SequenceHypothesis(nn.Module):
    def __init__(self, name, labels, backbone_options, assets=None):
        super().__init__()
        self.name = name
        self.labels = labels
        self.backbone = SequenceBackbone(len(labels), **backbone_options)
        self.assets = assets or {}
        self.head = self.make_head()

    def make_head(self):
        if not self.assets.get("head_active"):
            return None
        from .hypothesis_relations import (DecisionResidual, EvidenceMessages, PairJointHead,
                                           PrototypeReadout, ReweightedPairReadout)
        from .hypothesis_set import CardinalitySetHead

        dim, count = self.backbone.dim, len(self.labels)
        if self.name == "h1":
            return PrototypeReadout(count, dim)
        if self.name == "h2":
            mode = self.assets.get("gate_mode", "learned")
            return EvidenceMessages(count, dim, self.assets["edges"], "learned" if mode == "shuffled" else mode)
        if self.name == "h3":
            return CardinalitySetHead(dim, count)
        if self.name == "h4":
            return DecisionResidual(dim)
        if self.name == "h5":
            mode = self.assets.get("joint_mode", "both")
            if mode == "weighted_bce":
                return ReweightedPairReadout(dim, self.assets["pairs"])
            return PairJointHead(dim, self.assets["pairs"], distinction=mode != "interaction",
                                 interaction=mode != "distinction")
        raise ValueError(f"{self.name} has no deployment head")

    def forward(self, tokens, mask, threshold=0., temperature=1.):
        encoded = self.backbone(tokens, mask)
        rank = decision = encoded.logits.float()
        extra = {}
        if self.head is not None:
            evidence, summary = encoded.evidence, encoded.summary
            if self.name in ("h1", "h2"):
                rank = decision = self.head(evidence, rank).float()
            elif self.name == "h3":
                from .hypothesis_set import cardinality_output
                counts, energies = self.head(evidence, summary, rank)
                return cardinality_output(rank, counts, energies, temperature)
            elif self.name == "h4":
                decision, _ = self.head(evidence, summary, rank)
            elif self.name == "h5":
                if self.assets.get("joint_mode") == "weighted_bce":
                    rank = decision = self.head(evidence, rank)
                else:
                    rank, state_logits = self.head(evidence, rank)
                    decision = rank
                    extra["pair_probabilities"] = state_logits.float().softmax(dim=-1)
        return {"rank_scores": rank, "decision_scores": decision, "predicted": decision >= threshold, **extra}
