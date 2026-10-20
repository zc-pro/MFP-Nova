"""TriRelay's optional cross-resolution evidence supervision (15 axes / 39 subtypes)."""
import torch
from torch import nn
from torch.nn import functional as F

from .multimodal import function_projection
from .research_layers import Attention, FeedForward, TailClassifier, observed_label_loss


class FunctionAuxiliary(nn.Module):
    def __init__(self, primary_labels, statistics, dim, heads, dropout):
        super().__init__()
        self.statistics = statistics
        self.primary_is_fine = len(primary_labels) == 39
        primary_track = "fine39" if self.primary_is_fine else "major15"
        primary = torch.tensor(function_projection(primary_labels, primary_track))
        auxiliary = torch.tensor(function_projection(statistics["labels"], statistics["track"]))
        membership = auxiliary @ primary.T
        self.register_buffer("pool", membership/membership.sum(-1, keepdim=True))
        # Family membership biases evidence reading; other functional context stays available.
        self.register_buffer("family_bias", (membership+.1).log())
        fine, major = (primary, auxiliary) if self.primary_is_fine else (auxiliary, primary)
        self.register_buffer("parent_index", (fine @ major.T).argmax(-1))
        self.query = nn.Parameter(torch.randn(len(auxiliary), dim)*.02)
        self.read = Attention(dim, heads, dropout)
        self.refine = FeedForward(dim, 4, dropout)
        self.classifier = TailClassifier(len(auxiliary), dim, statistics)
        self.heads = heads

    def forward(self, evidence):
        query = self.query[None] + torch.einsum("ac,bcd->bad", self.pool, evidence)
        qmask = torch.ones(query.shape[:2], device=query.device, dtype=torch.bool)
        kmask = torch.ones(evidence.shape[:2], device=query.device, dtype=torch.bool)
        bias = self.family_bias[None, None].expand(len(evidence), self.heads, -1, -1)
        return self.classifier(self.refine(query+self.read(query, evidence, qmask, kmask, bias), qmask))

    def loss(self, primary_logits, auxiliary_logits, batch, settings):
        mask = batch["aux_mask"][:, None].expand_as(auxiliary_logits)
        classification = observed_label_loss(auxiliary_logits, batch["aux_y"], self.statistics,
                                             settings.get("unlabeled_weight", .3), mask)
        fine, major = ((primary_logits, auxiliary_logits) if self.primary_is_fine
                       else (auxiliary_logits, primary_logits))
        # Functional presence, not mutually exclusive categories or independent-function count.
        violation = F.relu(fine.float().sigmoid()-major.float().sigmoid()[:, self.parent_index]).square()
        valid = batch["aux_mask"][:, None].expand_as(violation)
        hierarchy = (violation*valid).sum()/valid.sum().clamp_min(1)
        return {"function_aux": classification, "function_hierarchy": hierarchy}
