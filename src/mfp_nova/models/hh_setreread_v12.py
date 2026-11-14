"""Fine39 evidence rereading with a fixed TheraSet prior."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .theraset import TheraSet
from ..multimodal import function_projection
from ..research_layers import observed_label_loss


class HHSetRereadV12(nn.Module):
    def __init__(self, statistics, input_dim, peptide_input_dim, num_labels, readout="conditional",
                 esm_bypass=True, condition_query=True, split_evidence=True, background_dropout=.25,
                 **backbone_args):
        super().__init__()
        if num_labels != 39 or (input_dim, peptide_input_dim) != (1280, 640):
            raise ValueError("HH V1.2 requires Fine39 and the ESM650M/peptide150M feature pair")
        if readout not in ("conditional", "unconditional", "raw_unconditional", "affine"):
            raise ValueError(f"Unknown HH V1.2 readout: {readout}")
        self.statistics = statistics
        self.readout = readout
        self.condition_query = condition_query
        self.split_evidence = split_evidence
        self.background_dropout = background_dropout
        self.backbone = TheraSet(statistics=statistics, input_dim=input_dim, peptide_input_dim=peptide_input_dim,
                                 num_labels=num_labels, **backbone_args).requires_grad_(False).eval()

        projection = torch.tensor(function_projection(statistics["labels"], "fine39"))
        axis_index = projection.argmax(dim=1)
        axis = torch.arange(projection.shape[1])
        related = (axis_index[:, None] == axis) | ((axis_index[:, None] == 4) & (axis < 4))
        related |= (axis_index[:, None] < 4) & (axis == 4)
        self.register_buffer("projection", projection)
        self.register_buffer("axis_index", axis_index)
        self.register_buffer("background_axes", ~related)

        if readout == "affine":
            self.log_slope = nn.Parameter(torch.zeros(num_labels))
            self.offset = nn.Parameter(torch.zeros(num_labels))
            return

        d = 128
        self.residue_projection = nn.Linear(backbone_args.get("dim", 256), d, bias=False)
        self.esm_projection = nn.Linear(input_dim, d, bias=False) if esm_bypass else None
        self.peptide_projection = nn.Linear(peptide_input_dim, d, bias=False) if esm_bypass else None
        self.residue_norm = nn.LayerNorm(d)
        self.keys = nn.Linear(d, d, bias=False)
        self.axis_queries = nn.Parameter(torch.randn(len(axis), d) * .02)
        self.fine_queries = nn.Parameter(torch.randn(num_labels, d) * .02)
        self.null_logits = nn.Parameter(torch.zeros(num_labels))
        self.background_projection = nn.Linear(d, d, bias=False)
        self.label_weights = nn.Parameter(torch.zeros(num_labels, d))
        self.label_bias = nn.Parameter(torch.zeros(num_labels))
        self.evidence_readout = nn.Sequential(nn.Linear(2*d, 64, bias=False), nn.GELU(),
                                              nn.Linear(64, 1, bias=False))
        nn.init.zeros_(self.evidence_readout[-1].weight)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def attention(self, queries, keys, mask):
        scores = queries @ keys.transpose(-1, -2) / math.sqrt(keys.shape[-1])
        scores = scores.masked_fill(~mask[:, None], -torch.inf)
        null = mask.sum(-1, keepdim=True).clamp_min(1).float().log() + self.null_logits
        return torch.cat((scores, null[..., None]), dim=-1).softmax(-1)[..., :-1]

    @torch.no_grad()
    def background(self, unary, attention, evidence):
        label_weight = (unary / 2).tanh().clamp_min(0)
        membership = self.projection.T.bool()
        axis_weight = (label_weight[:, None] * membership).amax(-1)
        specific_active = ((unary >= 0).float() @ self.projection[:, :4] > 0).any(-1)
        axis_weight[:, 4] *= ~specific_active
        if self.training and self.background_dropout:
            axis_weight *= torch.rand_like(axis_weight) >= self.background_dropout
        group_weight = label_weight @ self.projection
        # Zero support also gives a zero numerator. Keep nonzero weights normalized.
        group_weight = torch.where(group_weight > 0, group_weight, 1.)
        label_mix = self.projection.T * label_weight[:, None] / group_weight[..., None]
        mix = axis_weight[:, None] * self.background_axes
        mix = (mix / mix.sum(-1, keepdim=True).clamp_min(1.)) @ label_mix
        return mix @ evidence, mix @ attention

    def evidence_logits(self, evidence, shared, remaining):
        return ((evidence * self.label_weights).sum(-1) + self.label_bias
                + self.evidence_readout(torch.cat((shared, remaining), dim=-1)).squeeze(-1))

    def forward(self, batch, return_unconditional=False):
        with torch.no_grad():
            base = self.backbone(batch, return_residues=self.readout != "affine")
        mask = batch["mask"]
        with torch.autocast(mask.device.type, enabled=False):
            unary = base["logits"].float()
            prior = (unary / 2).tanh()
            if self.readout == "affine":
                logits = self.log_slope.exp() * unary + self.offset
                return {"logits": logits, "unary": unary, "prior": prior, "unconditional_logits": logits}
            residues = self.residue_projection(base["residues"].float())
            if self.esm_projection is not None:
                primary = F.layer_norm(batch["esm"].float(), (self.esm_projection.in_features,))
                peptide = F.layer_norm(batch["esm_peptide"].float(), (self.peptide_projection.in_features,))
                residues = residues + self.esm_projection(primary) + self.peptide_projection(peptide)
            residues = self.residue_norm(residues).masked_fill(~mask[..., None], 0.)
            keys = self.keys(residues)
            queries = self.axis_queries[self.axis_index] + self.fine_queries
            attention = self.attention(queries, keys, mask)
            evidence = attention @ residues
            output = {"unary": unary, "prior": prior}
            if self.readout != "conditional" or return_unconditional:
                offset = unary if self.readout == "raw_unconditional" else prior
                output["unconditional_logits"] = offset + self.evidence_logits(evidence, torch.zeros_like(evidence), evidence)
            if self.readout != "conditional":
                return {**output, "logits": output["unconditional_logits"]}
            context, occupancy = self.background(unary, attention, evidence)
            conditional = (self.attention(queries + self.background_projection(context), keys, mask)
                           if self.condition_query else attention)
            if self.split_evidence:
                shared_attention = torch.minimum(conditional, occupancy)
                shared = shared_attention @ residues
                remaining = (conditional - shared_attention) @ residues
            else:
                remaining = conditional @ residues
                shared = torch.zeros_like(remaining)
            logits = prior + self.evidence_logits(shared + remaining, shared, remaining)
            return {**output, "logits": logits}

    def compute_loss(self, batch, settings, return_output=False):
        output = self(batch, return_unconditional=True)
        target = batch["y"]
        weak = settings.get("unlabeled_weight", .3)
        loss = observed_label_loss(output["logits"], target, self.statistics, weak)
        terms = {"classification": loss}
        if self.readout == "conditional":
            unconditional = observed_label_loss(output["unconditional_logits"], target, self.statistics, weak)
            terms["unconditional"] = unconditional
            loss = .5 * (loss + unconditional)
        return (loss, terms, output) if return_output else (loss, terms)
