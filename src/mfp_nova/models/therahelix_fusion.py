"""Single model loaded from the trained TheraSet and compact HelixSet readers."""

import torch
from torch import nn

from ..research_layers import residue_distances
from .helixset import HelixSet
from .theraset import TheraSet
from .trievidence import center_evidence


class TheraHelixFusion(nn.Module):
    def __init__(self, theraset_args, helixset_args):
        super().__init__()
        self.theraset = TheraSet(**theraset_args)
        self.helixset = HelixSet(**helixset_args, direct_only=True)
        self.register_buffer("offsets", torch.zeros(2))

    def forward(self, batch, reuse_inputs=True):
        if reuse_inputs:
            batch = dict(batch)
            batch["_trievidence_dual_esm"] = self.theraset.sequence.normalized_dual_esm(batch)
            if self.theraset.structure_evidence and self.helixset.structure_evidence:
                structural = batch["struct_mask"] & batch["mask"]
                batch["_trievidence_distances"] = residue_distances(batch, structural)
        theraset = self.theraset(batch)["logits"]
        helixset = self.helixset(batch)["logits"]
        logits = center_evidence(torch.stack((theraset, helixset)), self.offsets).mean(0)
        return {"logits": logits, "theraset_logits": theraset, "latent_logits": helixset}
