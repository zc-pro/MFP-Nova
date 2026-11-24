"""Single model loaded from the trained TheraSet and compact HelixSet readers."""

import torch
from torch import nn

from ..research_layers import Attention, residue_distances
from .helixset import HelixSet
from .theraset import TheraSet
from .trievidence import center_evidence


class TheraHelixFusion(nn.Module):
    def __init__(self, theraset_args, helixset_args, interaction=False):
        super().__init__()
        self.theraset = TheraSet(**theraset_args)
        self.helixset = HelixSet(**helixset_args, direct_only=True)
        self.register_buffer("offsets", torch.zeros(2))
        if interaction:
            dim = theraset_args.get("dim", 256)
            latent_dim = helixset_args.get("dim", 384)
            self.latent_projection = nn.Linear(latent_dim, dim)
            self.interaction = Attention(dim, theraset_args.get("heads", 8), 0.)
            self.interaction_output = nn.Linear(dim, 1)
            nn.init.zeros_(self.interaction_output.weight)
            nn.init.zeros_(self.interaction_output.bias)

    def train(self, mode=True):
        super().train(mode)
        self.theraset.eval()
        self.helixset.eval()
        return self

    def forward(self, batch, reuse_inputs=True):
        if reuse_inputs:
            batch = dict(batch)
            batch["_trievidence_dual_esm"] = self.theraset.sequence.normalized_dual_esm(batch)
            if self.theraset.structure_evidence and self.helixset.structure_evidence:
                structural = batch["struct_mask"] & batch["mask"]
                batch["_trievidence_distances"] = residue_distances(batch, structural)
        with torch.no_grad():
            thera_output = self.theraset(batch, return_states=hasattr(self, "interaction"))
            helix_output = self.helixset(batch, return_evidence=hasattr(self, "interaction"))
        theraset = thera_output["logits"]
        helixset = helix_output["logits"]
        logits = center_evidence(torch.stack((theraset, helixset)), self.offsets).mean(0)
        baseline = logits
        if hasattr(self, "interaction"):
            queries = thera_output["evidence_state"]
            latent = self.latent_projection(helix_output["latent"])
            qmask = torch.ones(queries.shape[:2], device=queries.device, dtype=torch.bool)
            lmask = torch.ones(latent.shape[:2], device=latent.device, dtype=torch.bool)
            read = self.interaction(queries, latent, qmask, lmask)
            logits = logits + self.interaction_output(read).squeeze(-1)
        return {"logits": logits, "baseline_logits": baseline,
                "theraset_logits": theraset, "latent_logits": helixset}
