"""Joint and pretrained TheraSet–compact HelixSet fusion."""

from contextlib import nullcontext

import torch
from torch import nn

from ..research_layers import Attention, observed_label_loss, residue_distances, training_views
from .helixset import HelixSet
from .theraset import TheraSet
from .trievidence import center_evidence


class TheraHelixFusion(nn.Module):
    def __init__(self, theraset_args, helixset_args, interaction=False, train_readers=False,
                 branch_thresholds=(.5, .5)):
        super().__init__()
        self.train_readers = train_readers
        self.statistics = theraset_args["statistics"]
        self.theraset = TheraSet(**theraset_args)
        self.helixset = HelixSet(**helixset_args, direct_only=True)
        self.register_buffer("offsets", torch.tensor(branch_thresholds).logit())
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
        if not self.train_readers:
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
        with nullcontext() if self.train_readers else torch.no_grad():
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
        output = {"logits": logits, "baseline_logits": baseline,
                  "theraset_logits": theraset, "latent_logits": helixset}
        if self.train_readers:
            output.update(theraset_direct=thera_output["direct"], theraset_evidence=thera_output["evidence_logits"])
        return output

    def compute_loss(self, batch, settings, return_output=False):
        view = training_views(batch, settings.get("modality_dropout", .1), 0.)[0] if self.training else batch
        output = self(view)
        weights = {
            "logits": 1.,
            "theraset_logits": settings.get("theraset", .5),
            "latent_logits": settings.get("helixset", .5),
            "theraset_direct": settings.get("theraset_direct", .1),
            "theraset_evidence": settings.get("theraset_evidence", .05),
        }
        terms = {
            name: observed_label_loss(output[name], batch["y"], self.statistics,
                                      settings.get("unlabeled_weight", .3))
            for name in weights
        }
        loss = sum(weights[name] * term for name, term in terms.items())
        return (loss, terms, output) if return_output else (loss, terms)
