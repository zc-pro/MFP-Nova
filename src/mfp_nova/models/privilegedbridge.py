"""PrivilegedBridge: staged, observed-subset distillation with sequence-only deployment."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from .sequence_features import FrozenFeatures, ResidueCNN, prior_bias
from ..research_layers import (Attention, TransformerBlock, PropertyEncoder, masked_mean,
                               observed_label_loss, unlabeled_weights)
from ..consistency import bernoulli_rdrop


class SequenceBase(nn.Module):
    def __init__(self, input_dim, num_labels, statistics, dim, dropout, peptide_input_dim):
        super().__init__()
        self.dim = dim
        self.cnn = ResidueCNN(dropout)
        self.features = FrozenFeatures(input_dim, dim, peptide_input_dim)
        self.cnn_projection = nn.Linear(256, dim)
        self.position = nn.Embedding(256, dim)
        self.norm = nn.LayerNorm(dim)
        self.block = TransformerBlock(dim, 6, dropout)
        self.query = nn.Parameter(torch.randn(num_labels, dim)*.02)
        self.read = Attention(dim, 6, dropout)
        self.label_weight = nn.Parameter(torch.randn(num_labels, dim)*.02)
        self.bias = nn.Parameter(prior_bias(statistics))

    def forward(self, batch):
        mask = batch["mask"]
        x = self.features(batch)+self.cnn_projection(self.cnn(batch))
        x = self.norm(x+self.position(torch.arange(mask.shape[1], device=x.device)))
        h = self.block(x.masked_fill(~mask[..., None], 0), mask)
        query = self.query[None].expand(h.shape[0], -1, -1)
        r = self.read(query, h, torch.ones(query.shape[:2], dtype=torch.bool, device=h.device), mask)
        z = (r*self.label_weight).sum(-1)/math.sqrt(self.dim)+self.bias
        return {"logits": z, "labels": r, "global": masked_mean(h, mask)}


class EvidenceTeacher(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.mode = mode
        self.properties = PropertyEncoder(128, 4, 0., depth=0)
        self.geometry, self.dssp = nn.Linear(16, 128), nn.Embedding(10, 128)
        self.global_structure = nn.Linear(50, 128)
        self.queries = nn.ModuleList([nn.Linear(dim, 128) for _ in range(2)])
        self.reads = nn.ModuleList([Attention(128, 4, 0.) for _ in range(2)])
        self.compress = nn.ModuleList([nn.Linear(128, 64) for _ in range(2)])
        self.residual = nn.ModuleList([nn.Sequential(nn.Linear(dim+64, 128), nn.GELU(), nn.Linear(128, 1))
                                       for _ in range(2)])

    def forward(self, batch, reference):
        view = dict(batch)
        view["phys"] = batch["phys"].masked_fill(~batch["phys_mask"], 0)
        geometry = batch["geometry"].masked_fill(~batch["struct_mask"][..., None], 0)
        global_values = batch["struct_global"].masked_fill(~batch["struct_global_mask"], 0)
        dssp = batch["dssp"].masked_fill(~batch["struct_mask"], 9)
        if self.mode == "mask_only":
            view["phys"], geometry, global_values = torch.zeros_like(view["phys"]), torch.zeros_like(geometry), torch.zeros_like(global_values)
            dssp = torch.full_like(dssp, 9)
        prop, prop_mask = self.properties(view)
        global_token = self.global_structure(torch.cat([global_values, batch["struct_global_mask"].float()], -1))
        geom_mask = torch.cat([batch["struct_mask"], batch["struct_global_mask"].any(1, keepdim=True)], 1)
        geom = torch.cat([self.geometry(geometry)+self.dssp(dssp), global_token[:, None]], 1).masked_fill(~geom_mask[..., None], 0)
        observed = torch.stack([prop_mask.any(1), geom_mask.any(1)], -1)
        # Confidence is only an input quality weight, not teacher correctness.
        q = torch.where(batch["confidence_known"] & batch["struct_mask"],
                        .3+.7*batch["geometry"][..., 9].clamp(0, 1), 1.)
        quality = torch.cat([q, torch.ones(q.shape[0], 1, device=q.device)], 1)
        rho_g = (quality*geom_mask).sum(1)/geom_mask.sum(1).clamp_min(1)
        rho = torch.stack([torch.ones_like(rho_g), rho_g], -1)
        r = reference["labels"]
        residuals = []
        query_mask = torch.ones(r.shape[:2], dtype=torch.bool, device=r.device)
        for m, (tokens, mask) in enumerate([(prop, prop_mask), (geom, geom_mask)]):
            query = self.queries[m](r)
            bias = None
            if m == 1:
                bias = quality.clamp_min(1e-8).log()[:, None, None].expand(-1, 4, r.shape[1], -1)
            u = self.compress[m](self.reads[m](query, tokens, query_mask, mask, bias))
            u = u*observed[:, m, None, None]
            f = self.residual[m]
            delta = f(torch.cat([r, u], -1))
            if self.mode != "ordinary":
                delta = delta-f(torch.cat([r, torch.zeros_like(u)], -1))
            residuals.append(2*torch.tanh(delta.squeeze(-1)/2)*observed[:, m, None])
        residuals = torch.stack(residuals, -1)
        return {"logits": reference["logits"]+residuals.sum(-1), "residuals": residuals,
                "observed": observed, "quality": rho, "reference_logits": reference["logits"]}


class PrivilegedBridge(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=192, dropout=.2,
                 peptide_input_dim=0, stage="A", teacher_mode="zero_reference", kd_mode="prediction",
                 fixed_student=False, deployment_only=False):
        super().__init__()
        if stage not in ("A", "B", "C") or dim % 6:
            raise ValueError("PB requires stage A/B/C and a width divisible by six")
        if teacher_mode not in ("zero_reference", "mask_only", "ordinary") or kd_mode not in ("prediction", "component", "none"):
            raise ValueError("Unknown PB teacher/KD mode")
        if deployment_only and stage != "C":
            raise ValueError("Only stage C has a compact deployment model")
        self.statistics, self.stage = statistics, stage
        self.teacher_mode, self.kd_mode = teacher_mode, kd_mode
        self.fixed_student, self.deployment_only = fixed_student, deployment_only
        self.sequence = SequenceBase(input_dim, num_labels, statistics, dim, dropout, peptide_input_dim)
        if stage in ("B", "C") and not deployment_only:
            self.teacher = EvidenceTeacher(dim, teacher_mode)
        if stage == "C":
            if not deployment_only:
                self.reference = copy.deepcopy(self.sequence)
            self.bridges = nn.ModuleList([nn.Sequential(nn.Linear(2*dim, 128), nn.GELU(), nn.Linear(128, 1))
                                         for _ in range(2)])
            for bridge in self.bridges:
                nn.init.zeros_(bridge[-1].weight)
                nn.init.zeros_(bridge[-1].bias)
        self._set_trainable()

    def _set_trainable(self):
        if self.stage == "B":
            self.sequence.requires_grad_(False)
        if self.stage == "C":
            if hasattr(self, "reference"):
                self.reference.requires_grad_(False)
                self.teacher.requires_grad_(False)
            if self.fixed_student:
                self.sequence.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.stage == "B" or self.fixed_student:
            self.sequence.eval()
        if self.stage == "C" and hasattr(self, "reference"):
            self.reference.eval()
            self.teacher.eval()
        return self

    def initialize_from(self, previous):
        expected = "A" if self.stage == "B" else "B"
        if previous.stage != expected or self.stage == "A":
            raise ValueError(f"Stage {self.stage} must initialize from {expected}")
        self.sequence.load_state_dict(previous.sequence.state_dict())
        if self.stage == "C":
            self.reference.load_state_dict(previous.sequence.state_dict())
            self.teacher.load_state_dict(previous.teacher.state_dict())
        self.train(self.training)

    def forward(self, batch, return_evidence=False):
        if self.stage == "B":
            with torch.no_grad():
                reference = self.sequence(batch)
            return self.teacher(batch, reference)
        base = self.sequence(batch)
        if self.stage == "A":
            return base
        x = torch.cat([base["labels"], base["global"][:, None].expand_as(base["labels"])], -1)
        residuals = torch.stack([2*torch.tanh(bridge(x).squeeze(-1)/2) for bridge in self.bridges], -1)
        # No optional modality or observation mask is accessed on this route.
        return {"logits": base["logits"]+residuals.sum(-1), "sequence_logits": base["logits"], "residuals": residuals}

    def teacher_target(self, batch):
        with torch.no_grad():
            reference = self.reference(batch)
            return self.teacher(batch, reference)

    def compute_loss(self, batch, settings, return_output=False):
        if self.deployment_only:
            raise ValueError("Deployment exports do not contain a training teacher")
        unknown = settings.get("unlabeled_weight", .3)
        output = self(batch)
        classify = lambda z, mask=None: observed_label_loss(z, batch["y"], self.statistics, unknown, mask)
        if self.stage == "B":
            present = output["observed"]
            branches = []
            z0 = output["reference_logits"]
            for m in range(2):
                if present[:, m].any():
                    mask = present[:, m, None].expand_as(batch["y"])
                    branches.append(classify(z0+output["residuals"][..., m], mask))
            classification = classify(output["logits"])
            branch = torch.stack(branches).mean() if branches else classification*0
            valid = present[:, None].expand_as(output["residuals"])
            penalty = (output["residuals"].square()*valid).sum()/valid.sum().clamp_min(1)
            total = classification+settings.get("branch", .2)*branch+settings.get("residual_penalty", .001)*penalty
            terms = {"classification": classification, "branch": branch, "residual_penalty": penalty}
        else:
            target = self.teacher_target(batch) if self.stage == "C" and self.kd_mode != "none" else None
            def losses(prediction):
                classification = classify(prediction["logits"])
                kd = classification*0
                if target is not None:
                    observed = target["observed"]
                    quality = (target["quality"]*observed).sum(-1)/observed.sum(-1).clamp_min(1)
                    weight = batch["y"]+(1-batch["y"])*unlabeled_weights(batch["y"], self.statistics, unknown)
                    if self.kd_mode == "prediction":
                        actual = prediction["sequence_logits"]+(prediction["residuals"]*observed[:, None]).sum(-1)
                        distance = F.smooth_l1_loss(actual.float(), target["logits"].float(), reduction="none")
                        weight = weight*quality[:, None]
                    else:
                        distance = F.smooth_l1_loss(prediction["residuals"].float(), target["residuals"].float(), reduction="none")
                        weight = weight[..., None]*(observed*target["quality"])[:, None]
                    kd = (distance*weight).sum()/weight.sum().clamp_min(1)
                return classification, kd
            classification, kd = losses(output)
            agreement = classification*0
            if self.training and settings.get("rdrop", 0) > 0:
                second = self(batch)
                other_classification, other_kd = losses(second)
                classification, kd = (classification+other_classification)/2, (kd+other_kd)/2
                agreement = bernoulli_rdrop(output["logits"], second["logits"])
            total = classification+settings.get("kd", .2)*kd+settings.get("rdrop", 0)*agreement
            terms = {"classification": classification, "kd": kd, "rdrop": agreement}
        return (total, terms, output) if return_output else (total, terms)


def export_deployment(checkpoint, output):
    """Strip frozen training references; keep selected weights and input statistics."""
    if checkpoint["model_args"].get("stage") != "C":
        raise ValueError("Deployment export requires a stage C checkpoint")
    result = dict(checkpoint)
    result["model_args"] = {**checkpoint["model_args"], "deployment_only": True}
    result["model_state"] = {k: v for k, v in checkpoint["model_state"].items()
                       if k.startswith(("sequence.", "bridges."))}
    for key in ("optimizer_state", "scheduler_state", "scaler_state", "ema_state", "rng_state"):
        result.pop(key, None)
    result["deployment_only"] = True
    torch.save(result, output)


def configure_stage(config, stage):
    """Explicit phase defaults; CLI execution overrides are applied afterwards."""
    if stage not in ("A", "B", "C"):
        raise ValueError("Unknown PB stage")
    config["model"]["stage"] = stage
    train, loss = config["train"], config["loss"]
    train.update(epochs=100 if stage == "A" else 30, patience=15 if stage == "A" else 8,
                 warmup_epochs=5 if stage == "A" else 2, learning_rate=2e-5 if stage == "C" else 2e-4)
    if stage == "B":
        train.pop("ema_decay", None)
        train.pop("ema_start_epoch", None)
        loss["rdrop"] = 0.
    else:
        train.update(ema_decay=.999, ema_start_epoch=6 if stage == "A" else 3,
                     rdrop_start_epoch=6 if stage == "A" else 3, rdrop_ramp_epochs=5)
        loss["rdrop"] = .1
    if stage == "C":
        train["bridge_learning_rate"] = 2e-4
