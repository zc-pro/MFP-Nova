"""TheraSet-Bridge with train-only transfer between recorded Fine39 positives."""

import torch
from torch.nn import functional as F

from ..consistency import bernoulli_rdrop
from ..research_layers import training_views
from .theraset import TheraSet


class TheraSetBridge(TheraSet):
    """Reuse the single-prototype TheraSet backbone and own the bridge objective."""

    def __init__(self, *, prototype_readout=False, label_context=False, evidence_decoder=True,
                 b0_backbone=False, order_evidence=False, v5_mode=None, **arguments):
        if (prototype_readout or label_context or not evidence_decoder
                or b0_backbone or order_evidence or v5_mode is not None):
            raise ValueError("TheraSet-Bridge requires the original Fine39 V5 baseline backbone")
        super().__init__(prototype_readout=False, label_context=False, **arguments)

    def compute_loss(self, batch, settings, return_output=False):
        weight = settings.get("bridge", 0.)
        if not self.training or not weight:
            return super().compute_loss(batch, settings, return_output)

        view = training_views(batch, settings.get("modality_dropout", .1), 0.)[0]
        output = self(view, return_states=True)
        targets = batch["y"]
        unknown_weight = settings.get("unlabeled_weight", .3)
        terms = self.supervised_terms(output, targets, unknown_weight)
        second = output
        agreement = output["logits"].sum() * 0
        rdrop = settings.get("rdrop", 0.)
        if rdrop:
            second = self(view, return_states=True)
            other_terms = self.supervised_terms(second, targets, unknown_weight)
            terms = {key: .5 * (value + other_terms[key]) for key, value in terms.items()}
            agreement = bernoulli_rdrop(output["logits"], second["logits"])

        bridge, bridge_terms = cross_cluster_bridge(output, second, targets, batch["xbridge_cluster"],
                                                   self.label_weights, self.prior,
                                                   settings.get("bridge_margin", .5), settings["theraset_bridge"])
        total = (terms["classification"] + settings.get("direct", .2) * terms["direct"]
                 + settings.get("evidence", .1) * terms["evidence"] + rdrop * agreement + weight * bridge)
        terms.update(rdrop=agreement, **bridge_terms)
        return (total, terms, output) if return_output else (total, terms)


def cross_cluster_bridge(output, other, targets, clusters, label_weights, prior, margin, mode):
    with torch.autocast(device_type=targets.device.type, enabled=False):
        evidence = output["evidence_state"].float().transpose(0, 1)
        unit = F.normalize(evidence, dim=-1, eps=1e-6)
        cosine = unit @ unit.detach().transpose(1, 2)

        with torch.no_grad():
            recorded = targets.bool().T
            score = output["logits"].float().T
            other_score = other["logits"].float().T
            evidence_score = torch.minimum(output["evidence_logits"], other["evidence_logits"]).float().T
            donors = recorded & (torch.minimum(score, other_score) >= 0) & (evidence_score > prior[:, None])
            recipients = recorded & (torch.maximum(score, other_score) < 0)
            _, group = torch.unique(clusters, return_inverse=True)
            different = group[:, None] != group[None, :]
            candidates = recorded[:, :, None] & recorded[:, None, :] & different
            directed = recipients[:, :, None] & donors[:, None, :] & different
            # Directional derivative of the single-prototype evidence score.
            axis = (unit * label_weights.float()[:, None]).sum(-1)
            helps = axis[:, None, :] - cosine.detach() * axis[:, :, None] > 0
            eligible = directed & helps
            active = eligible & (cosine < margin)

        value = F.relu(margin - cosine).square()
        loss, cluster_pairs, labels = cluster_balanced_mean(value, eligible, active, group)
        control_scale = loss.detach().new_ones(())
        if mode == "positive":
            score = prior[:, None] + (evidence * label_weights.detach().float()[:, None]).sum(-1)
            control, _, _ = cluster_balanced_mean(F.softplus(-score)[:, :, None].expand_as(cosine),
                                                   eligible, active, group)
            align_grad, = torch.autograd.grad(loss, evidence, retain_graph=True)
            control_grad, = torch.autograd.grad(control, evidence, retain_graph=True)
            control_scale = align_grad.norm() / control_grad.norm().clamp_min(1e-8)
            loss = control_scale * control
        terms = {
            "bridge": loss, "bridge_candidates": candidates.sum() / 2,
            "bridge_directed": directed.sum(), "bridge_pairs": eligible.sum(),
            "bridge_cluster_pairs": cluster_pairs, "bridge_labels": labels,
            "bridge_active": active.sum() / eligible.sum().clamp_min(1),
            "bridge_control_scale": control_scale,
            "evidence_std": evidence.detach().std(1, unbiased=False).mean(),
        }
    return loss, terms


def cluster_balanced_mean(values, eligible, active, groups):
    size = len(groups)
    lower = torch.minimum(groups[:, None], groups[None, :])
    upper = torch.maximum(groups[:, None], groups[None, :])
    buckets = (lower * size + upper).flatten().expand(values.shape[0], -1)
    totals = values.new_zeros((values.shape[0], size * size))
    counts = torch.zeros_like(totals)
    totals.scatter_add_(1, buckets, (values * active).flatten(1))
    counts.scatter_add_(1, buckets, eligible.flatten(1).float())
    pairs = (counts > 0).sum(1)
    labels = (pairs > 0).sum()
    per_label = (totals / counts.clamp_min(1)).sum(1) / pairs.clamp_min(1)
    return per_label.sum() / labels.clamp_min(1), pairs.sum(), labels
