"""Multimodal evidence and a cardinality-conditioned distribution over label sets."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .sequence_features import prior_bias
from .theraset import TheraSet
from ..research_layers import observed_label_loss


def log_elementary_prefix(log_weights, degree=None):
    """At each prefix store only the reachable polynomial degrees."""
    count = log_weights.shape[-1]
    degree = count if degree is None else degree
    prefixes = [log_weights.new_zeros((*log_weights.shape[:-1], 1))]
    for log_weight in log_weights.unbind(-1):
        previous = prefixes[-1]
        included = previous + log_weight[..., None]
        middle = torch.logaddexp(previous[..., 1:], included[..., :-1])
        tail = included[..., -1:] if previous.shape[-1] <= degree else included[..., :0]
        prefixes.append(torch.cat((previous[..., :1], middle, tail), dim=-1))
    return prefixes


def observed_set_nll(energies, count_logits, mixture_logits, target):
    energies = energies.float()
    cardinality = target.sum(-1).long()
    normalization = log_elementary_prefix(energies, int(cardinality.max()))[-1]
    index = cardinality[:, None, None].expand(-1, energies.shape[1], 1)
    log_normalizer = normalization.gather(-1, index).squeeze(-1)
    log_count = count_logits.float().log_softmax(-1).gather(-1, index - 1).squeeze(-1)
    observed = (energies * target[:, None]).sum(-1)
    log_component = mixture_logits.float().log_softmax(-1) + log_count + observed - log_normalizer
    return -torch.logsumexp(log_component, dim=-1)


def label_marginals(energies, prefixes, count_probabilities, mixture_probabilities):
    """Propagate count probability backwards through the subset DP."""
    mass = F.pad(count_probabilities, (1, 0))
    labels = []
    for j in range(energies.shape[-1], 0, -1):
        previous, current = prefixes[j - 1], prefixes[j]
        included = mass[..., 1:] * (previous + energies[..., j - 1, None] - current[..., 1:]).exp()
        excluded = mass[..., :-1] * (previous - current[..., :-1]).exp()
        labels.append((included.sum(-1) * mixture_probabilities).sum(-1))
        mass = included + excluded
    return torch.stack(labels[::-1], dim=-1).clamp(0., 1.)


def expected_jaccard_prefixes(energies, log_count, log_mixture, normalizer, order, max_size):
    """Exact expected Jaccard for each size of each supplied label ordering."""
    classes = energies.shape[-1]
    sorted_energies = energies[:, None].expand(-1, order.shape[1], -1, -1).gather(
        -1, order[:, :, None].expand(-1, -1, energies.shape[1], -1))
    front = log_elementary_prefix(sorted_energies, max_size)
    back = log_elementary_prefix(sorted_energies.flip(-1), classes - 1)
    log_coefficient = log_mixture[..., None] + log_count - normalizer[..., 1:]
    labels = torch.arange(classes + 1, device=order.device)
    values = []
    for size in range(1, max_size + 1):
        inside, outside = labels[1:size + 1, None], labels[None, :classes - size + 1]
        coefficient = log_coefficient[:, None, :, inside + outside - 1]
        log_intersection = front[size][..., 1:, None]
        log_remaining = back[classes - size][..., None, :]
        log_utility = inside.float().log() - (size + outside).float().log()
        total = coefficient + log_intersection + log_remaining + log_utility
        values.append(torch.logsumexp(total, dim=(-3, -2, -1)).exp())
    return torch.stack(values, dim=1)


class FragmentEvidence(nn.Module):
    def __init__(self, backbone_dim, dim, components, labels, max_contacts):
        super().__init__()
        self.max_contacts = max_contacts
        self.fragment = nn.Linear(3 * backbone_dim + 3, dim)
        self.contact = nn.Linear(2 * backbone_dim + 6, dim)
        self.queries = nn.Parameter(torch.randn(components, labels, dim) / math.sqrt(dim))

    def read(self, tokens, mask):
        shape = (len(tokens), *self.queries.shape[:-1], tokens.shape[-1])
        queries = self.queries.flatten(0, 1)
        attention = (queries @ tokens.transpose(-1, -2)).float() / math.sqrt(tokens.shape[-1])
        attention = attention.masked_fill(~mask[:, None], -1e9).softmax(-1) * mask[:, None]
        return (attention.to(tokens.dtype) @ tokens).reshape(shape)

    def fragments(self, residues, mask):
        length = mask.sum(-1, keepdim=True)
        starts = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        rows = torch.arange(len(mask), device=mask.device)[:, None]
        prefix = F.pad(residues.float().cumsum(1), (0, 0, 1, 0))
        tokens, visible = [], []
        for width in (3, 7):
            sizes = length.clamp(max=width)
            ends = starts + sizes
            valid = ends <= length
            if width == 7:
                valid = valid & (length > 3)
            ends = ends.clamp(max=mask.shape[1])
            mean = (prefix[rows, ends] - prefix[rows, starts]) / sizes.clamp_min(1)[..., None]
            locations = torch.stack((starts / length, (length - ends) / length,
                                     sizes.expand_as(starts) / length), dim=-1)
            features = torch.cat((mean, residues, residues[rows, ends - 1], locations), dim=-1)
            tokens.append(F.gelu(self.fragment(features)).masked_fill(~valid[..., None], 0))
            visible.append(valid)
        return self.read(torch.cat(tokens, dim=1), torch.cat(visible, dim=1))

    def contacts(self, residues, batch):
        coordinates = batch["coords"].float()
        positions = torch.arange(residues.shape[1], device=residues.device)
        separation = positions[None, :] - positions[:, None]
        geometry_mask = batch["mask"] & batch["struct_mask"]
        pairs = ((batch["edge_type"] > 0) & (separation > 1)
                 & geometry_mask[:, :, None] & geometry_mask[:, None, :])
        distance = torch.linalg.vector_norm(coordinates[:, :, None] - coordinates[:, None, :], dim=-1)
        flat = distance.masked_fill(~pairs, torch.inf).flatten(1)
        distances, indices = flat.topk(min(self.max_contacts, flat.shape[-1]), largest=False)
        valid = torch.isfinite(distances)
        indices = indices.masked_fill(~valid, 0)
        left, right = indices // len(positions), indices % len(positions)
        rows = torch.arange(len(residues), device=residues.device)[:, None]
        known = batch["confidence_known"] & geometry_mask
        quality = batch["geometry"][..., 9].float().masked_fill(~known, 0)
        descriptors = torch.stack((
            (right - left) / (batch["mask"].sum(-1, keepdim=True) - 1).clamp_min(1),
            distances.masked_fill(~valid, 0.) / 8., quality[rows, left], quality[rows, right],
            known[rows, left], known[rows, right],
        ), dim=-1)
        features = torch.cat((residues[rows, left], residues[rows, right], descriptors), dim=-1)
        tokens = F.gelu(self.contact(features)).masked_fill(~valid[..., None], 0)
        return self.read(tokens, valid)


class HHSetRereadV16(nn.Module):
    def __init__(self, statistics, num_labels, components=3, fragments=True, fragment_dim=128,
                 max_contacts=32, max_output_size=14, count_support=None, **backbone_args):
        super().__init__()
        dim = backbone_args.get("dim", 256)
        self.statistics = statistics
        self.max_output_size = min(max_output_size, num_labels)
        self.backbone = TheraSet(statistics=statistics, num_labels=num_labels, **backbone_args)
        self.backbone_frozen = False
        self.state = nn.Linear(dim, fragment_dim)
        self.global_projection = nn.Linear(dim, fragment_dim)
        self.norm = nn.LayerNorm(fragment_dim)
        self.label_weights = nn.Parameter(torch.randn(components, num_labels, fragment_dim) / math.sqrt(fragment_dim))
        self.label_bias = nn.Parameter(prior_bias(statistics).expand(components, -1).clone())
        self.mixture = nn.Linear(dim, components)
        self.count = nn.Linear(dim, components * num_labels)
        if count_support is not None:
            nn.init.zeros_(self.count.weight)
            with torch.no_grad():
                prior = (torch.as_tensor(count_support, dtype=self.count.bias.dtype) + .5).log()
                self.count.bias.copy_(prior.repeat(components))
        if fragments:
            self.fragment_evidence = FragmentEvidence(dim, fragment_dim, components, num_labels, max_contacts)
            self.fragment_fusion = nn.Linear(fragment_dim, fragment_dim, bias=False)
            self.contact_fusion = nn.Linear(fragment_dim, fragment_dim, bias=False)
            self.fragment_weights = nn.Parameter(torch.randn(num_labels, fragment_dim) / math.sqrt(fragment_dim))
        else:
            self.fragment_evidence = None

    def freeze_backbone(self, frozen):
        self.backbone_frozen = frozen
        self.backbone.requires_grad_(not frozen).train(self.training and not frozen)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, batch):
        encoded = self.backbone(batch, return_states=True)
        summary = encoded["global_state"]
        read = self.state(encoded["evidence_state"])[:, None] + self.global_projection(summary)[:, None, None]
        output = {"logits": encoded["logits"]}
        if self.fragment_evidence is not None:
            residues = encoded["residues"]
            fragment = self.fragment_fusion(self.fragment_evidence.fragments(residues, batch["mask"]))
            contact = self.contact_fusion(self.fragment_evidence.contacts(residues, batch))
            read = read + fragment + contact
            output["fragment_logits"] = ((fragment + contact).mean(1) * self.fragment_weights).sum(-1)
        energies = (self.norm(read) * self.label_weights).sum(-1) + self.label_bias
        mixture_logits = self.mixture(summary)
        output.update(energies=energies, mixture_logits=mixture_logits,
                      count_logits=self.count(summary).reshape_as(energies))
        return output

    def set_nll(self, output, labels):
        with torch.autocast(device_type=labels.device.type, enabled=False):
            return observed_set_nll(output["energies"], output["count_logits"], output["mixture_logits"], labels)

    def compute_loss(self, batch, settings, return_output=False):
        output = self(batch)
        target = batch["y"].float()
        nll = self.set_nll(output, target)
        teacher = batch["base_logits"] >= 0 if "base_logits" in batch else None
        if teacher is None:
            joint = nll.mean()
        else:
            overlap = (teacher * target).sum(-1)
            union = teacher.sum(-1) + target.sum(-1) - overlap
            weights = 1. + settings.get("hard_row_weight", .5) * (1. - overlap / union.clamp_min(1))
            joint = (weights * nll).sum() / weights.sum()
        weak = observed_label_loss(output["logits"], target, self.statistics, settings["unlabeled_weight"])
        terms = {"joint_nll": joint, "weak": weak}
        loss = joint + settings["weak"] * weak
        if self.fragment_evidence is not None:
            fragment = observed_label_loss(output["fragment_logits"], target, self.statistics, settings["unlabeled_weight"])
            terms["fragment"] = fragment
            loss = loss + settings["fragment"] * fragment
        if teacher is not None and settings.get("retain", 0.):
            exact = (teacher == target).all(-1)
            t = batch["base_logits"].float() / 2.
            student = output["logits"].float() / 2.
            # Bernoulli KL from the OOF teacher, minus its constant entropy.
            kl = F.binary_cross_entropy_with_logits(student, t.sigmoid(), reduction="none")
            kl = kl - F.binary_cross_entropy_with_logits(t, t.sigmoid(), reduction="none")
            retain = (kl.mean(-1) * exact).sum() / exact.sum().clamp_min(1)
            terms["retain"] = retain
            loss = loss + settings["retain"] * retain
        return (loss, terms, output) if return_output else (loss, terms)

    @torch.inference_mode()
    def decode(self, output):
        energies = output["energies"].float()
        log_count = output["count_logits"].float().log_softmax(-1)
        log_mixture = output["mixture_logits"].float().log_softmax(-1)
        prefixes = log_elementary_prefix(energies)
        marginal = label_marginals(energies, prefixes, log_count.exp(), log_mixture.exp())
        order = torch.cat((marginal[:, None], energies), dim=1).argsort(dim=-1, descending=True, stable=True)
        utility = expected_jaccard_prefixes(energies, log_count, log_mixture, prefixes[-1], order, self.max_output_size)
        choice = utility.flatten(1).argmax(-1)
        size = choice // order.shape[1] + 1
        sorted_labels = order[torch.arange(len(order), device=order.device), choice % order.shape[1]]
        selected = (torch.arange(energies.shape[-1], device=order.device)[None] < size[:, None])
        predicted = torch.zeros_like(marginal, dtype=torch.bool).scatter_(1, sorted_labels, selected)
        count_mode = torch.logsumexp(log_mixture[..., None] + log_count, dim=1).argmax(-1) + 1
        by_count = torch.arange(energies.shape[-1], device=order.device)[None] < count_mode[:, None]
        count_predicted = torch.zeros_like(predicted).scatter_(1, order[:, 0], by_count)
        return marginal, predicted, count_predicted
