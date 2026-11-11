"""H3: conditional Bernoulli label sets and generalized F-measure decoding."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CardinalitySetHead(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.count = nn.Sequential(nn.Linear(dim + 1, 128), nn.GELU(),
                                   nn.Linear(128, num_labels + 1))
        self.project = nn.Linear(dim, 32)
        self.count_embedding = nn.Parameter(torch.zeros(num_labels + 1, 32))
        self.register_buffer("support", torch.zeros(num_labels + 1))

    @torch.no_grad()
    def set_support(self, counts):
        self.support.copy_(torch.as_tensor(counts, device=self.support.device))

    def forward(self, evidence, summary, base_logits):
        positive_rate = base_logits.sigmoid().mean(-1, keepdim=True)
        count_logits = self.count(torch.cat((summary, positive_rate), dim=-1))
        projected = self.project(evidence)
        adjustment = torch.einsum("bcd,kd->bkc", projected, self.count_embedding)
        shrinkage = self.support / (self.support + 20)
        energies = base_logits.float().unsqueeze(1) + adjustment.float() * shrinkage[None, :, None] / math.sqrt(32)
        return count_logits, energies


def count_support(targets, observed, groups):
    """Count distinct training families for each reliably known cardinality."""
    targets, observed, groups = map(np.asarray, (targets, observed, groups))
    complete = observed.all(axis=1)
    sizes = targets.sum(axis=1)
    return np.asarray([np.unique(groups[complete & (sizes == size)]).size
                       for size in range(targets.shape[1] + 1)])


def _prefix_partitions(energies, limit=None):
    """Log elementary symmetric polynomials using reachable states only."""
    energies = energies.float()
    partitions = [energies.new_zeros((*energies.shape[:-1], 1))]
    for energy in energies.unbind(-1):
        previous = partitions[-1]
        include = previous + energy.unsqueeze(-1)
        interior = torch.logaddexp(previous[..., 1:], include[..., :-1])
        if limit is None or previous.shape[-1] <= limit:
            current = torch.cat((previous[..., :1], interior, include[..., -1:]), dim=-1)
        else:
            current = torch.cat((previous[..., :1], interior), dim=-1)
        partitions.append(current)
    return partitions


def cardinality_nll(count_logits, energies, targets, observed, partial=None, reduction="mean"):
    """Exact likelihood for complete rows and explicitly audited partial rows.

    ``partial`` selects rows whose missingness is supported as ignorable. The
    likelihood for such rows sums over every compatible unknown-label subset.
    """
    count_log_probability = count_logits.float().log_softmax(-1)
    complete = observed.all(dim=-1)
    known_positive = targets.bool() & observed
    size = known_positive.sum(-1)
    rows = torch.where(complete)[0]
    complete_size = size[rows]
    loss = -count_log_probability[rows, complete_size].sum()
    # Empty and full sets are deterministic conditional on their size.
    mixed = (complete_size > 0) & (complete_size < targets.shape[-1])
    mixed_rows, mixed_size = rows[mixed], complete_size[mixed]
    if mixed_rows.numel():
        energy = energies[mixed_rows, mixed_size].float()
        energy = energy - energy.mean(-1, keepdim=True)
        partitions = _prefix_partitions(energy, int(mixed_size.max()))[-1]
        log_z = partitions.gather(-1, mixed_size[:, None]).squeeze(-1)
        loss = loss + (log_z - (energy * known_positive[mixed_rows]).sum(-1)).sum()

    n = rows.numel()
    if partial is not None:
        partial_rows = torch.where(partial & ~complete & observed.any(dim=-1))[0]
        for row in partial_rows.unbind():
            positive = known_positive[row]
            unknown = ~observed[row]
            possible = torch.arange(targets.shape[-1] + 1, device=targets.device)
            extra = possible - size[row]
            allowed = (extra >= 0) & (extra <= unknown.sum())
            counts, extra = possible[allowed], extra[allowed]
            energy = energies[row, counts].float()
            energy = energy - energy.mean(-1, keepdim=True)
            log_z = _prefix_partitions(energy)[-1].gather(-1, counts[:, None]).squeeze(-1)
            unknown_z = _prefix_partitions(energy[:, unknown])[-1].gather(-1, extra[:, None]).squeeze(-1)
            log_prob = count_log_probability[row, counts] + energy[:, positive].sum(-1) + unknown_z - log_z
            loss = loss - log_prob.logsumexp(0)
        n += partial_rows.numel()
    return loss if reduction == "sum" else loss / max(n, 1)


def conditional_marginals(energies):
    """Return P(Y_c = 1 | K=k, X) with shape [batch, label, count]."""
    batch, _, labels = energies.shape
    if labels == 1:
        return torch.cat((energies.new_zeros((batch, 1, 1), dtype=torch.float32),
                          energies.new_ones((batch, 1, 1), dtype=torch.float32)), dim=-1)

    energy = energies[:, 1:labels].float()
    energy = energy - energy.mean(-1, keepdim=True)
    prefix = _prefix_partitions(energy)
    suffix = _prefix_partitions(energy.flip(-1))
    sizes = torch.arange(1, labels, device=energy.device)
    log_z = prefix[-1].gather(-1, sizes[None, :, None].expand(batch, -1, -1)).squeeze(-1)
    marginals = []
    for label in range(labels):
        left, right = prefix[label], suffix[labels - label - 1]
        left_size = torch.arange(left.shape[-1], device=energy.device)
        right_size = sizes[:, None] - 1 - left_size
        feasible = (right_size >= 0) & (right_size < right.shape[-1])
        right_indices = right_size.clamp(0, right.shape[-1] - 1)
        other = left + right.gather(-1, right_indices.expand(batch, -1, -1))
        log_z_other = other.masked_fill(~feasible, -torch.inf).logsumexp(-1)
        marginal = (energy[..., label] + log_z_other - log_z).clamp_max(0).exp()
        marginals.append(marginal)

    interior = torch.stack(marginals, dim=1)
    return torch.cat((interior.new_zeros((batch, labels, 1)), interior,
                      interior.new_ones((batch, labels, 1))), dim=-1)


def gfm_decode(count_probabilities, joint_probabilities):
    """Maximize the model's expected sample F1, taking empty-empty as 1."""
    batch, labels, _ = joint_probabilities.shape
    sizes = torch.arange(1, labels + 1, device=joint_probabilities.device)
    with torch.autocast(device_type=joint_probabilities.device.type, enabled=False):
        weights = torch.einsum("bck,ks->bsc", joint_probabilities[..., 1:].float(),
                               2.0 / (sizes[:, None] + sizes))
    order = weights.argsort(dim=-1, descending=True, stable=True)
    utility = weights.gather(-1, order).cumsum(-1).diagonal(dim1=-2, dim2=-1)
    winner = torch.cat((count_probabilities[:, :1], utility), dim=-1).argmax(-1)
    selected_order = order[torch.arange(batch, device=order.device), (winner - 1).clamp_min(0)]
    return torch.zeros_like(selected_order, dtype=torch.bool).scatter(-1, selected_order, sizes <= winner[:, None])


def fit_count_temperature(count_logits, targets, observed):
    count_logits = torch.as_tensor(count_logits, dtype=torch.float32)
    targets = torch.as_tensor(targets, device=count_logits.device)
    complete = torch.as_tensor(observed, device=count_logits.device).bool().all(dim=-1)
    if not complete.any():
        return 1.0
    logits, sizes = count_logits[complete], targets[complete].sum(-1).long()
    return min((0.5, 0.75, 1.0, 1.5, 2.0),
               key=lambda t: (float(F.cross_entropy(logits / t, sizes)), abs(t - 1), t))


def cardinality_output(base_logits, count_logits, energies, temperature=1.0):
    count_probabilities = (count_logits.float() / temperature).softmax(-1)
    joint = conditional_marginals(energies) * count_probabilities[:, None, :]
    return {"rank_scores": base_logits, "decision_scores": joint.sum(-1).clamp(1e-6, 1 - 1e-6).logit(),
            "predicted": gfm_decode(count_probabilities, joint),
            "count_probabilities": count_probabilities, "joint_probabilities": joint}


def independent_cardinality(base_logits):
    """B0's independent Bernoulli probabilities for the matched GFM control."""
    base_logits = base_logits.float()
    log_z = _prefix_partitions(base_logits)[-1]
    count_probabilities = log_z.softmax(-1)
    energies = base_logits[:, None].expand(-1, base_logits.shape[-1] + 1, -1)
    return count_probabilities, conditional_marginals(energies) * count_probabilities[:, None]


def count_top_k(decision_scores, count_probabilities):
    """Select the highest marginal scores at the most probable count."""
    sizes = count_probabilities.argmax(-1)
    order = decision_scores.argsort(dim=-1, descending=True, stable=True)
    slots = torch.arange(order.shape[-1], device=order.device)
    return torch.zeros_like(order, dtype=torch.bool).scatter(-1, order, slots < sizes[:, None])
