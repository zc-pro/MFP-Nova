"""Functional background, directed messages, and pairwise evidence for H1/H2/H4/H5."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _readout(input_dim, zero_init=False):
    network = nn.Sequential(nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, 1))
    if zero_init:
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
    return network


class PrototypeReadout(nn.Module):
    """H1. Training can supply a separate bank for each row's held-out fold."""

    def __init__(self, num_labels, dim):
        super().__init__()
        self.register_buffer("prototypes", torch.zeros(num_labels, 2, 2, dim))
        self.register_buffer("active", torch.zeros(num_labels, dtype=torch.bool))
        self.alpha = nn.Parameter(torch.zeros(num_labels))
        self.gate = _readout(dim + 2)

    def forward(self, evidence, logits, prototypes=None, active=None):
        prototypes = self.prototypes if prototypes is None else prototypes
        active = self.active if active is None else active
        unit = F.normalize(evidence.float(), dim=-1)
        cosine = (unit[:, :, None, None] * prototypes.float()).sum(-1)
        contrasts = cosine[..., 1] - cosine[..., 0]
        mean = contrasts.mean(-1)
        disagreement = (contrasts[..., 0] - contrasts[..., 1]).abs()
        gate = self.gate(torch.cat((evidence, mean[..., None], disagreement[..., None]), -1))
        return logits + active * gate.squeeze(-1).sigmoid() * self.alpha.tanh() * mean


class EvidenceMessages(nn.Module):
    """H2. A frozen message branch can be fitted before the receiver gates."""

    def __init__(self, num_labels, dim, edges, gate_mode="learned"):
        super().__init__()
        self.register_buffer("edges", torch.as_tensor(edges, dtype=torch.long).reshape(-1, 2))
        if gate_mode not in ("learned", "static", "open", "closed"):
            raise ValueError(f"Unknown H2 gate mode {gate_mode!r}")
        self.gate_mode = gate_mode
        self.message_projection = nn.Linear(dim, 64)
        self.message = _readout(130, zero_init=True)
        if gate_mode == "static":
            self.static_gate = nn.Parameter(torch.zeros(len(self.edges)))
        elif gate_mode == "learned":
            self.gate_projection = nn.Linear(dim, 64)
            self.gate = _readout(130)

    def _edge_inputs(self, evidence, logits, projection):
        sender, receiver = self.edges.unbind(1)
        projected = projection(evidence)
        return torch.cat((projected[:, sender], projected[:, receiver],
                          logits[:, sender, None], logits[:, receiver, None]), dim=-1)

    def messages(self, evidence, logits):
        inputs = self._edge_inputs(evidence, logits, self.message_projection)
        return self.message(inputs).squeeze(-1).tanh()

    def gate_logits(self, evidence, logits):
        if self.gate_mode == "static":
            return self.static_gate.expand(logits.shape[0], -1)
        inputs = self._edge_inputs(evidence, logits, self.gate_projection)
        return self.gate(inputs).squeeze(-1)

    def edge_values(self, evidence, logits):
        return self.messages(evidence, logits), self.gate_logits(evidence, logits)

    def forward(self, evidence, logits, edge_mask=None, gate_mode=None):
        gate_mode = gate_mode or self.gate_mode
        delta = self.messages(evidence, logits)
        if gate_mode == "open":
            gates = torch.ones_like(delta)
        elif gate_mode == "closed":
            gates = torch.zeros_like(delta)
        else:
            gates = self.gate_logits(evidence, logits).sigmoid()
        return self.aggregate(logits, delta, gates, edge_mask)

    def aggregate(self, logits, delta, gates, edge_mask=None):
        if edge_mask is not None:
            gates = gates * edge_mask
        recipients = self.edges[:, 1].expand(logits.size(0), -1)
        # Accumulate in FP32 even under mixed-precision training.
        base, gates, delta = logits.float(), gates.float(), delta.float()
        numerator = torch.zeros_like(base).scatter_add(1, recipients, gates * delta)
        denominator = torch.ones_like(base).scatter_add(1, recipients, gates)
        return base + numerator / denominator


def message_probe_values(delta, logits, targets, observed, edges):
    """The loss difference from enabling one edge in isolation with its gate at 1."""
    sender, receiver = edges.unbind(1)
    observed = observed.bool()
    target = targets[:, receiver].float()
    closed = logits[:, receiver].float()
    opened = closed + delta.float() / 2
    opened_loss = F.binary_cross_entropy_with_logits(opened, target, reduction="none")
    gain = F.binary_cross_entropy_with_logits(closed, target, reduction="none") - opened_loss
    return opened_loss, gain, observed[:, sender] & observed[:, receiver]


def align_edge_probes(edges, fold_probes, num_rows):
    """Align per-fold probes to the full Train graph and discard unsupported edges.

    Each fold supplies (held_out_indices, fold_edges, gain, valid), as CPU arrays.
    The returned target and valid matrices have columns in the returned edge order.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    targets = np.zeros((num_rows, len(edges)), dtype=bool)
    observed = np.zeros_like(targets)
    for indices, fold_edges, gain, valid in fold_probes:
        positions = {tuple(edge): j for j, edge in enumerate(fold_edges)}
        for j, edge in enumerate(edges):
            position = positions.get(tuple(edge))
            if position is not None:
                targets[indices, j] = valid[:, position] & (gain[:, position] > 0.01)
                observed[indices, j] = valid[:, position]
    keep = observed.any(0)
    return edges[keep], targets[:, keep], observed[:, keep]


def shuffled_gate_targets(targets, observed, seed):
    """Permute reliable probe targets independently for each edge."""
    rng = np.random.default_rng(seed)
    shuffled = targets.copy()
    for edge, mask in enumerate(observed.T):
        shuffled[mask, edge] = rng.permutation(targets[mask, edge])
    return shuffled


class DecisionResidual(nn.Module):
    """H4. A sequence-dependent correction on top of frozen ranking scores."""

    def __init__(self, dim):
        super().__init__()
        self.network = _readout(2 * dim + 1, zero_init=True)

    def forward(self, evidence, summary, ranking):
        inputs = torch.cat((evidence, summary[:, None, :].expand_as(evidence),
                            ranking[..., None]), dim=-1)
        residual = 2 * self.network(inputs).squeeze(-1).tanh()
        return ranking + residual, residual


def paired_rank_loss(positive, negative, reduction="mean"):
    values = F.softplus(negative.float() - positive.float())
    total = values.sum()
    return total / max(1, values.numel()) if reduction == "mean" else total


class PairJointHead(nn.Module):
    """H5. Four-state logits for disjoint pairs, with independent logits elsewhere."""

    def __init__(self, dim, pairs, distinction=True, interaction=True):
        super().__init__()
        pairs = torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)
        if pairs.numel() != pairs.unique().numel():
            raise ValueError("H5 needs disjoint label pairs")
        self.register_buffer("pairs", pairs)
        self.distinction = _readout(dim, zero_init=True) if distinction else None
        self.interaction = _readout(3 * dim, zero_init=True) if interaction else None

    def forward(self, evidence, logits):
        a, b = self.pairs.unbind(1)
        evidence_a, evidence_b = evidence[:, a], evidence[:, b]
        z_a, z_b = logits[:, a].float(), logits[:, b].float()
        d = (2 * self.distinction(evidence_a - evidence_b).squeeze(-1).float().tanh()
             if self.distinction is not None else torch.zeros_like(z_a))
        j = (2 * self.interaction(torch.cat((evidence_a, evidence_b, evidence_a * evidence_b), -1))
             .squeeze(-1).float().tanh() if self.interaction is not None else torch.zeros_like(z_a))
        # State order is 00, 10, 01, 11.
        state = torch.stack((torch.zeros_like(z_a), z_a + d, z_b - d, z_a + z_b + j), dim=-1)
        zero, only_a, only_b, both = state.unbind(-1)
        marginal = logits.float().clone()
        marginal[:, a] = torch.logaddexp(only_a, both) - torch.logaddexp(zero, only_b)
        marginal[:, b] = torch.logaddexp(only_b, both) - torch.logaddexp(zero, only_a)
        return marginal, state


class ReweightedPairReadout(nn.Module):
    """H5 control with an independent linear correction on the same paired labels."""

    def __init__(self, dim, pairs):
        super().__init__()
        self.register_buffer("pairs", torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2))
        self.weight = nn.Parameter(torch.zeros(self.pairs.shape[0], 2, dim))
        self.bias = nn.Parameter(torch.zeros_like(self.pairs, dtype=torch.float32))

    def forward(self, evidence, logits):
        labels = self.pairs.flatten()
        correction = (evidence[:, self.pairs] * self.weight).sum(-1) + self.bias
        return logits.index_add(1, labels, correction.flatten(1).to(logits.dtype))


def joint_loss(state_logits, marginal_logits, targets, observed, pairs, reduction="mean"):
    """Normalize the four-state NLL and partial marginal BCE by reliable bits."""
    observed = observed.bool()
    a, b = pairs.unbind(1)
    full = observed[:, a] & observed[:, b]
    state = (targets[:, a] + 2 * targets[:, b]).long()
    joint = -state_logits.float().log_softmax(-1).gather(-1, state[..., None]).squeeze(-1)
    single = observed.clone()
    single[:, a] &= ~full
    single[:, b] &= ~full
    bce = F.binary_cross_entropy_with_logits(marginal_logits.float(), targets.float(), reduction="none")
    total = (joint * full).sum() + (bce * single).sum()
    return total / observed.sum().clamp_min(1) if reduction == "mean" else total


def four_state_support(y, observed, groups):
    """Distinct Train groups per ordered pair and state, in 00/10/01/11 order."""
    y = np.asarray(y, dtype=np.uint8)
    observed = np.asarray(observed, dtype=bool)
    groups = np.asarray(groups)
    num_labels = y.shape[1]
    support = np.zeros((num_labels, num_labels, 4), dtype=np.int64)
    # Each state needs a reliably observed negative for at least one endpoint.
    has_negative = (observed & (y == 0)).any(0)
    labels = np.flatnonzero(has_negative)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            both = observed[:, a] & observed[:, b]
            state = y[both, a] + 2 * y[both, b]
            known_groups = groups[both]
            support[a, b] = [len(np.unique(known_groups[state == s])) for s in range(4)]
            support[b, a] = support[a, b, [0, 2, 1, 3]]
    return support


def _backgrounds(support, label_ids, receiver, limit, min_groups):
    coverage = support[:, receiver].min(-1)
    candidates = [int(label) for label in np.flatnonzero(coverage >= min_groups) if label != receiver]
    return sorted(candidates, key=lambda label: (-coverage[label], label_ids[label]))[:limit]


def select_edges(support, label_ids, max_incoming=4, min_groups=5):
    """Select directed candidate edges using only the current training complement."""
    edges = [(sender, receiver) for receiver in range(len(label_ids))
             for sender in _backgrounds(support, label_ids, receiver, max_incoming, min_groups)]
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def _group_evidence(evidence, target, selected, groups, row_indices):
    _, inverse, counts = np.unique(groups[selected], return_inverse=True, return_counts=True)
    rows = np.flatnonzero(selected)
    if row_indices is not None:
        rows = row_indices[rows]
    device = evidence.device
    unit = F.normalize(evidence[torch.as_tensor(rows, device=device), target].detach().float(), dim=-1)
    group_index = torch.as_tensor(inverse, device=device)
    grouped = unit.new_zeros(len(counts), unit.size(-1)).index_add_(0, group_index, unit)
    return grouped / torch.as_tensor(counts, dtype=torch.float32, device=device)[:, None]


def _global_prototypes(evidence, target, known, y, groups, row_indices, seed):
    from sklearn.cluster import KMeans

    centroids = []
    for positive in (0, 1):
        group_vectors = _group_evidence(evidence, target, known & (y[:, target] == positive),
                                        groups, row_indices)
        estimator = KMeans(n_clusters=min(2, len(group_vectors)), n_init=10, random_state=seed)
        estimator.fit(group_vectors.cpu().numpy())
        centroids.append(torch.as_tensor(estimator.cluster_centers_, device=evidence.device).expand(2, -1))
    # Give the control the same two positive-negative comparisons without using companion labels.
    cosine = F.normalize(centroids[0], dim=-1) @ F.normalize(centroids[1], dim=-1).T
    if cosine.diag().sum() < cosine.fliplr().diag().sum():
        centroids[1] = centroids[1].flip(0)
    return torch.stack(centroids, dim=1)


def build_prototypes(evidence, y, observed, groups, label_ids, min_groups=5, mode="background", seed=0,
                     row_indices=None, support=None):
    """Return (FP32 bank, bool active labels, int64 companion indices).

    Bank and mask share the input evidence's device. The companion index is -1 for
    disabled labels. Repeated observations within one family receive group-equal mass.
    """
    evidence = torch.as_tensor(evidence)
    y = np.asarray(y)
    observed = np.asarray(observed, dtype=bool)
    groups = np.asarray(groups)
    num_labels, dim = evidence.shape[1:]
    prototypes = torch.zeros(num_labels, 2, 2, dim, dtype=torch.float32, device=evidence.device)
    active = torch.zeros(num_labels, dtype=torch.bool, device=evidence.device)
    companions = np.full(num_labels, -1, dtype=np.int64)
    if support is None:
        support = four_state_support(y, observed, groups)
    rng = np.random.default_rng(seed)
    if mode not in ("background", "global", "shuffled"):
        raise ValueError(f"Unknown prototype background mode {mode!r}")
    for target in range(num_labels):
        candidates = _backgrounds(support, label_ids, target, 1, min_groups)
        if not candidates:
            continue
        companion = candidates[0]
        known = observed[:, target] & observed[:, companion]
        background = y[:, companion].copy()
        if mode == "global":
            prototypes[target] = _global_prototypes(evidence, target, known, y, groups, row_indices, seed)
        else:
            if mode == "shuffled":
                # Shuffle within target state while preserving its context frequency.
                for positive in (0, 1):
                    selected = known & (y[:, target] == positive)
                    background[selected] = rng.permutation(background[selected])
            for value in (0, 1):
                for positive in (0, 1):
                    selected = known & (y[:, target] == positive) & (background == value)
                    if selected.any():
                        group_vectors = _group_evidence(evidence, target, selected, groups, row_indices)
                        prototypes[target, value, positive] = group_vectors.mean(0)
        norms = prototypes[target].norm(dim=-1)
        if bool((norms >= 1e-6).all()):
            prototypes[target] /= norms[..., None]
            active[target] = True
            companions[target] = companion
    return prototypes, active, companions


def select_joint_pairs(y, observed, groups, label_ids, oof_logits, min_groups=5, return_details=False):
    """Rank four-state-supported pairs by equal-group OOF confusion on singletons."""
    y = np.asarray(y)
    observed = np.asarray(observed, dtype=bool)
    groups = np.asarray(groups)
    oof_logits = np.asarray(oof_logits)
    support = four_state_support(y, observed & np.isfinite(oof_logits), groups)
    candidates = []
    for a in sorted(range(len(label_ids)), key=label_ids.__getitem__):
        for b in sorted(np.flatnonzero(support[a].min(-1) >= min_groups), key=label_ids.__getitem__):
            if label_ids[a] >= label_ids[b]:
                continue
            known = (observed[:, a] & observed[:, b] &
                     np.isfinite(oof_logits[:, a]) & np.isfinite(oof_logits[:, b]))
            errors = []
            for positive, negative in ((a, b), (b, a)):
                selected = known & (y[:, positive] == 1) & (y[:, negative] == 0)
                _, inverse, counts = np.unique(groups[selected], return_inverse=True, return_counts=True)
                loss = np.logaddexp(0, oof_logits[selected, negative])
                errors.append((loss / counts[inverse]).sum() / len(counts))
            candidates.append((float(np.mean(errors)), a, int(b)))
    candidates.sort(key=lambda item: (-item[0], label_ids[item[1]], label_ids[item[2]]))
    pairs, used, details = [], set(), []
    for confusion, a, b in candidates:
        selected = not ({a, b} & used)
        if selected:
            pairs.append((a, b))
            used.update((a, b))
        details.append({"labels": [label_ids[a], label_ids[b]], "confusion": confusion,
                        "group_support": support[a, b].tolist(), "selected": selected})
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return (pairs, details) if return_details else pairs


def ranking_backgrounds(y, observed, groups, label_ids, min_groups=5):
    """Fit the H4 companion list on Train, then reuse it for independent diagnostics."""
    support = four_state_support(y, observed, groups)
    return [_backgrounds(support, label_ids, target, 3, min_groups)
            for target in range(len(label_ids))]


def matched_pairs(y, observed, groups, label_ids, lengths, sources, ids, *,
                  backgrounds=None, mode="matched", scores=None, seed=0, min_groups=5):
    """Return (positive row, confirmed negative row, label) in the input row order.

    The controls use the same matchable positives and relax the background
    match. All modes retain source, length, and family constraints.
    """
    if mode not in ("matched", "random", "hard"):
        raise ValueError(f"Unknown H4 pair selection mode {mode!r}")
    if mode == "hard" and scores is None:
        raise ValueError("Hard H4 pairing requires Train B0 scores")
    y = np.asarray(y, dtype=np.int8)
    observed = np.asarray(observed, dtype=bool)
    groups, lengths, ids = np.asarray(groups), np.asarray(lengths), np.asarray(ids, dtype=str)
    sources = np.asarray(["" if source is None else str(source) for source in sources])
    if backgrounds is None:
        backgrounds = ranking_backgrounds(y, observed, groups, label_ids, min_groups)
    if scores is not None:
        scores = np.asarray(scores)
    pairs = []
    rng = np.random.default_rng(seed)
    for label, background in enumerate(backgrounds):
        if not background:
            continue
        signature = np.where(observed[:, background], y[:, background], -1)
        positive = np.flatnonzero(observed[:, label] & (y[:, label] == 1))
        negative = np.flatnonzero(observed[:, label] & (y[:, label] == 0))
        # A source has at most 27 signatures for the three selected companions.
        buckets = {}
        for index in negative[np.argsort(ids[negative], kind="stable")]:
            by_signature = buckets.setdefault(sources[index], {})
            by_length = by_signature.setdefault(tuple(signature[index]), {})
            by_length.setdefault(lengths[index], []).append(index)
        for index in positive:
            by_signature = buckets.get(sources[index], {})
            lo = (4 * lengths[index] + 4) // 5
            hi = 5 * lengths[index] // 4
            family = groups[index]
            best = None
            for negative_signature, by_length in by_signature.items():
                shared = [(a, b) for a, b in zip(signature[index], negative_signature) if a >= 0 and b >= 0]
                if not shared or any(a != b for a, b in shared):
                    continue
                if best is not None and len(shared) < -best[0]:
                    continue
                for length, rows in by_length.items():
                    if not lo <= length <= hi:
                        continue
                    difference = abs(lengths[index] - length)
                    if best is not None and (-len(shared), difference) > best[:2]:
                        continue
                    selected = next((row for row in rows if groups[row] != family), None)
                    if selected is not None:
                        key = (-len(shared), difference, ids[selected], selected)
                        if best is None or key < best:
                            best = key
            if best is None:
                continue
            selected = best[-1]
            if mode != "matched":
                available = [row for by_length in by_signature.values()
                             for length, rows in by_length.items() if lo <= length <= hi
                             for row in rows if groups[row] != family]
                if mode == "hard":
                    selected = min(available, key=lambda row: (-scores[row, label], ids[row]))
                else:
                    selected = rng.choice(available)
            pairs.append((index, selected, label))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 3)


class RankingPairSampler:
    """Sample participating labels uniformly, then one pair per chosen label."""

    def __init__(self, pairs):
        self.by_label = [pairs[pairs[:, 2] == label] for label in np.unique(pairs[:, 2])]

    def sample(self, rng, size=32):
        if not self.by_label:
            return np.empty((0, 3), dtype=np.int64)
        groups = rng.integers(len(self.by_label), size=size)
        return np.stack([self.by_label[group][rng.integers(len(self.by_label[group]))] for group in groups])
