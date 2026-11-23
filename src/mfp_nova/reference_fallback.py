"""Fixed Train-cluster fallback for empty DuoEvidence predictions."""

import numpy as np

from .models.theraset import residue_encoding


def aa34_summary(sequences):
    alphabet = {aa: i + 1 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    table = residue_encoding().numpy()
    features = []
    for sequence in sequences:
        tokens = np.array([alphabet.get(aa, 0) for aa in sequence])
        props = table[tokens]
        pairs = np.bincount(21 * tokens[:-1] + tokens[1:], minlength=441)[1:].astype(np.float32)
        pairs = pairs / max(1, len(tokens) - 1)
        features.append(np.concatenate((props.mean(0), np.sqrt(pairs))))
    return np.stack(features)


def support_scores(fit, query, targets, clusters, batch_size=128):
    """For each label, average the best matches from two distinct Train clusters."""
    center = fit.mean(0)
    fit, query = fit - center, query - center
    fit /= np.linalg.norm(fit, axis=1, keepdims=True).clip(min=1e-12)
    query /= np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-12)
    label_groups = []
    for positives in targets.T:
        idx = np.flatnonzero(positives)
        idx = idx[np.argsort(clusters[idx], kind="stable")]
        starts = np.flatnonzero(np.r_[True, clusters[idx][1:] != clusters[idx][:-1]]) if len(idx) else idx
        label_groups.append((idx, starts))
    support = np.full((len(query), targets.shape[1]), np.nan, dtype=np.float32)
    for start in range(0, len(query), batch_size):
        end = min(start + batch_size, len(query))
        similarities = query[start:end] @ fit.T
        for label, (idx, starts) in enumerate(label_groups):
            if len(starts) < 2:
                continue
            by_cluster = np.maximum.reduceat(similarities[:, idx], starts, axis=1)
            top_two = np.partition(by_cluster, -2, axis=1)[:, -2:]
            support[start:end, label] = top_two.mean(1)
    return support


def paired_cluster_interval(delta, clusters):
    _, inverse, sizes = np.unique(clusters, return_inverse=True, return_counts=True)
    totals = np.bincount(inverse, weights=delta)
    rng = np.random.default_rng(42)
    changes = []
    for _ in range(2000):
        sample = rng.integers(len(sizes), size=len(sizes))
        changes.append(100 * totals[sample].sum() / sizes[sample].sum())
    return np.quantile(changes, [.025, .975]).tolist()


def select_fallback(raw, baseline, theraset_logits, support):
    """Replace the one-label fallback only on empty raw rows with full pool support."""
    candidates = np.argsort(-theraset_logits, axis=1, kind="stable")[:, :10]
    scores = np.take_along_axis(support, candidates, axis=1)
    eligible = ~raw.any(1) & np.isfinite(scores).all(1)
    rows = np.flatnonzero(eligible)
    label = candidates[rows, scores[rows].argmax(1)]
    updated = baseline.copy()
    updated[rows] = False
    updated[rows, label] = True
    return eligible, updated


def apply_train_reference(data, indices, raw, baseline, theraset_logits):
    train = data.indices("train")
    if np.intersect1d(data.clusters[train], data.clusters[indices]).size:
        raise ValueError("Train reference and evaluation clusters overlap")
    empty = ~raw.any(1)
    eligible = np.zeros(len(raw), dtype=bool)
    updated = baseline.copy()
    if empty.any():
        support = support_scores(aa34_summary(data.sequences[train]), aa34_summary(data.sequences[indices[empty]]),
                                 data.y[train].astype(bool), data.clusters[train])
        eligible[empty], updated[empty] = select_fallback(raw[empty], baseline[empty], theraset_logits[empty], support)
    return eligible, updated


def fallback_comparison(y, baseline, updated, eligible, clusters):
    def jaccard(prediction):
        return (prediction & y).sum(1) / (prediction | y).sum(1).clip(min=1)

    before = (baseline & y).any(1)
    after = (updated & y).any(1)
    delta = jaccard(updated) - jaccard(baseline)
    return {
        "eligible": int(eligible.sum()),
        "rescued": int((~before & after).sum()),
        "new_zero": int((before & ~after).sum()),
        "delta_pp": float(100 * delta.mean()),
        "paired_cluster_ci95_pp": paired_cluster_interval(delta, clusters),
    }


