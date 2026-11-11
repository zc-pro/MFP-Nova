"""Train-only teacher targets (H7) and observation propensities (H8)."""

import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..hypothesis_data import family_folds


def _has_support(target, groups, minimum):
    return (len(np.unique(groups[target])) >= minimum
            and len(np.unique(groups[~target])) >= minimum)


def _group_mean(values, groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return weights @ values / len(counts)


def _temperature(logits, target, groups, minimum):
    if not _has_support(target, groups, minimum):
        return 1.0
    sign = 1 - 2 * target.astype(np.float64)
    candidates = (1.0, 0.75, 0.5, 1.5, 2.0)
    return min(candidates, key=lambda t: np.logaddexp(0, sign * logits / t).mean())


def _bernoulli_loss(probability, target):
    correct = np.where(target, probability, 1 - probability)
    return -np.log(correct.clip(1e-12, 1))


def _teacher_loss(probabilities, target, groups):
    return _group_mean(_bernoulli_loss(probabilities, target[:, None]), groups)


def _teacher_weights(losses):
    weights = np.exp(losses.min() - losses)
    return weights / weights.sum()


def _fitted_teacher(probabilities, target, groups, overall, layer, minimum):
    if not _has_support(target[layer], groups[layer], minimum):
        return None
    loss = _teacher_loss(probabilities[overall], target[overall], groups[overall])
    if not np.array_equal(overall, layer):
        n = len(np.unique(groups[layer]))
        layer_loss = _teacher_loss(probabilities[layer], target[layer], groups[layer])
        loss = (n * layer_loss + 20 * loss) / (n + 20)
    return _teacher_weights(loss), int(loss.argmin())


def _improvement(probabilities, target, groups, fit, layer, halves, minimum, margin):
    """Compare fitted mixtures with both controls on the opposite half."""
    gains = []
    for side in range(2):
        source = fit & (halves == side)
        estimate = _fitted_teacher(probabilities, target, groups, source,
                                   source & layer, minimum)
        valid = fit & layer & (halves != side)
        if estimate is None or not valid.any():
            return False
        weights, best = estimate
        scores = probabilities[valid]
        alternatives = np.column_stack((scores.mean(axis=1), scores[:, best], scores @ weights))
        losses = _teacher_loss(alternatives, target[valid], groups[valid])
        gains.append(losses[:2] - losses[2])
    gains = np.stack(gains)
    return bool((gains > 0).all() and (gains.mean(axis=0) >= margin).all())


def _calibrated_teachers(logits, available, targets, observed, calibration, groups, minimum):
    temperatures = np.ones(logits.shape[1:])
    for teacher, label in np.ndindex(temperatures.shape):
        use = calibration & observed[:, label] & available[:, teacher, label]
        temperatures[teacher, label] = _temperature(logits[use, teacher, label], targets[use, label],
                                                     groups[use], minimum)
    return np.exp(-np.logaddexp(0, -np.where(available, logits, 0) / temperatures))


def distillation_targets(teacher_logits, targets, observed, groups, folds, lengths, *,
                         mode="selective", seed=42, min_groups=5, margin=0.005, roles=None):
    """Calibrate and select teachers within outer-fold held-out rows.

    Return soft probabilities, the accepted mask, and the denominator mask.
    Missing teacher logits can be represented by NaN. The caller supplies two
    teacher predictions from models fitted outside each row's outer fold.
    """
    if mode not in ("selective", "equal", "best", "weighted", "random", "off"):
        raise ValueError(f"Unknown distillation mode: {mode}")
    logits = np.asarray(teacher_logits, dtype=np.float64)
    y, groups, folds = np.asarray(targets, dtype=bool), np.asarray(groups), np.asarray(folds)
    lengths = np.asarray(lengths)
    available = np.isfinite(logits)
    observed = np.asarray(observed, dtype=bool)
    eligible = observed & available.all(axis=1)
    accepted = np.zeros_like(eligible)
    soft = np.zeros(y.shape, dtype=np.float32)
    if mode == "off":
        return soft, accepted, eligible

    rng = np.random.default_rng(seed)
    background = np.searchsorted((20, 40), lengths, side="left")
    for fold in np.unique(folds):
        indices = np.flatnonzero(folds == fold)
        fold_groups, fold_targets = groups[indices], y[indices]
        fold_eligible = eligible[indices]
        fold_roles = family_folds(fold_groups, 3) if roles is None else roles[indices]
        for rotation in range(3):
            calibration = fold_roles == rotation
            fitting = fold_roles == (rotation + 1) % 3
            application = fold_roles == (rotation + 2) % 3
            candidates = application[:, None] & fold_eligible
            if not candidates.any():
                continue
            probabilities = _calibrated_teachers(logits[indices], available[indices], fold_targets,
                                                observed[indices], calibration, fold_groups, min_groups)
            if mode in ("equal", "best"):
                if mode == "equal":
                    mixture = probabilities.mean(axis=1)
                else:
                    fit_rows, fit_labels = np.where(fitting[:, None] & fold_eligible)
                    if not len(fit_rows):
                        continue
                    losses = _teacher_loss(probabilities[fit_rows, :, fit_labels],
                                           fold_targets[fit_rows, fit_labels], fold_groups[fit_rows])
                    mixture = probabilities[:, losses.argmin()]
                rows, labels = np.where(candidates)
                soft[indices[rows], labels] = mixture[rows, labels]
                accepted[indices[rows], labels] = True
                continue

            halves = np.zeros(len(indices), dtype=np.int64)
            halves[fitting] = family_folds(fold_groups[fitting], 2)
            for label in range(y.shape[1]):
                candidate = candidates[:, label]
                if not candidate.any():
                    continue
                target = fold_targets[:, label]
                scores = probabilities[:, :, label]
                fit = fitting & fold_eligible[:, label]
                final = scores.mean(axis=1)
                unselected = final.copy()
                selected = np.zeros(len(indices), dtype=bool)
                layers = (np.ones_like(fit),)
                if mode != "weighted":
                    layers += tuple(background[indices] == band for band in range(3))
                for layer in layers:
                    apply = candidate & layer
                    if not apply.any():
                        continue
                    estimate = _fitted_teacher(scores, target, fold_groups, fit, fit & layer, min_groups)
                    if estimate is None:
                        continue
                    weights, _ = estimate
                    unselected[apply] = scores[apply] @ weights
                    if mode == "weighted" or _improvement(scores, target, fold_groups, fit, layer,
                                                          halves, min_groups, margin):
                        selected |= apply
                        final[apply] = unselected[apply]

                if mode == "random":
                    final[~selected] = unselected[~selected]
                    active = rng.choice(np.flatnonzero(candidate), size=selected.sum(), replace=False)
                else:
                    active = np.flatnonzero(candidate if mode == "weighted" else selected)
                rows = indices[active]
                soft[rows, label] = final[active]
                accepted[rows, label] = True
    return soft, accepted, eligible


class ObservationHead(nn.Module):
    """Estimate whether a reliable measurement is available for each label."""

    def __init__(self, summary_dim, labels, sources):
        super().__init__()
        self.source = nn.Embedding(sources + 1, 32, padding_idx=0)
        self.head = nn.Sequential(nn.Linear(summary_dim + 32, 64), nn.GELU(), nn.Linear(64, labels))

    def forward(self, summary, source):
        return self.head(torch.cat((summary, self.source(source)), dim=-1))


def observation_domain(sources, observed, audited, groups, *, min_groups=5):
    """Return the fold's source vocabulary and source/label support mask."""
    sources, groups = np.asarray(sources), np.asarray(groups)
    observed, audited = np.asarray(observed, dtype=bool), np.asarray(audited, dtype=bool)
    vocabulary = {source: i + 1 for i, source in enumerate(s for s in np.unique(sources) if s)}
    domain = np.zeros((len(vocabulary) + 1, observed.shape[1]), dtype=bool)
    audited_rows = defaultdict(list)
    for row in np.flatnonzero(audited.any(axis=1)):
        if sources[row]:
            audited_rows[sources[row]].append(row)
    for source, rows in audited_rows.items():
        source_observed, source_audited, source_groups = observed[rows], audited[rows], groups[rows]
        for label in range(observed.shape[1]):
            use = source_audited[:, label]
            domain[vocabulary[source], label] = _has_support(source_observed[use, label], source_groups[use], min_groups)
    return vocabulary, domain


def fit_observation_fold(train_summary, held_summary, train_source, held_source, train_observed,
                         train_audited, train_groups, *, device="cpu", seed=42, epochs=20,
                         batch_size=32, learning_rate=1e-3, min_groups=5, max_steps=None,
                         source_domain=None):
    """Fit H8 on a fold complement and predict its excluded groups in FP32."""
    if source_domain is None:
        source_domain = observation_domain(train_source, train_observed, train_audited,
                                            train_groups, min_groups=min_groups)
    vocabulary, domain = source_domain
    held_index = np.array([vocabulary.get(source, 0) for source in held_source], dtype=np.int64)
    applicable = domain[held_index]
    if not domain.any():
        return np.full(applicable.shape, np.nan, dtype=np.float32), applicable

    rng = np.random.default_rng(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = ObservationHead(train_summary.shape[1], domain.shape[1], len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
    audited = torch.as_tensor(train_audited, dtype=torch.bool)
    observed = torch.as_tensor(train_observed, dtype=torch.float32)
    summary = torch.as_tensor(train_summary, dtype=torch.float32).detach()
    source = torch.tensor([vocabulary.get(s, 0) for s in train_source], dtype=torch.long)
    rows = torch.where(audited.any(dim=1))[0].numpy()
    steps = math.ceil(len(rows) / batch_size) * epochs
    if max_steps:
        steps = min(steps, max_steps)
    warmup = math.ceil(0.05 * steps)
    step = 0
    for _ in range(epochs):
        permutation = rng.permutation(rows)
        for start in range(0, len(rows), batch_size):
            selected = permutation[start:start + batch_size]
            if step < warmup:
                factor = (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, steps - warmup)
                factor = (1 + math.cos(math.pi * progress)) / 2
            optimizer.param_groups[0]["lr"] = learning_rate * factor
            optimizer.zero_grad(set_to_none=True)
            logits = head(summary[selected].to(device), source[selected].to(device))
            loss = F.binary_cross_entropy_with_logits(logits, observed[selected].to(device), reduction="none")
            loss.masked_select(audited[selected].to(device)).mean().backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step == steps:
                break
        if step == steps:
            break

    head.eval()
    with torch.inference_mode():
        held_summary = torch.as_tensor(held_summary, dtype=torch.float32)
        held_index = torch.from_numpy(held_index)
        predictions = [head(batch.to(device), src.to(device)).sigmoid().cpu()
                       for batch, src in zip(held_summary.split(batch_size), held_index.split(batch_size))]
    return torch.cat(predictions).numpy(), applicable


def observation_weights(propensity, applicable, observed, floor=0.1):
    """Use clipped inverse propensities only at reliable supported positions."""
    use = np.asarray(applicable, dtype=bool) & np.asarray(observed, dtype=bool)
    weights = np.ones(use.shape, dtype=np.float32)
    weights[use] = 1 / np.maximum(np.asarray(propensity)[use], floor)
    return weights
