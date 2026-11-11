"""Training-only cross-fitting phases for H7 and H8."""

import time

import numpy as np

from .hypothesis_data import family_folds
from .models.hypothesis_supervision import (distillation_targets, fit_observation_fold,
                                            observation_domain, observation_weights)
from .trainer import seed_everything


def _save_training_asset(session, filename, **arrays):
    if session.run is None:
        return
    data, rows = session.data, session.train_indices
    np.savez_compressed(session.run / filename, row_id=data.rows[rows], ID=data.ids[rows].astype(str),
                        label_names=data.labels, group_id=data.clusters[rows].astype(str),
                        fold_id=session.folds, **arrays)


def _teacher_statistics(teachers, target, observed, accepted, eligible):
    wrong = (teachers >= 0) != target[:, None, :]
    jointly_wrong = wrong.all(axis=1) & eligible
    complete = observed.all(axis=1)
    counts = target.sum(axis=1)
    count_ranges = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5),
                    ("6-9", 6, 9), ("10+", 10, target.shape[1]))

    def summary(rows):
        available = int(eligible[rows].sum())
        selected = int(accepted[rows].sum())
        return {"rows": int(rows.sum()), "eligible": available, "accepted": selected,
                "acceptance_rate": selected / available if available else None,
                "joint_error_rate_at_zero_logit": int(jointly_wrong[rows].sum()) / available if available else None}

    strata = {name: summary(complete & (counts >= low) & (counts <= high))
              for name, low, high in count_ranges}
    strata["unknown_count"] = summary(~complete)
    return {"eligible_bits": int(eligible.sum()), "accepted_bits": int(accepted.sum()),
            "eligible_by_label": eligible.sum(axis=0).tolist(), "accepted_by_label": accepted.sum(axis=0).tolist(),
            "teacher_joint_errors_by_label": jointly_wrong.sum(axis=0).tolist(), "by_true_count": strata}


def fit_h7(session, model):
    rows, data = session.train_indices, session.data
    mode = session.settings.get("distillation_mode", "selective")
    y, observed = data.targets[rows].astype(bool), data.observed[rows]
    groups = data.clusters[rows]
    binary_support = any(len(np.unique(groups[y[:, c] & observed[:, c]])) >= 10
                         and len(np.unique(groups[~y[:, c] & observed[:, c]])) >= 10
                         for c in range(y.shape[1]))
    soft = None
    if mode == "off" or len(np.unique(groups)) < 2 or (mode in ("selective", "random") and not binary_support):
        model.assets.update(mechanism_active=False, distillation_mode=mode, eligible_bits=0, accepted_bits=0,
                            reason="distillation_disabled" if mode == "off" else "insufficient_reliable_group_support")
    else:
        teachers = np.full((len(rows), 2, len(data.labels)), np.nan, dtype=np.float32)
        roles = np.empty(len(rows), dtype=np.int64)
        for fold, fitted, held in session.training_folds():
            roles[held] = family_folds(groups[held], 3)
            for teacher, balanced in enumerate((False, True)):
                backbone = session.fold_backbone(fold, rows[fitted], balanced=balanced)
                teachers[held, teacher] = session.encode(backbone, rows[held], fields=("logits",))["logits"].numpy()
        probability, accepted, eligible = distillation_targets(teachers, y, observed, groups, session.folds,
                                                               data.lengths[rows], mode=mode, seed=session.seed, roles=roles)
        _save_training_asset(session, "h7_train_targets.npz", teacher_logits=teachers,
                             teacher_id=["b0", "label_balanced_b0"], teacher_seed=[session.seed, session.seed + 1],
                             role=roles, soft_probability=probability, accepted=accepted, eligible=eligible)
        model.assets.update(mechanism_active=bool(accepted.any()), distillation_mode=mode,
                            **_teacher_statistics(teachers, y, observed, accepted, eligible))
        if accepted.any():
            aligned_probability = np.zeros(data.y.shape, dtype=np.float32)
            aligned_accepted = np.zeros(data.y.shape, dtype=bool)
            aligned_eligible = np.zeros(data.y.shape, dtype=bool)
            aligned_probability[rows], aligned_accepted[rows], aligned_eligible[rows] = probability, accepted, eligible
            soft = aligned_probability, aligned_accepted, aligned_eligible
        else:
            model.assets["reason"] = "no_reliable_teacher_gain"

    seed_everything(session.seed)
    session.fit_backbone(model.backbone, rows, "H7 student", seed=session.seed, soft=soft)


def _weight_statistics(weights, reliable):
    values = weights[reliable].astype(np.float64)
    if not len(values):
        return {"effective_sample_size": 0.0, "weight_quantiles": None}
    return {"effective_sample_size": float(values.sum() ** 2 / (values @ values)),
            "weight_quantiles": np.quantile(values, (0.05, 0.5, 0.95)).tolist()}


def fit_h8(session, model):
    rows, data = session.train_indices, session.data
    settings = session.settings
    mode = settings.get("observation_mode", "weighted")
    floor = settings.get("observation_floor", 0.1)
    if mode not in ("weighted", "masked", "weak") or not 0 < floor <= 1:
        raise ValueError("H8 requires a supported observation mode and 0 < observation_floor <= 1")
    observed, audited = data.observed[rows], data.audited[rows]
    propensity = np.full(observed.shape, np.nan, dtype=np.float32)
    applicable = np.zeros_like(observed)
    evaluated_folds = []
    if mode == "weighted" and observation_domain(data.sources[rows], observed, audited, data.clusters[rows])[1].any():
        for fold, fitted, held in session.training_folds():
            fitted_rows, held_rows = rows[fitted], rows[held]
            fitted_sources, held_sources = data.sources[fitted_rows], data.sources[held_rows]
            vocabulary, domain = observation_domain(fitted_sources, observed[fitted], audited[fitted],
                                                    data.clusters[fitted_rows])
            if not domain[[vocabulary.get(source, 0) for source in held_sources]].any():
                continue
            backbone = session.fold_backbone(fold, fitted_rows)
            fitted_summary = session.encode(backbone, fitted_rows, fields=("summary",))["summary"]
            held_summary = session.encode(backbone, held_rows, fields=("summary",))["summary"]
            started = time.perf_counter()
            propensity[held], applicable[held] = fit_observation_fold(
                fitted_summary, held_summary, fitted_sources, held_sources, observed[fitted], audited[fitted],
                data.clusters[fitted_rows], device=session.device, seed=session.seed, batch_size=session.batch,
                epochs=settings["head_epochs"], learning_rate=settings["head_learning_rate"],
                max_steps=settings.get("max_steps"), source_domain=(vocabulary, domain))
            evaluated_folds.append(fold)
            session.phases.append({"phase": f"H8 observation fold {fold}", "seconds": time.perf_counter() - started})

    weights = observation_weights(propensity, applicable, observed, floor=floor)
    supported = applicable & observed
    audited_oof = applicable & audited
    model.assets.update(mechanism_active=bool(supported.any()), observation_mode=mode, observation_floor=floor,
                        supported_bits=int(supported.sum()), reliable_bits=int(observed.sum()),
                        observation_folds=evaluated_folds, audited_oof_bits=int(audited_oof.sum()),
                        oof_observation_brier=float(np.square(propensity[audited_oof] - observed[audited_oof]).mean())
                        if audited_oof.any() else None, **_weight_statistics(weights, observed))
    if mode == "weighted" and not supported.any():
        model.assets["reason"] = "no_audited_observation_domain"
    if evaluated_folds:
        model.assets["clipping_sensitivity"] = {
            str(clip): _weight_statistics(observation_weights(propensity, applicable, observed, floor=clip), observed)
            for clip in (0.05, 0.2)}
        _save_training_asset(session, "h8_train_propensity.npz", source_id=data.sources[rows].astype(str),
                             propensity=propensity, applicable=applicable, observed=observed,
                             audited=audited, weight=weights)

    aligned_weights = None
    if mode != "weak":
        aligned_weights = np.zeros(data.y.shape, dtype=np.float32)
        aligned_weights[rows] = weights * observed
    seed_everything(session.seed)
    session.fit_backbone(model.backbone, rows, "H8 function branch", seed=session.seed, observation=aligned_weights)
