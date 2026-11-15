"""Learn when to replace the TheraSet label set using its ranked candidates."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from compare_validation_predictions import read_validation, sample_jaccard
from mfp_nova.data import PeptideData
from mfp_nova.hh_v13 import load_oof
from mfp_nova.hypothesis_data import family_folds
from mfp_nova.metrics import fit_global_threshold
from mfp_nova.neighbor_evidence import NEIGHBORS, fold_neighbor_evidence


MAX_K = 6
RANK_RANGES = [(0, k) for k in range(1, MAX_K + 1)] + [(k, k + 1) for k in range(1, MAX_K)]
CANDIDATE_NAMES = ["keep"] + [f"top_{k}" for k in range(1, MAX_K + 1)] + [
    f"only_rank_{k}" for k in range(2, MAX_K + 1)]
GAIN_THRESHOLDS = (0., .01, .02, .04, .06, .10, .15, .20, np.inf)
HARM_BUDGETS = (0, 10, 25, 50, 84)


def proposal_features(probabilities, lengths):
    """Return the baseline, alternative sets, distinct edits, and target-free features."""
    ranked = np.argsort(-probabilities, axis=1, kind="stable")[:, :MAX_K + 1]
    top = np.take_along_axis(probabilities, ranked, axis=1)
    base = probabilities >= .5
    n, labels = base.shape
    candidates = np.zeros((n, len(CANDIDATE_NAMES), labels), dtype=bool)
    candidates[:, 0] = base
    for option, (start, end) in enumerate(RANK_RANGES, start=1):
        np.put_along_axis(candidates[:, option], ranked[:, start:end], True, axis=1)
    valid = (candidates[:, 1:] != base[:, None]).any(axis=2)
    base_size = base.sum(axis=1)
    gaps = top[:, :-1] - top[:, 1:]
    features = np.stack([
        np.column_stack((top, gaps, lengths, base_size, np.full(n, end - start), np.full(n, start + 1),
                         top[:, start], top[:, end - 1], top[:, end], gaps[:, end - 1],
                         ranked[:, 0], ranked[:, 1], ranked[:, start], ranked[:, end]))
        for start, end in RANK_RANGES
    ], axis=1).astype(np.float32)
    return candidates, valid, features


def candidate_jaccard(candidates, y):
    overlap = (candidates & y[:, None]).sum(axis=-1)
    union = (candidates | y[:, None]).sum(axis=-1)
    return overlap / np.maximum(union, 1)


def fit_regressor(features, targets, seed, categorical_count, weights=None):
    model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=15, min_samples_leaf=128, learning_rate=.08,
        categorical_features=list(range(features.shape[-1] - categorical_count, features.shape[-1])),
        early_stopping=False, random_state=seed,
    )
    return model.fit(features, targets, sample_weight=weights)


def fit_selector(features, valid, scores, seed):
    delta = scores[:, 1:] - scores[:, :1]
    weights = np.broadcast_to(1. / valid.sum(axis=1, keepdims=True), valid.shape)
    return fit_regressor(features[valid], delta[valid], seed, categorical_count=4, weights=weights[valid])


def predict_gains(model, features, valid):
    gains = np.full(valid.shape, -np.inf)
    gains[valid] = model.predict(features[valid])
    return gains


def add_neighbor_features(features, candidates, evidence):
    vote, positive_sim, nearest = evidence
    shape = candidates[:, 1:].shape[:2]
    before, choices = candidates[:, :1], candidates[:, 1:]

    def summarize(mask):
        mean_vote = (mask * vote[:, None]).sum(axis=-1) / mask.sum(axis=-1).clip(min=1)
        strongest_vote = np.where(mask, vote[:, None], 0.).max(axis=-1)
        best_similarity = np.where(mask & (vote[:, None] > 0.), positive_sim[:, None], -np.inf).max(axis=-1)
        best_similarity = np.where(np.isfinite(best_similarity), best_similarity, 0.)
        return tuple(np.broadcast_to(field, shape) for field in (mean_vote, strongest_vote, best_similarity))

    base = summarize(before)
    added = summarize(choices & ~before)
    removed = summarize(before & ~choices)
    extra = np.stack((*base, *added, *removed, added[0] - removed[0],
                      np.broadcast_to(nearest[:, None], shape)), axis=-1).astype(np.float32)
    # Keep the four categorical label fields at the end.
    return np.concatenate((features[..., :-4], extra, features[..., -4:]), axis=-1)


def singleton_pair_features(probabilities, candidates, lengths, evidence):
    """Estimate one keep value for A and compare it against each singleton B."""
    rows = np.flatnonzero(candidates[:, 0].sum(axis=1) == 1)
    a_label = candidates[rows, 0].argmax(axis=1)
    b_labels = candidates[rows, MAX_K + 1:].argmax(axis=2)
    pair_rows = rows[:, None]
    vote, positive_sim, nearest = evidence
    p_a, p_b = probabilities[rows, a_label], probabilities[pair_rows, b_labels]
    v_a, v_b = vote[rows, a_label], vote[pair_rows, b_labels]
    s_a, s_b = positive_sim[rows, a_label], positive_sim[pair_rows, b_labels]
    shared = np.column_stack((lengths[rows], nearest[rows], p_a, p_b[:, 0], p_a - p_b[:, 0],
                              v_a, v_b.max(axis=1), s_a, s_b.max(axis=1)))
    keep = np.column_stack((shared, a_label)).astype(np.float32)
    context = np.broadcast_to(shared[:, None], (*b_labels.shape, shared.shape[-1]))
    alternatives = np.stack((
        np.broadcast_to(np.arange(2, MAX_K + 1), b_labels.shape),
        p_b, v_b, s_b, p_b - p_a[:, None], v_b - v_a[:, None], s_b - s_a[:, None],
        np.broadcast_to(a_label[:, None], b_labels.shape), b_labels,
    ), axis=-1)
    replace = np.concatenate((context, alternatives), axis=-1).astype(np.float32)
    return rows, keep, replace


def fit_singleton_pairs(pair_features, scores, seed):
    rows, keep, replace = pair_features
    # A singleton scores 1 / |Y| when its label is recorded and 0 otherwise.
    return (
        fit_regressor(keep, scores[rows, 0], seed, categorical_count=1),
        fit_regressor(replace.reshape(-1, replace.shape[-1]),
                      scores[rows, MAX_K + 1:].ravel(), seed, categorical_count=2),
    )


def predict_singleton_gains(models, pair_features, neighbor_gains):
    rows, keep, replace = pair_features
    keep_model, replace_model = models
    keep_value = keep_model.predict(keep).clip(0., 1.)
    replace_value = replace_model.predict(replace.reshape(-1, replace.shape[-1])).clip(0., 1.)
    replace_value = replace_value.reshape(replace.shape[:2])
    gains = neighbor_gains.copy()
    gains[rows, MAX_K:] = replace_value - keep_value[:, None]
    return gains, keep_value, replace_value


def selection(gains, threshold):
    best = gains.argmax(axis=1)
    return np.where(gains[np.arange(len(gains)), best] > threshold, best + 1, 0)


def edit_summary(y, base, selected):
    before, after = sample_jaccard(y, base), sample_jaccard(y, selected)
    delta = after - before
    edited = (base != selected).any(axis=1)
    base_size = base.sum(axis=1)
    wrong = (before == 0) & (base_size > 0)
    wrong_single = wrong & (base_size == 1) & (y.sum(axis=1) == 1)
    return {
        "sample_jaccard": float(after.mean()), "delta_jaccard_pp": float(100. * delta.mean()),
        "micro_f1": float(f1_score(y, selected, average="micro", zero_division=0)),
        "exact_match": float((after == 1).mean()),
        "edited": int(edited.sum()), "improved": int((delta > 0).sum()),
        "harmed": int((delta < 0).sum()), "neutral_edits": int((edited & (delta == 0)).sum()),
        "wrong_singletons": int(wrong_single.sum()),
        "wrong_singletons_corrected": int((wrong_single & (after == 1)).sum()),
        "correct_before": int((before == 1).sum()),
        "correct_before_harmed": int(((before == 1) & (after < 1)).sum()),
        "wrong_nonempty_recovered": int((wrong & (after > 0)).sum()),
        "empty_filled_with_overlap": int(((base_size == 0) & (after > 0)).sum()),
        "jaccard_delta_empty_pp": float(100. * delta[base_size == 0].sum() / len(y)),
        "jaccard_delta_wrong_nonempty_pp": float(100. * delta[wrong].sum() / len(y)),
        "jaccard_delta_with_overlap_pp": float(100. * delta[before > 0].sum() / len(y)),
    }


def compare_with_fixed_empty_fill(y, candidates, scores, gains, clusters, folds=None):
    """Compare nonempty decisions while both arms fill empty sets with Top-1."""
    rows = np.arange(len(y))
    base = candidates[:, 0]
    empty = ~base.any(axis=1)
    best = gains.argmax(axis=1)
    max_gain = gains[rows, best]
    proposed = best + 1
    decisions = np.stack([np.where(empty, 1, np.where(max_gain > t, proposed, 0)) for t in GAIN_THRESHOLDS])
    decision_scores = scores[rows, decisions]
    crossfit = np.empty(len(y), dtype=int)
    thresholds = []
    if folds is None:
        folds = GroupKFold(3).split(rows, groups=clusters)
    for fit, hold in folds:
        # Prefer the stricter threshold when Jaccard ties.
        fit_jaccard = decision_scores[:, fit].mean(axis=1)
        best_index = len(GAIN_THRESHOLDS) - 1 - fit_jaccard[::-1].argmax()
        crossfit[hold] = decisions[best_index, hold]
        threshold = GAIN_THRESHOLDS[best_index]
        thresholds.append(threshold if np.isfinite(threshold) else None)

    order = np.flatnonzero(~empty & (max_gain > 0))
    order = order[np.argsort(-max_gain[order], kind="stable")]
    harms = (scores[order, 0] == 1.) & (scores[order, proposed[order]] < 1.)
    cumulative = np.cumsum(harms)
    budgets = {}
    for cap in HARM_BUDGETS:
        selected = empty.astype(int)
        within_budget = order[:np.searchsorted(cumulative, cap, side="right")]
        selected[within_budget] = proposed[within_budget]
        budgets[str(cap)] = edit_summary(y, base, candidates[rows, selected])
    return {
        "cluster_cross_validation": {"thresholds": thresholds, **edit_summary(y, base, candidates[rows, crossfit])},
        "fixed_zero_gain": edit_summary(y, base, candidates[rows, decisions[0]]),
        "matched_harm_budgets": budgets,
    }, crossfit


def proposal_summary(candidates, scores, targets):
    base = scores[:, 0]
    best = scores.max(axis=1)
    prefix_best = scores[:, :MAX_K + 1].max(axis=1)
    base_size = candidates[:, 0].sum(axis=1)
    wrong_single = (base == 0) & (base_size == 1) & (targets.sum(axis=1) == 1)
    return {
        "rows": len(base), "baseline": float(base.mean()), "oracle": float(best.mean()),
        "prefix_oracle": float(prefix_best.mean()),
        "correctable_rows": int((best > base).sum()),
        "mean_base_size": float(base_size.mean()), "empty_base_rows": int((base_size == 0).sum()),
        "wrong_singletons": int(wrong_single.sum()),
        "wrong_singletons_exactly_correctable": int((wrong_single & (best == 1)).sum()),
    }


def run(args):
    started = time.perf_counter()
    data = PeptideData(args.data_root, "fine39")
    train_rows, logits = load_oof(data, args.oof_files)
    probs = np.exp(-np.logaddexp(0., -logits))
    y = data.y[train_rows].astype(bool)
    train_sets, train_valid, train_features = proposal_features(probs, np.char.str_len(data.sequences[train_rows]))
    train_scores = candidate_jaccard(train_sets, y)
    model = fit_selector(train_features, train_valid, train_scores, args.seed)
    # fit_global_threshold expects calibration labels. Here the base predictions
    # come from Train rows unseen by each scoring model.
    calibration_threshold = fit_global_threshold(y, probs)

    val_rows = data.indices("val")
    val_rows = val_rows[np.argsort(data.rows[val_rows])]
    probabilities, base = read_validation(args.baseline, data, val_rows)
    if not np.array_equal(base, probabilities >= .5):
        raise ValueError("Use the fixed-0.50 TheraSet baseline for prefix selection")
    targets = data.y[val_rows].astype(bool)
    candidates, valid, features = proposal_features(probabilities, np.char.str_len(data.sequences[val_rows]))
    scores = candidate_jaccard(candidates, targets)
    gains = predict_gains(model, features, valid)
    rows = np.arange(len(base))
    curve = []
    prefix_curve = []
    for threshold in GAIN_THRESHOLDS:
        choice = selection(gains, threshold)
        prefix_choice = selection(gains[:, :MAX_K], threshold)
        gate = {"gain_threshold": threshold if np.isfinite(threshold) else None}
        curve.append({**gate, **edit_summary(targets, base, candidates[rows, choice])})
        prefix_curve.append({**gate, **edit_summary(targets, base, candidates[rows, prefix_choice])})
    winner = max(range(len(curve)), key=lambda i: (curve[i]["sample_jaccard"], -curve[i]["edited"]))
    decision = selection(gains, GAIN_THRESHOLDS[winner])
    fill = np.where(base.any(axis=1)[:, None], base, candidates[:, 1])
    outcome = {
        "task": "fine39", "split": "val", "seed": args.seed, "elapsed_seconds": time.perf_counter() - started,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "data_root": str(args.data_root), "baseline_run": str(args.baseline),
        "oof_files": [str(path) for path in args.oof_files],
        "train_oof": proposal_summary(train_sets, train_scores, y),
        "validation_candidates": proposal_summary(candidates, scores, targets),
        "baseline": curve[-1],
        "empty_top1_control": edit_summary(targets, base, fill),
        "global_threshold_control": {"threshold_from_train_oof": calibration_threshold,
                                     **edit_summary(targets, base, probabilities >= calibration_threshold)},
        "fixed_zero_gain_selector": curve[0],
        "val_selected_selector": curve[winner],
        "val_selected_prefix_only": max(prefix_curve, key=lambda r: (r["sample_jaccard"], -r["edited"])),
        "selected_candidate_counts": dict(zip(CANDIDATE_NAMES, np.bincount(decision, minlength=len(CANDIDATE_NAMES)).tolist())),
        "gain_threshold_curve": curve,
    }
    exports = {}
    checkpoint = {"model": model, "gain_threshold": outcome["val_selected_selector"]["gain_threshold"],
                  "labels": data.labels}
    if args.neighbor_feature_root is not None:
        print(f"Reading {NEIGHBORS} neighbors per peptide with training-cluster isolation", flush=True)
        train_evidence, val_evidence = fold_neighbor_evidence(data, args.neighbor_feature_root, train_rows, val_rows)
        enhanced_model = fit_selector(add_neighbor_features(train_features, train_sets, train_evidence),
                                      train_valid, train_scores, args.seed)
        enhanced_gains = predict_gains(enhanced_model, add_neighbor_features(features, candidates, val_evidence), valid)
        clusters = data.clusters[val_rows]
        folds = None
        if args.paired_replacements:
            fold_ids = family_folds(clusters, 3)
            folds = [(fold_ids != k, fold_ids == k) for k in range(3)]
        control_result, control_choice = compare_with_fixed_empty_fill(targets, candidates, scores, gains, clusters, folds)
        enhanced_result, enhanced_choice = compare_with_fixed_empty_fill(
            targets, candidates, scores, enhanced_gains, clusters, folds)
        outcome["neighbor_comparison"] = {
            "feature_root": str(args.neighbor_feature_root), "neighbors": NEIGHBORS,
            "elapsed_seconds_including_retrieval": time.perf_counter() - started,
            "scores_only": control_result, "with_neighbor_evidence": enhanced_result,
        }
        checkpoint["neighbor_model"] = enhanced_model
        exports = {"neighbor_candidate_gains": enhanced_gains, "score_only_crossfit_candidate": control_choice,
                   "neighbor_crossfit_candidate": enhanced_choice, "neighbor_label_support": val_evidence[0],
                   "neighbor_best_positive_similarity": val_evidence[1], "neighbor_max_similarity": val_evidence[2]}
        if args.paired_replacements:
            train_pairs = singleton_pair_features(probs, train_sets, np.char.str_len(data.sequences[train_rows]), train_evidence)
            val_pairs = singleton_pair_features(probabilities, candidates, np.char.str_len(data.sequences[val_rows]), val_evidence)
            pair_models = fit_singleton_pairs(train_pairs, train_scores, args.seed)
            pair_gains, keep_value, replace_value = predict_singleton_gains(pair_models, val_pairs, enhanced_gains)
            pair_result, pair_choice = compare_with_fixed_empty_fill(targets, candidates, scores, pair_gains, clusters, folds)
            replacement_mask = (base.sum(axis=1) == 1)[:, None] & (np.arange(valid.shape[1]) >= MAX_K)
            only_replacement = {}
            for arm, arm_gains in (("with_neighbor_evidence", enhanced_gains), ("paired", pair_gains)):
                only_replacement[arm], _ = compare_with_fixed_empty_fill(
                    targets, candidates, scores, np.where(replacement_mask, arm_gains, -np.inf), clusters, folds)
            outcome["paired_replacement_comparison"] = {
                "train_single_label_predictions": len(train_pairs[0]), "val_single_label_predictions": len(val_pairs[0]),
                "elapsed_seconds": time.perf_counter() - started,
                "paired": pair_result, "singleton_replacement_only": only_replacement,
            }
            checkpoint["singleton_pair_models"] = pair_models
            exports.update(paired_candidate_gains=pair_gains, paired_crossfit_candidate=pair_choice,
                           validation_fold=fold_ids, singleton_row_id=data.rows[val_rows[val_pairs[0]]],
                           singleton_keep_value=keep_value, singleton_replacement_values=replace_value)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    joblib.dump(checkpoint, args.output / "selector.joblib")
    np.savez_compressed(
        args.output / "val_predictions.npz",
        row_id=data.rows[val_rows], ID=data.ids[val_rows], cluster_id=data.clusters[val_rows],
        label_names=data.labels, targets=targets, probabilities=probabilities, baseline=base,
        candidate_names=CANDIDATE_NAMES, candidate_gains=gains, selected_candidate=decision,
        predicted=candidates[rows, decision],
        split="val", task=data.track, **exports,
    )
    print(json.dumps({key: val for key, val in outcome.items() if key != "gain_threshold_curve"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="TheraSet run with best_validation.npz")
    parser.add_argument("--oof-files", required=True, type=Path, nargs=3)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=157, type=int)
    parser.add_argument("--data-root", default=Path("data/therapeutic_2025"), type=Path)
    parser.add_argument("--neighbor-feature-root", type=Path,
                        help="Peptide ESM cache for the matched annotation-evidence comparison")
    parser.add_argument("--paired-replacements", action="store_true",
                        help="Compare a keep/replace utility model on base singleton predictions")
    args = parser.parse_args()
    if args.paired_replacements and args.neighbor_feature_root is None:
        parser.error("--paired-replacements requires --neighbor-feature-root")
    run(args)


if __name__ == "__main__":
    main()
