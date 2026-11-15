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

from compare_validation_predictions import read_validation, sample_jaccard
from mfp_nova.data import PeptideData
from mfp_nova.hh_v13 import load_oof
from mfp_nova.metrics import fit_global_threshold


MAX_K = 6
RANK_RANGES = [(0, k) for k in range(1, MAX_K + 1)] + [(k, k + 1) for k in range(1, MAX_K)]
CANDIDATE_NAMES = ["keep"] + [f"top_{k}" for k in range(1, MAX_K + 1)] + [
    f"only_rank_{k}" for k in range(2, MAX_K + 1)]
GAIN_THRESHOLDS = (0., .01, .02, .04, .06, .10, .15, .20, np.inf)


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


def fit_selector(features, valid, scores, seed):
    delta = scores[:, 1:] - scores[:, :1]
    weights = np.broadcast_to(1. / valid.sum(axis=1, keepdims=True), valid.shape)
    # The last four fields are label IDs and enter as categorical features.
    model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=15, min_samples_leaf=128, learning_rate=.08,
        categorical_features=list(range(features.shape[-1] - 4, features.shape[-1])),
        early_stopping=False, random_state=seed,
    )
    model.fit(features[valid], delta[valid], sample_weight=weights[valid])
    return model


def predict_gains(model, features, valid):
    gains = np.full(valid.shape, -np.inf)
    gains[valid] = model.predict(features[valid])
    return gains


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
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    joblib.dump({"model": model, "gain_threshold": outcome["val_selected_selector"]["gain_threshold"],
                 "labels": data.labels}, args.output / "selector.joblib")
    np.savez_compressed(
        args.output / "val_predictions.npz",
        row_id=data.rows[val_rows], ID=data.ids[val_rows], cluster_id=data.clusters[val_rows],
        label_names=data.labels, targets=targets, probabilities=probabilities, baseline=base,
        candidate_names=CANDIDATE_NAMES, candidate_gains=gains, selected_candidate=decision,
        predicted=candidates[rows, decision],
        split="val", task=data.track,
    )
    print(json.dumps({key: val for key, val in outcome.items() if key != "gain_threshold_curve"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="TheraSet run with best_validation.npz")
    parser.add_argument("--oof-files", required=True, type=Path, nargs=3)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=157, type=int)
    parser.add_argument("--data-root", default=Path("data/therapeutic_2025"), type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
