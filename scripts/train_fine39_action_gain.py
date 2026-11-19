"""Learn the observed Jaccard gain of Fine39 add/swap actions from Train OOF."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from evaluate_fine39_oof_fallback import load_branch_oof
from train_fine39_pair_ranker import read_val, row_jaccard
from mfp_nova.data import PeptideData
from mfp_nova.hypothesis_data import family_folds


def ranks(scores):
    return np.argsort(np.argsort(-scores, axis=1, kind="stable"), axis=1) + 1


def actions(base, direct, predicted, y, support):
    rank, direct_rank = ranks(base), ranks(direct)
    incumbent = predicted.copy()
    empty = ~predicted.any(1)
    incumbent[np.flatnonzero(empty), base[empty].argmax(1)] = True
    row, candidate = np.where(((rank <= 10) | (direct_rank <= 5)) & ~incumbent)
    position, remove = np.where(incumbent[row])
    n_add = len(row)
    action = np.concatenate((np.arange(n_add), position))
    row, candidate = row[action], candidate[action]
    sentinel = y.shape[1]
    remove = np.concatenate((np.full(n_add, sentinel), remove))
    add = remove == sentinel

    size = incumbent.sum(1)
    overlap = (incumbent & y).sum(1)
    observed = np.pad(y, ((0, 0), (0, 1)))
    new_overlap = overlap[row] + y[row, candidate].astype(int) - observed[row, remove]
    new_size = size[row] + add
    baseline = row_jaccard(y, incumbent)
    gain = new_overlap / (new_size + y.sum(1)[row] - new_overlap) - baseline[row]

    features = [size[row], empty[row], add]
    for scores in (base, rank, base - base.max(1, keepdims=True), np.broadcast_to(support, base.shape)):
        padded = np.pad(scores.astype(np.float32), ((0, 0), (0, 1)), constant_values=np.nan)
        features.extend((scores[row, candidate], padded[row, remove]))
    simple_width = len(features)
    for scores in (direct, direct_rank, base - direct, direct - direct.max(1, keepdims=True)):
        padded = np.pad(scores.astype(np.float32), ((0, 0), (0, 1)), constant_values=np.nan)
        features.extend((scores[row, candidate], padded[row, remove]))
    return np.column_stack(features), simple_width, gain, row, candidate, remove, add, incumbent


def choose_actions(row, scores, choices):
    order = choices[np.argsort(-scores[choices], kind="stable")]
    _, first = np.unique(row[order], return_index=True)
    return order[first]


def interval(delta, clusters):
    _, group = np.unique(clusters, return_inverse=True)
    counts = np.bincount(group)
    sums = np.bincount(group, weights=delta)
    rng = np.random.default_rng(42)
    values = []
    for _ in range(2000):
        draw = rng.integers(len(counts), size=len(counts))
        values.append(100 * sums[draw].sum() / counts[draw].sum())
    return np.quantile(values, [.025, .975]).tolist()


def assess(scores, pack, y, folds, clusters, hard, allowed):
    _, _, gain, row, label, remove, add, incumbent = pack
    candidate = choose_actions(row, scores, np.flatnonzero(allowed))
    fixed_thresholds = (float("inf"), .15, .10, .075, .05, .025, .01, 0.)
    selected = []
    used_threshold = []
    held_fold = folds[row[candidate]]
    for fold in range(5):
        fit = candidate[held_fold != fold]
        quantile_thresholds = np.quantile(scores[fit], [.80, .90, .95, .975, .99, .995, .999])
        threshold = max((*fixed_thresholds, *quantile_thresholds),
                        key=lambda cutoff: gain[fit[scores[fit] > cutoff]].sum())
        held = candidate[(held_fold == fold) & (scores[candidate] > threshold)]
        selected.extend(held)
        used_threshold.append(None if np.isinf(threshold) else threshold)
    selected = np.array(selected, dtype=int)
    edited = incumbent.copy()
    edited[row[selected], label[selected]] = True
    swaps = selected[~add[selected]]
    edited[row[swaps], remove[swaps]] = False
    baseline = row_jaccard(y, incumbent)
    result = row_jaccard(y, edited)
    delta = result - baseline
    np.testing.assert_allclose(delta[row[selected]], gain[selected], atol=1e-12)
    before, after = (incumbent & y).any(1), (edited & y).any(1)
    return {
        "acc_pct": float(100 * result.mean()), "delta_pp": float(100 * delta.mean()),
        "paired_cluster_ci95_pp": interval(delta, clusters),
        "thresholds_by_fold": used_threshold, "actions_executed": len(selected),
        "adds": int(add[selected].sum()), "swaps": len(swaps),
        "improved": int((delta > 0).sum()), "harmed": int((delta < 0).sum()),
        "new_hit_rows": int((~before & after).sum()), "lost_hit_rows": int((before & ~after).sum()),
        "hard_rows_recovered": int((hard & ~before & after).sum()),
        "zero_hit_rows": int((~after).sum()),
        "proposed_action_positive": int((gain[candidate] > 0).sum()),
        "proposed_action_negative": int((gain[candidate] < 0).sum()),
        "proposed_action_score_range": [float(scores[candidate].min()), float(scores[candidate].max())],
        "oracle_gain_pp_on_proposed_actions": float(100 * np.maximum(gain[candidate], 0).sum() / len(y)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", nargs=3, type=Path, required=True)
    parser.add_argument("--evidence-val", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    args = parser.parse_args()
    data = PeptideData(args.data_root, "fine39")
    train, base, direct = load_branch_oof(data, args.oof)
    support = np.log1p(data.y[train].sum(0))
    train_targets = data.y[train].astype(bool)
    train_oof_acc = float(100 * row_jaccard(train_targets, base >= 0).mean())
    x, width, gain, row, *_ = actions(base, direct, base >= 0, train_targets, support)
    val = data.indices("val")
    base, direct, base_prob, direct_prob = read_val(data, args.evidence_val, val)
    y = data.y[val].astype(bool)
    primary = base_prob >= .5
    if not np.isclose(row_jaccard(y, primary).mean(), .6127036236779466):
        raise ValueError("Expected the historical strong TheraSet checkpoint")
    pack = actions(base, direct, primary, y, support)
    clusters = data.clusters[val]
    folds = family_folds(clusters, 5)
    hard = ~((y & (ranks(base_prob) <= 5)).any(1) | (y & (ranks(direct_prob) <= 5)).any(1))
    result = {"train_rows": len(train), "validation_rows": len(val), "train_actions": len(gain),
              "train_actions_positive": int((gain > 0).sum()), "hard_rows": int(hard.sum()),
              "theraset_train_oof_acc_pct": train_oof_acc,
              "theraset_val_acc_pct": float(100 * row_jaccard(y, primary).mean()),
              "incumbent_acc_pct": float(100 * row_jaccard(y, pack[-1]).mean()), "arms": {}}
    weights = 1 / np.bincount(row)[row]
    for mode, columns in (("final_scores", width), ("with_direct_and_correction", x.shape[1])):
        print(f"Fitting {mode} on {len(gain)} Train actions", flush=True)
        model = HistGradientBoostingRegressor(max_iter=65, max_leaf_nodes=15, min_samples_leaf=100,
                                              l2_regularization=1, early_stopping=False, random_state=42)
        with threadpool_limits(limits=4):
            model.fit(x[:, :columns], gain, sample_weight=weights)
            predictions = model.predict(pack[0][:, :columns])
        add = pack[-2]
        result["arms"][mode] = {
            name: assess(predictions, pack, y, folds, clusters, hard, selected)
            for name, selected in (("add_or_swap", np.ones_like(add)), ("swap_only", ~add), ("add_only", add))
        }
        print(json.dumps(result["arms"][mode], indent=2), flush=True)
    run = Path("runs") / datetime.now().strftime("fine39_action_gain_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    path = run / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
