"""Test 1–2 extra candidates without retraining the Fine39 decision model."""

import argparse
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from evaluate_fine39_oof_fallback import load_branch_oof
from train_fine39_action_gain import actions, assess, choose_actions, fit_action_model, interval, learn_threshold, ranks
from train_fine39_pair_ranker import read_val, row_jaccard
from mfp_nova.data import PeptideData
from mfp_nova.hypothesis_data import family_folds


def read_candidate_scores(path, ids, labels, targets):
    scores = {}
    with tarfile.open(path) as archive:
        for mode in ("pool", "span"):
            with np.load(io.BytesIO(archive.extractfile(f"{mode}/val_logits.npz").read())) as export:
                order = np.argsort(export["row_id"])
                index = order[np.searchsorted(export["row_id"][order], ids)]
                np.testing.assert_array_equal(export["row_id"][index], ids)
                np.testing.assert_array_equal(export["label_names"], labels)
                np.testing.assert_array_equal(export["targets"][index], targets)
                scores[mode] = export["logits"][index]
    return scores


def max_swap_gain(pack):
    _, _, gain, rows, _, _, add, incumbent = pack
    best = np.zeros(len(incumbent))
    np.maximum.at(best, rows[~add], gain[~add])
    return best


def crossfit_span_only(pack, nominees, span, clusters, hard, missing, base_selected, base_delta):
    """Evaluate a separate gate fitted only to the extra span candidates of other Val clusters."""
    x, _, gains, rows, labels, remove, add, incumbent = pack
    valid = np.flatnonzero(~add & nominees[rows, labels])
    row, label, removed = rows[valid], labels[valid], remove[valid]
    span_rank = ranks(span)
    features = np.column_stack((x[valid], span[row, label], span[row, removed],
                                span_rank[row, label], span_rank[row, removed]))
    gains = gains[valid]
    folds = family_folds(clusters, 5)
    forbidden = np.zeros(len(clusters), dtype=bool)
    forbidden[base_selected] = True
    selected, proposed, cutoffs = [], [], []

    def predict(fit, evaluate):
        model = fit_action_model(features[fit], gains[fit], row[fit])
        with threadpool_limits(limits=4):
            score = model.predict(features[evaluate])
        subset = np.flatnonzero(evaluate)
        which = choose_actions(row[evaluate], score, np.arange(len(subset)))
        return subset[which], score[which]

    for fold in range(5):
        held = folds == fold
        fit = ~held
        inner = np.full(len(folds), -1)
        inner[fit] = family_folds(clusters[fit], 3)
        inner_items, inner_scores = [], []
        for part in range(3):
            item, score = predict(fit[row] & (inner[row] != part), inner[row] == part)
            inner_items.extend(item)
            inner_scores.extend(score)
        threshold = learn_threshold(np.array(inner_scores), gains[np.array(inner_items, dtype=int)])
        item, score = predict(fit[row], held[row])
        proposed.extend(item[~forbidden[row[item]]])
        selected.extend(item[(score > threshold) & ~forbidden[row[item]]])
        cutoffs.append(None if np.isinf(threshold) else threshold)

    selected = np.array(selected, dtype=int)
    proposed = np.array(proposed, dtype=int)
    extra = np.zeros(len(clusters))
    extra[row[selected]] = gains[selected]
    return {
        "scope": "Nested cluster cross-fit on original Validation, Train OOF score decision takes priority",
        "extra_swaps": len(selected), "thresholds_by_fold": cutoffs,
        "positive_proposals_before_threshold_in_293": int(((gains[proposed] > 0) & missing[row[proposed]]).sum()),
        "new_hits_in_293": int(((gains[selected] > 0) & missing[row[selected]]).sum()),
        "new_hits_in_655": int(((gains[selected] > 0) & hard[row[selected]]).sum()),
        "improved_rows": int((extra > 0).sum()), "harmed_rows": int((extra < 0).sum()),
        "delta_vs_original_pool_pp": float(100 * extra.mean()),
        "paired_ci95_vs_original_pool_pp": interval(extra, clusters),
        "delta_vs_single_model_pp": float(100 * (base_delta + extra).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", nargs=3, type=Path, required=True)
    parser.add_argument("--evidence-val", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    args = parser.parse_args()
    data = PeptideData(args.data_root, "fine39")
    train, train_base, train_direct = load_branch_oof(data, args.oof)
    support = np.log1p(data.y[train].sum(0))
    x, _, gain, row, *_ = actions(train_base, train_direct, train_base >= 0, data.y[train].astype(bool), support)
    model = fit_action_model(x, gain, row)
    val = data.indices("val")
    base, direct, prob, direct_prob = read_val(data, args.evidence_val, val)
    targets = data.y[val].astype(bool)
    clusters = data.clusters[val]
    folds = family_folds(clusters, 5)
    original = prob >= .5
    if not np.isclose(row_jaccard(targets, original).mean(), .6127036236779466):
        raise ValueError("Expected the historical 61.27% TheraSet Validation predictions")
    hard = ~((targets & (ranks(prob) <= 5)).any(1) | (targets & (ranks(direct_prob) <= 5)).any(1))
    pool = (ranks(base) <= 10) | (ranks(direct) <= 5)
    original_pack = actions(base, direct, original, targets, support)
    original_coverage = max_swap_gain(original_pack)
    missing = hard & (original_coverage == 0)

    def decisions(pack):
        with threadpool_limits(limits=4):
            scores = model.predict(pack[0])
        return assess(scores, pack, targets, folds, clusters, hard, ~pack[-2], return_edits=True)

    baseline, selected_base, base_delta = decisions(original_pack)
    result = {
        "train_rows": len(train), "validation_rows": len(val), "double_miss_rows": int(hard.sum()),
        "unreachable_with_original_pool": int(missing.sum()),
        "original_pool": baseline,
        "expanded": {},
    }
    head_scores = read_candidate_scores(args.candidate_archive, data.rows[val], data.labels, targets)
    head_scores.update(final_next=base, direct_next=direct)
    for name, scores in head_scores.items():
        order = np.argsort(-np.where(pool, -np.inf, scores), axis=1, kind="stable")
        for count in (1, 2):
            nominees = np.zeros_like(pool)
            np.put_along_axis(nominees, order[:, :count], True, axis=1)
            pack = actions(base, direct, original, targets, support, nominees)
            coverage = max_swap_gain(pack)
            decided, selected, delta = decisions(pack)
            row, label = pack[3:5]
            nominated_swap = selected[nominees[row[selected], label[selected]]]
            result["expanded"][f"{name}_{count}"] = {
                "newly_recoverable_in_293": int((missing & (coverage > 0)).sum()),
                "newly_recoverable_all_rows": int(((original_coverage == 0) & (coverage > 0)).sum()),
                "oracle_gain_pp_from_extra_candidates": float(100 * (coverage - original_coverage).mean()),
                "executed_swaps_using_new_label": len(nominated_swap),
                "executed_new_label_swaps_hard": int(hard[row[nominated_swap]].sum()),
                "beneficial_new_label_swaps": int((pack[2][nominated_swap] > 0).sum()),
                "delta_vs_original_pool_pp": float(100 * (delta - base_delta).mean()),
                "paired_ci95_vs_original_pool_pp": interval(delta - base_delta, clusters),
                **decided,
            }
            if name == "span" and count == 2:
                result["span2_dedicated_gate"] = crossfit_span_only(
                    pack, nominees, scores, clusters, hard, missing, original_pack[3][selected_base], base_delta)
                result["span2_dedicated_gate"]["acc_pct"] = (
                    baseline["acc_pct"] + result["span2_dedicated_gate"]["delta_vs_original_pool_pp"])
    run = Path("runs") / datetime.now().strftime("fine39_candidate_expansion_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    path = run / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
