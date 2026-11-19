"""Probe single replacements on nonempty outputs of the fixed Fine39 ensemble."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from evaluate_fine39_fixed_fusion import align, from_zip, logit, read_theraset
from train_fine39_action_gain import actions, choose_actions, fit_action_model, interval, learn_threshold, ranks
from train_fine39_pair_ranker import row_jaccard
from mfp_nova.hypothesis_data import family_folds


def crossfit(x, gains, row, clusters):
    outer = family_folds(clusters, 5)
    selected, thresholds = [], []

    def proposals(fit, held):
        ranker = fit_action_model(x[fit], gains[fit], row[fit])
        indices = np.flatnonzero(held)
        with threadpool_limits(limits=4):
            scores = ranker.predict(x[indices])
        take = choose_actions(row[held], scores, np.arange(len(scores)))
        return indices[take], scores[take]

    for fold in range(5):
        fit_rows = outer != fold
        inner = np.full(len(outer), -1)
        inner[fit_rows] = family_folds(clusters[fit_rows], 3)
        inner_pairs = [
            proposals(fit_rows[row] & (inner[row] != subfold), inner[row] == subfold)
            for subfold in range(3)
        ]
        inner_actions = np.concatenate([pair[0] for pair in inner_pairs])
        inner_scores = np.concatenate([pair[1] for pair in inner_pairs])
        threshold = learn_threshold(inner_scores, gains[inner_actions])
        proposed, score = proposals(fit_rows[row], ~fit_rows[row])
        selected.extend(proposed[score > threshold])
        thresholds.append(None if np.isinf(threshold) else threshold)
    return np.array(selected, dtype=int), thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("theraset", "therapep", "helixset"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    thera = read_theraset(args.theraset, "val")
    cnn = from_zip(args.therapep, "/fine39/traditional5/val.predictions.npz")
    helix = from_zip(args.helixset, "/best_validation.npz")
    a = logit(thera["probabilities"]) - logit(.5)
    b, c = logit(align(cnn, thera)) - logit(.28), logit(align(helix, thera)) - logit(.64)
    fusion = (a + b + c) / 3
    y = thera["targets"].astype(bool)
    thera_top10 = ranks(a) <= 10
    initial = fusion >= 0
    empty = ~initial.any(1)
    baseline = initial.copy()
    baseline[np.flatnonzero(empty), np.where(thera_top10[empty], fusion[empty], -np.inf).argmax(1)] = True
    base_score = row_jaccard(y, baseline)
    base_hit = (baseline & y).any(1)
    if len(y) != 8529 or not np.isclose(base_score.mean(), .6362541338886749):
        raise ValueError("Expected the frozen Fine39 Validation fusion and fallback")
    support = np.log1p(cnn["train_support"])
    x, _, gains, row, new, old, add, _ = actions(fusion, a, baseline, y, support, nominees=thera_top10)
    allowed = ~add & ~empty[row]
    x, gains, row, new, old = (item[allowed] for item in (x, gains, row, new, old))
    extra = []
    for score in (b, c, ranks(b), ranks(c)):
        extra.extend((score[row, new], score[row, old]))
    x = np.column_stack((x, *extra))
    clusters = thera["cluster_id"]
    chosen, thresholds = crossfit(x, gains, row, clusters)
    prediction = baseline.copy()
    prediction[row[chosen], new[chosen]] = True
    prediction[row[chosen], old[chosen]] = False
    actual = row_jaccard(y, prediction)
    delta = actual - base_score
    np.testing.assert_allclose(delta[row[chosen]], gains[chosen], atol=1e-12)
    new_hit = (prediction & y).any(1)
    report = {"scope": "Nested cluster cross-fit on the original Validation for feasibility only",
              "rows": len(y), "nonempty_zero_hits_before": int((~empty & ~base_hit).sum()),
              "baseline_acc_pct": float(100 * base_score.mean()), "edited_acc_pct": float(100 * actual.mean()),
              "delta_pp": float(100 * delta.mean()), "paired_cluster_ci95_pp": interval(delta, clusters),
              "swaps": len(chosen), "thresholds_by_fold": thresholds,
              "improved_rows": int((delta > 0).sum()), "harmed_rows": int((delta < 0).sum()),
              "zero_hit_rows_recovered": int((~base_hit & new_hit).sum()),
              "previous_hit_rows_lost": int((base_hit & ~new_hit).sum())}
    out = Path("runs") / datetime.now().strftime("fine39_fusion_nonempty_swap_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    path = out / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(path.resolve())


if __name__ == "__main__":
    main()
