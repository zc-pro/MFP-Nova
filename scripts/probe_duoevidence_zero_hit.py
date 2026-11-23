"""Cluster-held-out offline probe of one-label replacements in DuoEvidence."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits


def probe(path, seed):
    with np.load(path) as saved:
        if str(saved["split"]) != "val":
            raise ValueError("Use the original Validation export")
        y, raw, final = (saved[key].astype(bool) for key in ("targets", "predicted_raw", "predicted"))
        a, b = (saved[key] for key in ("theraset_logits", "aa34_logits"))
        offsets = saved["offsets"]
        _, clusters = np.unique(saved["cluster_id"], return_inverse=True)
    if len(y) != 8529 or y.shape[1] != 39:
        raise ValueError("Expected the original Fine39 Validation")
    limit = np.log(999999.)
    fused = ((np.clip(a, -limit, limit) - offsets[0]) + (np.clip(b, -limit, limit) - offsets[1])) / 2
    rank_a = np.argsort(-a, axis=1, kind="stable").argsort(axis=1)
    rank_b = np.argsort(-b, axis=1, kind="stable").argsort(axis=1)
    hits = (final & y).sum(1)
    count, truth = final.sum(1), y.sum(1)
    union = count + truth - hits
    baseline = hits / np.maximum(1, union)
    # Preserve cardinality and replace only the weakest label in a nonempty raw set.
    remove = np.where(raw, fused, np.inf).argmin(1)
    pool = ((rank_a < 10) | (rank_b < 10)) & ~raw & raw.any(1, keepdims=True)
    rows, labels = np.where(pool)
    eligible = pool.any(1)
    gone = remove[rows]
    new_hits = hits[rows] - y[rows, gone] + y[rows, labels]
    gains = new_hits / (count[rows] + truth[rows] - new_hits) - baseline[rows]
    rescue = (hits == 0) & (pool & y).any(1)

    ra, rb = a.max(1), b.max(1)
    sorted_fused = np.sort(fused, axis=1)
    row_features = np.column_stack((count, ra, rb, sorted_fused[:, -1], sorted_fused[:, -2],
                                    fused[np.arange(len(y)), remove],
                                    ((rank_a < 3) & (rank_b < 3)).sum(1),
                                    (rank_a.argmin(1) == rank_b.argmin(1))))
    action_features = np.column_stack((
        row_features[rows], a[rows, labels], b[rows, labels], fused[rows, labels],
        rank_a[rows, labels], rank_b[rows, labels], rank_a[rows, gone], rank_b[rows, gone],
        a[rows, labels] - a[rows, gone], b[rows, labels] - b[rows, gone],
        fused[rows, labels] - fused[rows, gone]))
    fold = np.random.default_rng(seed).permutation(clusters.max() + 1)[clusters] % 5
    gate = np.zeros(len(y))
    pair_probability = np.zeros(len(rows))
    expected_gain = np.zeros(len(rows))
    model_options = dict(max_iter=100, max_leaf_nodes=15, l2_regularization=2, random_state=seed)
    with threadpool_limits(limits=2):
        for k in range(5):
            held = fold == k
            train_rows = eligible & ~held
            held_rows = eligible & held
            action_held = held[rows]
            gate_model = HistGradientBoostingClassifier(**model_options)
            gate_model.fit(row_features[train_rows], rescue[train_rows])
            gate[held_rows] = gate_model.predict_proba(row_features[held_rows])[:, 1]
            # This model ranks candidates conditional on an original zero hit.
            rank_train = ~action_held & (hits[rows] == 0)
            ranker = HistGradientBoostingClassifier(**model_options)
            ranker.fit(action_features[rank_train], y[rows[rank_train], labels[rank_train]])
            pair_probability[action_held] = ranker.predict_proba(action_features[action_held])[:, 1]
            regressor = HistGradientBoostingRegressor(**model_options)
            regressor.fit(action_features[~action_held], gains[~action_held])
            expected_gain[action_held] = regressor.predict(action_features[action_held])

    def policies(action_score, board=None):
        order = np.lexsort((-action_score, rows))
        _, first = np.unique(rows[order], return_index=True)
        best = order[first]
        rank_score = action_score[best] if board is None else board[rows[best]]
        chosen = best[np.argsort(-rank_score, kind="stable")]

        def measure(actions):
            delta = gains[actions]
            old_hits, updated = hits[rows[actions]], new_hits[actions]
            return {"edits": len(actions), "delta_pp": float(100 * delta.sum() / len(y)),
                    "better": int((delta > 0).sum()), "worse": int((delta < 0).sum()),
                    "zero_rescued": int(((old_hits == 0) & (updated > 0)).sum()),
                    "zero_created": int(((old_hits > 0) & (updated == 0)).sum())}

        result = {str(budget): measure(chosen[:budget]) for budget in (50, 100, 200, 400)}
        if board is None:
            result["positive_expected_gain"] = measure(best[rank_score > 0])
        return result, best

    gated, best = policies(pair_probability, gate)
    direct, _ = policies(expected_gain)
    return {"scope": "Exploratory five-fold cluster cross-fit within original Validation",
            "rows": len(y), "clusters": int(clusters.max() + 1), "seed": seed,
            "baseline_acc": float(baseline.mean()), "eligible_rows": int(eligible.sum()),
            "recoverable_zero_hit_rows": int(rescue.sum()),
            "oracle_zero_hit_replacement_gain_pp": float(100 * np.sum(1 / (count[rescue] + truth[rescue] - 1)) / len(y)),
            "recoverable_prevalence": float(rescue[eligible].mean()),
            "gate_average_precision": float(average_precision_score(rescue[eligible], gate[eligible])),
            "gate_auroc": float(roc_auc_score(rescue[eligible], gate[eligible])),
            "candidate_success_given_recoverable": float(y[rows[best], labels[best]][rescue[rows[best]]].mean()),
            "gate_then_rank": gated, "direct_expected_gain": direct}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.predictions, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as report:
        json.dump(result, report, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
