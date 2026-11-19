"""Learn separate candidate and incumbent label confidence from Fine39 OOF scores."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from evaluate_fine39_oof_fallback import candidate_features, load_branch_oof
from train_fine39_action_gain import actions, assess, fit_action_model, interval, ranks
from train_fine39_pair_ranker import read_val, row_jaccard
from mfp_nova.data import PeptideData
from mfp_nova.hypothesis_data import family_folds


def label_features(base, direct, incumbent, empty, support):
    order = np.broadcast_to(np.arange(base.shape[1]), base.shape)
    features = candidate_features(base, direct, order, support)
    context = np.stack((incumbent.sum(1), empty, base.max(1), direct.max(1)), axis=-1)
    context = np.broadcast_to(context[:, None], (*base.shape, context.shape[-1]))
    return np.concatenate((features, context), axis=-1)


def fit_label_model(features, targets, selected):
    model = HistGradientBoostingClassifier(max_iter=65, max_leaf_nodes=15, min_samples_leaf=100,
                                           l2_regularization=1, early_stopping=False, random_state=42)
    weights = np.maximum(1, selected.sum(1))[:, None]
    weights = np.broadcast_to(1 / weights, selected.shape)[selected]
    with threadpool_limits(limits=4):
        model.fit(features[selected], targets[selected], sample_weight=weights)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", nargs=3, type=Path, required=True)
    parser.add_argument("--evidence-val", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    args = parser.parse_args()
    data = PeptideData(args.data_root, "fine39")
    train, base, direct = load_branch_oof(data, args.oof)
    train_truth = data.y[train].astype(bool)
    support = train_truth.sum(0)
    original = base >= 0
    train_pack = actions(base, direct, original, train_truth, np.log1p(support))
    incumbent = train_pack[-1]
    eligible = ((ranks(base) <= 10) | (ranks(direct) <= 5)) & ~incumbent
    train_features = label_features(base, direct, incumbent, ~original.any(1), support)
    candidate_head = fit_label_model(train_features, train_truth, eligible)
    retain_head = fit_label_model(train_features, train_truth, incumbent)
    del train_features
    tree = fit_action_model(train_pack[0], train_pack[2], train_pack[3])

    val = data.indices("val")
    base, direct, p, q = read_val(data, args.evidence_val, val)
    y = data.y[val].astype(bool)
    original = p >= .5
    if not np.isclose(row_jaccard(y, original).mean(), .6127036236779466):
        raise ValueError("Expected full Validation from the original strong TheraSet checkpoint")
    val_pack = actions(base, direct, original, y, np.log1p(support))
    _, _, _, row, candidate, removed, adds, incumbent = val_pack
    features = label_features(base, direct, incumbent, ~original.any(1), support)
    values = features.reshape(-1, features.shape[-1])
    with threadpool_limits(limits=4):
        p_candidate = candidate_head.predict_proba(values)[:, 1].reshape(base.shape)
        p_retain = retain_head.predict_proba(values)[:, 1].reshape(base.shape)
        tree_score = tree.predict(val_pack[0])
    swap = ~adds
    factorized = np.full(len(adds), -np.inf)
    # P(new observed, old unobserved) minus P(old observed, new unobserved).
    # This identity does not require independence between the two labels.
    factorized[swap] = p_candidate[row[swap], candidate[swap]] - p_retain[row[swap], removed[swap]]
    clusters = data.clusters[val]
    folds = family_folds(clusters, 5)
    hard = ~((y & (ranks(p) <= 5)).any(1) | (y & (ranks(q) <= 5)).any(1))
    report = {"train_rows": len(train), "validation_rows": len(val),
              "incumbent_acc_pct": float(100 * row_jaccard(y, incumbent).mean()),
              "candidate_training_pairs": int(eligible.sum()),
              "retain_training_pairs": int(train_pack[-1].sum()), "comparisons": {}}
    for scope, allowed in (("all", swap), ("originally_nonempty", swap & original.any(1)[row])):
        outcomes = {}
        for name, score in (("direct_jaccard_gain", tree_score), ("candidate_and_retain", factorized)):
            data_report, selected, delta = assess(score, val_pack, y, folds, clusters, hard, allowed,
                                                   return_edits=True)
            data_report["swaps_in_originally_empty"] = int((~original.any(1)[row[selected]]).sum())
            data_report["delta_from_originally_empty_pp"] = float(100 * delta[~original.any(1)].sum() / len(y))
            data_report["delta_from_originally_nonempty_pp"] = float(
                100 * delta[original.any(1)].sum() / len(y))
            outcomes[name] = (data_report, delta)
        a, da = outcomes["direct_jaccard_gain"]
        b, db = outcomes["candidate_and_retain"]
        report["comparisons"][scope] = {
            "direct_jaccard_gain": a, "candidate_and_retain": b,
            "factorized_minus_direct_pp": float(100 * (db - da).mean()),
            "paired_cluster_ci95_factorized_vs_direct_pp": interval(db - da, clusters),
        }
    run = Path("runs") / datetime.now().strftime("fine39_two_head_swap_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    path = run / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
