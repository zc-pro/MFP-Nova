"""Fit a TheraSet label-confidence model on Train OOF scores and evaluate Val."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from evaluate_fine39_fixed_fusion import align, from_zip, logit, read_theraset
from mfp_nova.data import PeptideData
from mfp_nova.hh_v13 import load_oof


def candidate_features(base, direct, order, support):
    def rank(scores):
        return (np.argsort(np.argsort(-scores, axis=1, kind="stable"), axis=1) + 1)

    def gather(scores):
        return np.take_along_axis(scores, order, axis=1)

    return np.stack((
        gather(base), gather(direct), gather(base - direct),
        gather(rank(base)), gather(rank(direct)),
        gather(base) - base.max(1, keepdims=True),
        gather(direct) - direct.max(1, keepdims=True),
        np.log1p(support[order]),
    ), axis=-1)


def load_branch_oof(data, paths):
    train, base = load_oof(data, paths)
    direct = np.empty_like(base)
    train_rows = data.rows[train]
    row_order = np.argsort(train_rows)
    # load_oof validates each fold and the original Train row/label alignment.
    for path in paths:
        with np.load(path) as saved:
            position = row_order[np.searchsorted(train_rows[row_order], saved["row_id"])]
            if not np.array_equal(train_rows[position], saved["row_id"]):
                raise ValueError(f"Train row IDs differ in {path}")
            direct[position] = saved["direct_logits"]
    return train, base, direct


def train_confidence(data, paths):
    train, base, direct = load_branch_oof(data, paths)
    support = data.y[train].sum(0)
    order = np.argsort(-base, axis=1, kind="stable")[:, :10]
    feature = candidate_features(base, direct, order, support)
    targets = np.take_along_axis(data.y[train], order, axis=1)
    model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15,
                                           min_samples_leaf=100, l2_regularization=1, random_state=42,
                                           early_stopping=False)
    with threadpool_limits(limits=3):
        model.fit(feature.reshape(-1, feature.shape[-1]), targets.ravel())
    return model, support, len(train)


def evaluate(model, support, thera, cnn, helix, evidence):
    index = np.argsort(evidence["row_id"])
    index = index[np.searchsorted(evidence["row_id"][index], thera["row_id"])]
    np.testing.assert_array_equal(evidence["row_id"][index], thera["row_id"])
    np.testing.assert_array_equal(evidence["label_names"], thera["label_names"])
    np.testing.assert_allclose(evidence["probabilities"][index], thera["probabilities"], atol=1e-5, rtol=1e-5)
    p, c, h = thera["probabilities"], align(cnn, thera), align(helix, thera)
    base, direct = evidence["logits"][index], evidence["direct_logits"][index]
    fused = (logit(p) + logit(c) - logit(.28) + logit(h) - logit(.64)) / 3
    prediction = fused >= 0
    empty = ~prediction.any(1)
    y = thera["targets"].astype(bool)
    if len(y) != 8529 or float(thera["threshold"]) != .5 or str(thera["split"]) != "val":
        raise ValueError("Expected full original Fine39 Validation")
    order = np.argsort(-p, axis=1, kind="stable")[:, :10]
    missed = ~np.take_along_axis(y, order[:, :5], axis=1).any(1)
    rows = np.arange(len(y))
    ranked_fusion = np.take_along_axis(fused, order, axis=1)
    hi, lo = ranked_fusion[:, :5].argmax(1), ranked_fusion[:, 5:].argmax(1) + 5
    targets = {"top10": ranked_fusion.argmax(1),
               "low_rank_bonus": np.where(ranked_fusion[rows, lo] + 2 > ranked_fusion[rows, hi], lo, hi)}
    feature = candidate_features(base, direct, order, support)
    confidence = model.predict_proba(feature.reshape(-1, feature.shape[-1]))[:, 1].reshape(order.shape)
    baseline = (prediction & y).sum(1) / (prediction | y).sum(1)
    results = {}
    for name, position in targets.items():
        labels = order[rows, position]
        curve = []
        for floor in (0., .10, .15, .20, .25, .30, .40, .50):
            added = empty & (confidence[rows, position] >= floor)
            hits = added & y[rows, labels]
            gain = (hits / y.sum(1)).mean()
            curve.append({"confidence_floor": floor, "acc_pct": float(100 * (baseline.mean() + gain)),
                          "added": int(added.sum()), "hits": int(hits.sum()),
                          "hits_in_theraset_top5_misses": int((hits & missed).sum()),
                          "addition_hit_rate_pct": float(100 * hits.sum() / added.sum()) if added.any() else None})
        results[name] = curve
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", nargs=3, required=True, type=Path)
    parser.add_argument("--theraset-val", required=True, type=Path)
    parser.add_argument("--therapep-archive", required=True, type=Path)
    parser.add_argument("--helixset-archive", required=True, type=Path)
    parser.add_argument("--evidence-val", required=True, type=Path)
    parser.add_argument("--data-root", default="data/therapeutic_2025", type=Path)
    args = parser.parse_args()
    if not args.evidence_val.is_file():
        parser.error("Expected the saved strong-checkpoint evidence_validation.npz")
    data = PeptideData(args.data_root, "fine39")
    model, support, n_train = train_confidence(data, args.oof)
    thera = read_theraset(args.theraset_val, "val")
    cnn = from_zip(args.therapep_archive, "/fine39/traditional5/val.predictions.npz")
    helix = from_zip(args.helixset_archive, "/best_validation.npz")
    with np.load(args.evidence_val) as evidence:
        report = evaluate(model, support, thera, cnn, helix, evidence)
    run = Path("runs") / datetime.now().strftime("fine39_oof_fallback_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    path = run / "summary.json"
    path.write_text(json.dumps({"train_oof_rows": n_train, "split": "val", "curves": report}, indent=2),
                    encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()
