"""Cross-fit a global fusion threshold on saved Fine39 Validation scores."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--nearest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.baseline) as a, np.load(args.candidate) as b:
        for key in ("row_id", "cluster_id", "targets", "label_names"):
            np.testing.assert_array_equal(a[key], b[key])
        if str(a["split"]) != "val" or str(b["split"]) != "val":
            raise ValueError("Expected Validation exports")
        y = b["targets"].astype(bool)
        ref = [a[k] for k in ("predicted_raw", "predicted")]
        original = [b[k] for k in ("predicted_raw", "predicted")]
        scores, rows = b["probabilities"], b["row_id"]
        _, clusters = np.unique(b["cluster_id"], return_inverse=True)
        shortlist = np.argsort(-b["theraset_logits"], axis=1, kind="stable")[:, :10]
        cfg = json.loads(args.candidate.with_name("results.json").read_text(encoding="utf-8"))
        t = np.asarray(cfg["branch_thresholds"])
        limit = np.log((1 - 1e-6) / 1e-6)
        keys = ("theraset_logits", "aa34_logits", "latent_logits")
        centered = (np.clip(np.stack([b[k] for k in keys]), -limit, limit) -
                    np.log(t / (1 - t)).astype(np.float32)[:, None, None])
    fused_shortlist = np.take_along_axis(centered.mean(0), shortlist, axis=1)
    priority = shortlist + y.shape[1] * (np.arange(10) >= 5)
    label = np.where(fused_shortlist == fused_shortlist.max(1, keepdims=True),
                     priority, 2 * y.shape[1]).min(1) % y.shape[1]
    one = np.zeros_like(y)
    one[np.arange(len(y)), label] = True

    def decide(threshold):
        raw = scores >= threshold
        return raw, raw | (one & ~raw.any(1, keepdims=True))

    def jac(pred):
        return (pred & y).sum(1) / np.maximum(1, (pred | y).sum(1))

    np.testing.assert_array_equal(decide(.5)[1], original[1])
    grid = np.arange(50, 66) / 100
    cached = [decide(t) for t in grid]
    per_row = np.array([[jac(p) for p in pair] for pair in cached])
    folds = np.random.default_rng(42).permutation(clusters.max() + 1)[clusters] % 5
    selected = [np.zeros_like(y), np.zeros_like(y)]
    by_fold = []
    for fold in range(5):
        holdout = folds == fold
        fit = per_row[:, :, ~holdout].mean(2)
        # Favor the threshold with the best worst-case improvement over the current rule.
        index = int(np.argmax((fit - fit[0]).min(1)))
        for dest, pred in zip(selected, cached[index]):
            dest[holdout] = pred[holdout]
        by_fold.append({"fold": fold, "rows": int(holdout.sum()), "threshold": float(grid[index])})
    groups = {"all": np.ones(len(y), bool)}
    if args.nearest:
        with np.load(args.nearest) as z:
            order = np.argsort(z["row_id"])
            np.testing.assert_array_equal(z["row_id"][order], rows)
            sim = z["similarity"][order]
        groups.update(similarity_lt70=sim < .7, similarity_ge80=sim >= .8)
    report = {"scope": "Exploratory cluster cross-fit on the original Validation", "folds": by_fold,
              "fixed_threshold_grid": [{"threshold": float(t), "raw_acc_pct": 100 * s[0].mean(),
                                        "fallback_acc_pct": 100 * s[1].mean()}
                                       for t, s in zip(grid, per_row)], "groups": {}}
    for group, mask in groups.items():
        report["groups"][group] = {}
        for name, pair in (("baseline", ref), ("candidate", original), ("cross_fitted", selected)):
            report["groups"][group][name] = {
                mode: {"acc_pct": float(100 * jac(p)[mask].mean()),
                       "zero_hits": int((~(p & y).any(1) & mask).sum())}
                for mode, p in zip(("raw", "fallback"), pair)}
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
