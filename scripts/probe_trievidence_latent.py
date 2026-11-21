"""Measure whether pooled HelixSet features add candidate information to branch logits."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def top(scores, k):
    keep = np.zeros(scores.shape, bool)
    np.put_along_axis(keep, np.argsort(-scores, axis=1, kind="stable")[:, :k], True, axis=1)
    return keep


def probe(predictions, features, nearest):
    with np.load(predictions) as p, np.load(nearest) as n:
        if str(p["split"]) != "val":
            raise ValueError("Expected Validation exports")
        rows = p["row_id"]
        order = np.argsort(n["row_id"])
        np.testing.assert_array_equal(rows, n["row_id"][order])
        low = n["similarity"][order] < .7
        y, pred, clusters = p["targets"].astype(bool), p["predicted"], p["cluster_id"]
        logits = [p[key + "_logits"] for key in ("theraset", "aa34", "latent")]
        x = np.column_stack(logits)
    designs = {"logits_only": x}
    if features is not None:
        with np.load(features) as f:
            np.testing.assert_array_equal(rows, f["row_id"])
            designs["logits_and_pooled_latent"] = np.column_stack((x, f["pooled_latent"]))
    scores = {name: np.empty(y.shape) for name in designs}
    with threadpool_limits(limits=4):
        for train, held in GroupKFold(5).split(x, groups=clusters):
            for name, design in designs.items():
                scale = StandardScaler().fit(design[train])
                head = Ridge(alpha=100).fit(scale.transform(design[train]), y[train].astype(float))
                scores[name][held] = head.predict(scale.transform(design[held]))
    result = {"scope": "Exploratory cluster cross-fit on original Validation", "groups": {}}
    for k in (5, 10):
        common = top(logits[0], k) | top(logits[1], k)
        reference = common | top(logits[2], k)
        truth_reference = y & reference
        misses = ~truth_reference.any(1)
        for name, mask in (("all", np.ones(len(y), bool)), ("similarity_lt70", low)):
            key = f"top{k}_{name}"
            group = {"rows": int(mask.sum()), "original_pool_misses": int((mask & misses).sum())}
            for arm, values in scores.items():
                pool = common | top(values, k)
                covered = y & pool
                group[arm] = {
                    "mean_unique_candidates": float(pool[mask].sum(1).mean()),
                    "pool_misses": int((mask & ~covered.any(1)).sum()),
                    "rescued_original_pool_misses": int((mask & misses & covered.any(1)).sum()),
                    "new_pool_misses": int((mask & ~misses & ~covered.any(1)).sum()),
                    "gained_unpredicted_labels": int((covered & ~truth_reference & ~pred)[mask].sum()),
                    "lost_unpredicted_labels": int((truth_reference & ~covered & ~pred)[mask].sum()),
                }
            result["groups"][key] = group
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("nearest", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.predictions, args.features, args.nearest)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(args.output.resolve())
