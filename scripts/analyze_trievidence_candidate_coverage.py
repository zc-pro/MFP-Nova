"""Locate Fine39 labels absent from existing TriEvidence candidate rankings."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def candidate_pool(rankings, shape, k):
    proposals = np.zeros((3, *shape), bool)
    for mask, order in zip(proposals, rankings):
        np.put_along_axis(mask, order[:, :k], True, axis=1)
    return proposals


def analyze(path, metadata):
    with np.load(path) as data:
        if str(data["split"]) != "val":
            raise ValueError("Expected Validation predictions")
        rows = data["row_id"]
        truth = data["targets"].astype(bool)
        pred = data["predicted"].astype(bool)
        labels = data["label_names"]
        names = ("theraset", "aa34", "latent")
        rankings = [np.argsort(-data[name + "_logits"], axis=1, kind="stable") for name in names]
    with metadata.open(encoding="utf-8", newline="") as handle:
        meta = {int(row["row_id"]): row for row in csv.DictReader(handle)}
    similarity = np.array([float(meta[int(row)]["similarity"]) for row in rows])
    length = np.array([int(meta[int(row)]["length"]) for row in rows])
    counts = truth.sum(1)
    hits = (truth & pred).sum(1)
    jaccard = hits / np.maximum(1, (truth | pred).sum(1))
    strata = {"all": np.ones(len(rows), bool), "one_label": counts == 1,
              "two_labels": counts == 2, "three_plus_labels": counts >= 3,
              "similarity_lt70": similarity < .7,
              "similarity_70_to_80": (similarity >= .7) & (similarity < .8),
              "similarity_ge80": similarity >= .8,
              "length_2_8": length <= 8, "length_9_16": (length > 8) & (length <= 16),
              "length_17_32": (length > 16) & (length <= 32), "length_33_plus": length > 32}
    result = {"rows": len(rows), "baseline_acc_pct": 100 * jaccard.mean(), "by_k": {}}
    for k in (5, 10, 15):
        proposals = candidate_pool(rankings, truth.shape, k)
        pool = proposals.any(0)
        covered = (truth & pool).sum(1)
        missing = truth & ~pool
        outside_final = missing & ~pred
        by_group = {}
        for name, mask in strata.items():
            recorded = int(counts[mask].sum())
            if not recorded:
                continue
            by_group[name] = {
                "rows": int(mask.sum()),
                "no_true_candidate": int((mask & (covered == 0) & (counts > 0)).sum()),
                "has_missing_true_label": int((mask & (covered < counts)).sum()),
                "missing_true_labels": int(missing[mask].sum()),
                "recorded_labels": recorded,
                "baseline_zero_hits": int((mask & (hits == 0)).sum()),
                "baseline_zero_hits_with_true_candidate": int((mask & (hits == 0) & (covered > 0)).sum()),
                "unpredicted_labels_in_pool": int((truth & pool & ~pred)[mask].sum()),
                "unpredicted_labels_outside_pool": int(outside_final[mask].sum()),
                "oracle_acc_pct_pool_plus_final": float(100 * (
                    (truth & (pool | pred))[mask].sum(1) / np.maximum(1, counts[mask])).mean()),
            }
        result["by_k"][k] = {"groups": by_group,
                             "mean_unique_candidates": float(pool.sum(1).mean())}
        if k == 10:
            result["by_k"][k]["per_label"] = [
                {"label": str(label), "support": int(truth[:, i].sum()),
                 "missing": int(missing[:, i].sum())}
                for i, label in enumerate(labels)]
            result["by_k"][k]["unique_true_proposals"] = {
                name: {key: int((truth & proposals[i] & (proposals.sum(0) == 1))[mask].sum())
                       for key, mask in strata.items()}
                for i, name in enumerate(names)}
    return result


def compare(paths, nearest):
    keys = ("row_id", "targets", "cluster_id", "label_names", "theraset_logits", "aa34_logits")
    exports = []
    for path in paths:
        with np.load(path) as z:
            if str(z["split"]) != "val":
                raise ValueError("Expected Validation predictions")
            exports.append({key: z[key] for key in (*keys, "latent_logits", "predicted")})
    original, control, ranking = exports
    for candidate in (control, ranking):
        for key in keys:
            np.testing.assert_array_equal(original[key], candidate[key], err_msg=key)
    with np.load(nearest) as z:
        order = np.argsort(z["row_id"])
        np.testing.assert_array_equal(original["row_id"], z["row_id"][order])
        low = z["similarity"][order] < .7
    truth = original["targets"].astype(bool)
    strata = {"all": np.ones(len(truth), bool), "similarity_lt70": low, "multilabel": truth.sum(1) >= 2}
    orders = [[np.argsort(-z[key], axis=1, kind="stable")
               for key in ("theraset_logits", "aa34_logits", "latent_logits")] for z in exports]
    report = {"scope": "Paired Fine39 Validation candidate coverage", "by_k": {}}
    for k in (5, 10):
        pools = [candidate_pool(order, truth.shape, k).any(0) for order in orders]
        covered = [pool & truth for pool in pools]
        result = {}
        for name, mask in strata.items():
            _, cluster = np.unique(original["cluster_id"][mask], return_inverse=True)
            sizes = np.bincount(cluster)
            draws = np.random.default_rng(42).integers(len(sizes), size=(2000, len(sizes)))
            denominator = sizes[draws].sum(1)
            result[name] = {"rows": int(mask.sum()), "arms": {}, "paired": {}}
            for i, label in enumerate(("original", "control", "ranking")):
                hits = covered[i][mask]
                predicted = exports[i]["predicted"][mask]
                result[name]["arms"][label] = {
                    "pool_misses": int((~hits.any(1)).sum()),
                    "covered_recorded_labels": int(hits.sum()),
                    "mean_unique_candidates": float(pools[i][mask].sum(1).mean()),
                    "zero_hits_with_true_candidate": int((hits.any(1) & ~(predicted & truth[mask]).any(1)).sum()),
                }
            for i, j, key in ((0, 1, "control_vs_original"), (0, 2, "ranking_vs_original"),
                              (1, 2, "ranking_vs_control")):
                before, after = covered[i][mask], covered[j][mask]
                old_hit, new_hit = before.any(1), after.any(1)
                delta = new_hit.astype(int) - old_hit
                gains = np.bincount(cluster, weights=delta)
                result[name]["paired"][key] = {
                    "rescued_pool_misses": int((~old_hit & new_hit).sum()),
                    "new_pool_misses": int((old_hit & ~new_hit).sum()),
                    "gained_labels": int((after & ~before).sum()),
                    "lost_labels": int((before & ~after).sum()),
                    "pool_hit_delta_pp": float(100 * delta.mean()),
                    "paired_cluster_ci95_pp": np.quantile(
                        100 * gains[draws].sum(1) / denominator, [.025, .975]).tolist(),
                }
        report["by_k"][k] = result
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("metadata", type=Path, help="Validation metadata CSV, or nearest NPZ in paired mode")
    parser.add_argument("--paired", nargs=2, type=Path, metavar=("CONTROL", "RANKING"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (compare([args.predictions, *args.paired], args.metadata) if args.paired
              else analyze(args.predictions, args.metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(args.output.resolve())
