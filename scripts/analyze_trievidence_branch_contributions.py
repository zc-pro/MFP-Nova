"""Measure marginal contribution of each saved TriEvidence branch on Validation."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("row_metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    names = ("theraset", "aa34", "latent")
    cfg = json.loads(args.predictions.with_name("results.json").read_text(encoding="utf-8"))
    with np.load(args.predictions) as z:
        if str(z["split"]) != "val":
            raise ValueError("Expected Validation scores")
        y, rows = z["targets"].astype(bool), z["row_id"]
        logits = np.stack([z[name + "_logits"] for name in names])
        t = np.asarray(cfg["branch_thresholds"])
        limit = np.log((1 - 1e-6) / 1e-6)
        centered = np.clip(logits, -limit, limit) - np.log(t / (1 - t)).astype(np.float32)[:, None, None]
        full, fallback = z["predicted_raw"], z["predicted"]
    with args.row_metadata.open(encoding="utf-8", newline="") as handle:
        meta = {int(r["row_id"]): r for r in csv.DictReader(handle)}
    sim = np.array([float(meta[int(i)]["similarity"]) for i in rows])
    length = np.array([int(meta[int(i)]["length"]) for i in rows])
    groups = {"all": np.ones(len(rows), dtype=bool), "similarity_lt70": sim < .7,
              "similarity_70_to_80": (sim >= .7) & (sim < .8), "similarity_ge80": sim >= .8,
              "length_2_8": length <= 8, "length_9_16": (length > 8) & (length <= 16),
              "length_17_32": (length > 16) & (length <= 32), "length_33_plus": length > 32,
              "one_label": y.sum(1) == 1, "two_labels": y.sum(1) == 2,
              "three_plus_labels": y.sum(1) >= 3}

    def measure(pred):
        hits = (pred & y).sum(1)
        return hits / np.maximum(1, (pred | y).sum(1)), hits

    full_score, full_hits = measure(full)
    arms, removal = {}, {}
    for size in (1, 2, 3):
        for indices in combinations(range(3), size):
            pred = centered[list(indices)].mean(0) >= 0
            score, hits = measure(pred)
            arm = "+".join(names[i] for i in indices)
            arms[arm] = {"acc_pct": 100 * score.mean(), "zero_hit": int((hits == 0).sum())}
            if size == 3:
                np.testing.assert_array_equal(pred, full)
            if size == 2:
                removed = next(i for i in range(3) if i not in indices)
                removal[names[removed]] = {
                    key: {"rows": int(mask.sum()), "adding_delta_pp": 100 * (full_score - score)[mask].mean(),
                          "rescued": int((mask & (hits == 0) & (full_hits > 0)).sum()),
                          "lost": int((mask & (hits > 0) & (full_hits == 0)).sum())}
                    for key, mask in groups.items() if mask.any()
                }
    fallback_score, fallback_hits = measure(fallback)
    report = {"rows": len(rows), "thresholds": t.tolist(), "raw_arms": arms,
              "adding_to_other_two": removal,
              "fallback": {"acc_pct": 100 * fallback_score.mean(), "eligible_empty": int((~full.any(1)).sum()),
                           "rescued": int(((full_hits == 0) & (fallback_hits > 0)).sum())}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
