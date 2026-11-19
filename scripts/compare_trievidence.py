"""Compare two TriEvidence evaluations on identical Fine39 rows."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_fine39_fixed_fusion import summarize


def compare(baseline, candidate):
    base, other = [json.loads((p / "results.json").read_text(encoding="utf-8"))
                   for p in (baseline, candidate)]
    split = base["split"]
    if split != other["split"]:
        raise ValueError("Compare matching evaluation splits")
    with np.load(baseline / f"{split}_predictions.npz") as a, np.load(candidate / f"{split}_predictions.npz") as b:
        for key in ("row_id", "targets", "cluster_id", "label_names"):
            np.testing.assert_array_equal(a[key], b[key])
        y, clusters = a["targets"].astype(bool), a["cluster_id"]
        metrics = {}
        for key, arm in (("predicted_raw", "fusion"), ("predicted", "fusion_empty_top10")):
            reference, pred = a[key], b[key]
            scores = (y & reference).sum(1) / np.maximum((y | reference).sum(1), 1)
            delta = summarize(y, pred, scores, clusters, np.zeros(len(y), bool), np.zeros(len(y), bool))
            metrics[arm] = {"baseline_acc_pct": 100 * float(scores.mean()), "candidate_acc_pct": delta["acc_pct"],
                            "delta_pp": delta["delta_vs_theraset_pp"],
                            "paired_cluster_ci95_pp": delta["paired_cluster_ci95_pp"],
                            "changed_rows": int((reference != pred).any(1).sum()),
                            "zero_hit_baseline": base["arms"][arm]["zero_hit_rows"],
                            "zero_hit_candidate": other["arms"][arm]["zero_hit_rows"],
                            "multifunction_jaccard_baseline": base["arms"][arm].get("multifunction_sample_jaccard"),
                            "multifunction_jaccard_candidate": other["arms"][arm].get("multifunction_sample_jaccard")}
    return {"split": split, "rows": len(y), "metrics": metrics,
            "both_acc_at_least_baseline": all(v["delta_pp"] >= 0 for v in metrics.values()),
            "parameters": [r.get("parameters") for r in (base, other)],
            "inference": [r.get("inference") for r in (base, other)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate), indent=2))
