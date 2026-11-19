"""Compare two TriEvidence evaluations on identical Fine39 rows."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_fine39_fixed_fusion import summarize


def compare(baseline, candidate, nearest=None, structure=None):
    base, other = [json.loads((p / "results.json").read_text(encoding="utf-8"))
                   for p in (baseline, candidate)]
    split = base["split"]
    if split != other["split"]:
        raise ValueError("Compare matching evaluation splits")
    with np.load(baseline / f"{split}_predictions.npz") as a, np.load(candidate / f"{split}_predictions.npz") as b:
        for key in ("row_id", "targets", "cluster_id", "label_names"):
            np.testing.assert_array_equal(a[key], b[key])
        y, clusters = a["targets"].astype(bool), a["cluster_id"]
        logits_delta = {k: float(np.abs(a[k] - b[k]).max())
                        for k in ("theraset_logits", "aa34_logits", "latent_logits") if k in a and k in b}
        strata = {}
        if nearest is not None:
            if split != "val":
                raise ValueError("Nearest-Train strata here are for Validation")
            with np.load(nearest) as source:
                order = np.argsort(source["row_id"])
                np.testing.assert_array_equal(source["row_id"][order], a["row_id"])
                similarity = source["similarity"][order]
            strata.update(nearest_below_070=similarity < .7, nearest_at_least_080=similarity >= .8)
        if structure is not None:
            with np.load(structure) as source:
                order = np.argsort(source["row_id"])
                index = order[np.searchsorted(source["row_id"][order], a["row_id"])]
                np.testing.assert_array_equal(source["row_id"][index], a["row_id"])
                present = source["structure_available"][index]
            strata.update(structure_present=present, structure_missing=~present)
            if nearest is not None:
                strata["low_sim_structure_present"] = strata["nearest_below_070"] & present
                strata["low_sim_structure_missing"] = strata["nearest_below_070"] & ~present
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
            if strata:
                metrics[arm]["strata"] = {}
                gained, lost = y & pred & ~reference, y & reference & ~pred
                for name, keep in strata.items():
                    if not keep.any():
                        continue
                    zeros = np.zeros(keep.sum(), bool)
                    values = summarize(y[keep], pred[keep], scores[keep], clusters[keep], zeros, zeros)
                    metrics[arm]["strata"][name] = {
                        "rows": int(keep.sum()), "delta_pp": values["delta_vs_theraset_pp"],
                        "paired_cluster_ci95_pp": values["paired_cluster_ci95_pp"],
                        "gained_observed_labels": int(gained[keep].sum()), "lost_observed_labels": int(lost[keep].sum())}
    return {"split": split, "rows": len(y), "metrics": metrics,
            "branch_logits_max_abs_delta": logits_delta,
            "both_acc_at_least_baseline": all(v["delta_pp"] >= 0 for v in metrics.values()),
            "parameters": [r.get("parameters") for r in (base, other)],
            "inference": [r.get("inference") for r in (base, other)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--nearest", type=Path, help="Fine39 nearest-Train Validation export")
    parser.add_argument("--structure", type=Path, help="Dataset structure_primary.npz")
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate, args.nearest, args.structure), indent=2))
