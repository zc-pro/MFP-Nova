"""Audit fixed Validation cohorts and fixed fallback on saved DuoEvidence exports."""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from mfp_nova.data import PeptideData
from mfp_nova.reference_fallback import apply_train_reference, paired_cluster_interval
from probe_duoevidence_helix import compare


def audit(data, base, pointwise, relative, helix):
    val = data.indices("val")
    for export in (base, pointwise, relative):
        if str(export["split"]) != "val":
            raise ValueError("Use Validation predictions")
        for key, expected in (("row_id", data.rows[val]), ("targets", data.y[val]),
                              ("cluster_id", data.clusters[val]), ("label_names", data.labels)):
            np.testing.assert_array_equal(export[key], expected)
    y, clusters = data.y[val].astype(bool), data.clusters[val]
    helix_rescue, cohorts = {}, {}
    compare(base, helix, masks=helix_rescue, cohorts=cohorts)
    empty = ~base["predicted_raw"].any(1)
    def jaccard(pred):
        return (pred & y).sum(1) / (pred | y).sum(1).clip(min=1)

    def inspect(export, fixed):
        raw = export["predicted_raw"].astype(bool)
        final = export["predicted"].astype(bool)
        baseline_j = jaccard(base["predicted"].astype(bool))
        direct = raw.any(1)
        changes = jaccard(final) - baseline_j
        score = export["probabilities"]
        ranks = np.argsort(np.argsort(-score, axis=1, kind="stable"), axis=1)
        top5 = (y & (ranks < 5)).any(1)
        known_above_line = (raw & y).any(1)
        hit = (final & y).any(1)
        report = {"original_empty": {
            name: {"rows": int(mask.sum()),
                   "hits": int((mask & hit).sum()),
                   "delta_contribution_pp": float(100 * changes[mask].sum() / len(y))}
            for name, mask in (("became_nonempty", empty & direct), ("remained_empty", empty & ~direct))
        }}
        for name in ("nonempty_zero", "both_prefer_error_true_top5"):
            group, h = cohorts[name], helix_rescue[name]
            missed = h & ~hit
            report[name] = {"rows": int(group.sum()), "rescued": int((group & hit).sum()),
                            "rescued_with_helix": int((h & hit).sum()),
                            "rescued_without_helix": int((group & hit & ~h).sum()),
                            "helix_rescues_still_missed": int(missed.sum()),
                            "among_still_missed_outside_top5": int((missed & ~top5).sum()),
                            "among_still_missed_inside_top5_below_cutoff": int((missed & top5 & ~known_above_line).sum()),
                            "among_still_missed_wrong_raw_nonempty": int((missed & direct & ~known_above_line).sum())}
        if fixed:
            _, revised = apply_train_reference(data, val, raw, final, export["theraset_logits"])
            report["score_fixed_fallback"] = jaccard(revised)
        return report

    original = inspect(base, True)
    point = inspect(pointwise, True)
    rel = inspect(relative, False)
    old_fixed = original.pop("score_fixed_fallback")
    point_fixed = point.pop("score_fixed_fallback")
    delta = point_fixed - old_fixed
    return {"acc": {"original_standard": float(jaccard(base["predicted"].astype(bool)).mean()),
                    "original_fixed_fallback": float(old_fixed.mean()),
                    "pointwise_standard": float(jaccard(pointwise["predicted"].astype(bool)).mean()),
                    "pointwise_fixed_fallback": float(point_fixed.mean()),
                    "pointwise_vs_original_fixed_delta_pp": float(100 * delta.mean()),
                    "pointwise_vs_original_fixed_ci95_pp": paired_cluster_interval(delta, clusters),
                    "original_empty_rows": int(empty.sum()),
                    "pointwise_vs_original_fixed_original_empty_contribution_pp":
                        float(100 * delta[empty].sum() / len(y))},
            "pointwise": point, "relative": rel}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "pointwise", "relative", "helix"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    args = parser.parse_args()
    with np.load(args.base) as base, np.load(args.pointwise) as pointwise, \
            np.load(args.relative) as relative, np.load(args.helix) as helix, threadpool_limits(limits=4):
        print(json.dumps(audit(PeptideData(args.data_root), base, pointwise, relative, helix), indent=2))
