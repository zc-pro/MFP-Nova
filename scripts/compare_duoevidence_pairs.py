"""Compare three two-branch combinations without calling a third branch."""

import argparse
import io
import itertools
import json
import tarfile
from pathlib import Path

import numpy as np

from mfp_nova.data import PeptideData
from mfp_nova.reference_fallback import apply_train_reference, paired_cluster_interval
from probe_duoevidence_helix import compare as three_branch_probe


def align(base, helix):
    order = np.argsort(helix["row_id"])
    idx = order[np.searchsorted(helix["row_id"][order], base["row_id"])]
    for key in ("row_id", "targets", "cluster_id"):
        np.testing.assert_array_equal(base[key], helix[key][idx])
    np.testing.assert_array_equal(base["label_names"], helix["label_names"])
    if any(str(export["split"]) != "val" for export in (base, helix)):
        raise ValueError("Use the aligned Fine39 Validation exports")
    return idx


def compare(base, helix, data=None, full_helix=None):
    idx = align(base, helix)
    y = base["targets"].astype(bool)
    rows = np.arange(len(y))
    old = base["predicted"].astype(bool)
    clusters = base["cluster_id"]
    limit = np.log(999999.)
    a, b = base["offsets"]
    logits = {"theraset": base["theraset_logits"].clip(-limit, limit),
              "aa34": base["aa34_logits"].clip(-limit, limit),
              "helix": helix["latent_logits"][idx].clip(-limit, limit)}
    offsets = {"theraset": a, "aa34": b, "helix": np.log(.64/.36)}
    reference_zero = ~(old & y).any(1)
    three_branch_rescues, cohorts = {}, {}
    three_branch_probe(base, helix, masks=three_branch_rescues, cohorts=cohorts)
    def jaccard(p):
        return (p & y).sum(1) / (p | y).sum(1).clip(min=1)

    results, outputs = {}, {}
    fixed, raw_outputs = {}, {}
    if data is not None:
        val = data.indices("val")
        for key, expected in (("row_id", data.rows[val]), ("cluster_id", data.clusters[val]),
                              ("targets", data.y[val]), ("label_names", data.labels)):
            np.testing.assert_array_equal(base[key], expected)
    for left, right in itertools.combinations(logits, 2):
        key = f"{left}_{right}"
        score = ((logits[left] - offsets[left]) + (logits[right] - offsets[right])) / 2
        raw = score >= 0
        if key == "theraset_aa34":
            np.testing.assert_array_equal(raw, base["predicted_raw"])
        empty = ~raw.any(1)
        proposed = raw.copy()
        proposed[empty, score[empty].argmax(1)] = True
        delta = jaccard(proposed) - jaccard(old)
        hit = (proposed & y).any(1)
        lhit, rhit = (y[rows, logits[name].argmax(1)] for name in (left, right))
        results[key] = {
            "raw_acc": float(jaccard(raw).mean()),
            "fused_top1_fallback_acc": float(jaccard(proposed).mean()),
            "delta_vs_original_protocol_pp": float(100 * delta.mean()),
            "empty_before_fallback": int(empty.sum()),
            "rescued_original_nonempty_zero": int((reference_zero & base["predicted_raw"].any(1) & hit).sum()),
            "new_zero_vs_original": int((~reference_zero & ~hit).sum()),
            "zero_hit": int((~hit).sum()),
            "left_only_top1": int((lhit & ~rhit).sum()),
            "right_only_top1": int((rhit & ~lhit).sum()),
        }
        results[key]["fixed_original_cohorts"] = {}
        for name, mask in cohorts.items():
            entry = {"rows": int(mask.sum()), "rescued": int((mask & reference_zero & hit).sum()),
                     "new_zero": int((mask & ~reference_zero & ~hit).sum()),
                     "delta_contribution_pp": float(100 * delta[mask].sum() / len(y))}
            if name in three_branch_rescues:
                reference = three_branch_rescues[name]
                entry.update(shared_rescues=int((reference & hit).sum()),
                             lost_three_branch_rescues=int((reference & ~hit).sum()),
                             additional_rescues=int((mask & ~reference & hit).sum()))
            results[key]["fixed_original_cohorts"][name] = entry
        outputs[key] = jaccard(proposed)
        if data is not None and left == "theraset":
            # Apply the same frozen Train reference on each pair's own empty rows.
            initial = old if right == "aa34" else proposed
            _, fixed[key] = apply_train_reference(data, val, raw, initial, base["theraset_logits"])
            raw_outputs[key] = raw
    same_rule = outputs["theraset_aa34"]
    for key, result in results.items():
        d = outputs[key] - same_rule
        result.update(delta_vs_theraset_aa34_same_rule_pp=float(100 * d.mean()),
                      ci95_vs_theraset_aa34_same_rule_pp=paired_cluster_interval(d, clusters))
    report = {"rows": len(y), "original_protocol_acc": float(jaccard(old).mean()), "pairs": results}
    if fixed:
        control, candidate = (fixed[key] for key in ("theraset_aa34", "theraset_helix"))
        delta = jaccard(candidate) - jaccard(control)
        control_hit, candidate_hit = ((p & y).any(1) for p in (control, candidate))
        old_empty = ~raw_outputs["theraset_aa34"].any(1)
        new_empty = ~raw_outputs["theraset_helix"].any(1)
        groups = {"original_empty_candidate_direct": old_empty & ~new_empty,
                  "both_need_fallback": old_empty & new_empty,
                  "original_direct_candidate_empty": ~old_empty & new_empty,
                  "both_direct": ~old_empty & ~new_empty}
        report["fixed_reference_comparison"] = {
            "theraset_aa34_acc": float(jaccard(control).mean()),
            "theraset_helix_acc": float(jaccard(candidate).mean()),
            "delta_pp": float(100 * delta.mean()),
            "ci95_pp": paired_cluster_interval(delta, clusters),
            "groups": {name: {"rows": int(mask.sum()),
                              "delta_contribution_pp": float(100 * delta[mask].sum() / len(y)),
                              "rescued": int((mask & ~control_hit & candidate_hit).sum()),
                              "new_zero": int((mask & control_hit & ~candidate_hit).sum())}
                       for name, mask in groups.items()},
        }
        if full_helix is not None:
            full_idx = align(base, full_helix)
            h = full_helix["latent_logits"][full_idx].clip(-limit, limit)
            full_score = ((logits["theraset"] - a) + (h - offsets["helix"])) / 2
            full_raw = full_score >= 0
            full_pred = full_raw.copy()
            empty = ~full_raw.any(1)
            full_pred[empty, full_score[empty].argmax(1)] = True
            _, full_pred = apply_train_reference(data, val, full_raw, full_pred, base["theraset_logits"])
            full_hit = (full_pred & y).any(1)
            full_score_j = jaccard(full_pred)
            delta_full = jaccard(candidate) - full_score_j
            report["compressed_vs_full_fixed"] = {
                "full_acc": float(full_score_j.mean()),
                "candidate_acc": float(jaccard(candidate).mean()),
                "delta_pp": float(100 * delta_full.mean()),
                "ci95_pp": paired_cluster_interval(delta_full, clusters),
                "groups": {},
            }
            for name, group in cohorts.items():
                eligible = group & reference_zero
                report["compressed_vs_full_fixed"]["groups"][name] = {
                    "rows": int(group.sum()),
                    "full_rescued": int((eligible & full_hit).sum()),
                    "candidate_rescued": int((eligible & candidate_hit).sum()),
                    "retained": int((eligible & full_hit & candidate_hit).sum()),
                    "lost": int((eligible & full_hit & ~candidate_hit).sum()),
                    "gained": int((eligible & ~full_hit & candidate_hit).sum()),
                    "delta_contribution_pp": float(100 * delta_full[group].sum() / len(y)),
                }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("helix", type=Path)
    parser.add_argument("--member", help="Base prediction member in a tar archive")
    parser.add_argument("--fixed-reference", action="store_true", help="Use the fixed Train fallback on both TheraSet pairs")
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--reference-helix", type=Path, help="Compare against full HelixSet under fixed fallback")
    args = parser.parse_args()
    if args.reference_helix and not args.fixed_reference:
        parser.error("--reference-helix requires --fixed-reference")
    source = args.base
    if args.member:
        with tarfile.open(source) as archive:
            source = io.BytesIO(archive.extractfile(args.member).read())
    with np.load(source) as base, np.load(args.helix) as helix:
        data = PeptideData(args.data_root) if args.fixed_reference else None
        if args.reference_helix:
            with np.load(args.reference_helix) as full:
                report = compare(base, helix, data, full)
        else:
            report = compare(base, helix, data)
        print(json.dumps(report, indent=2))
