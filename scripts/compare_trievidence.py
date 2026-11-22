"""Compare two TriEvidence evaluations on identical Fine39 rows."""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_fine39_fixed_fusion import summarize


def fallback_diagnostics(a, b, base, other, y, strata):
    """Cross the empty-row mask and candidate ranking without fitting a rule."""
    empty, chosen, raw, final = [], [], [], []
    for z, report in ((a, base), (b, other)):
        raw.append(z["predicted_raw"])
        final.append(z["predicted"])
        empty.append(~raw[-1].any(1))
        keys = ("theraset_logits", "aa34_logits", "latent_logits")
        t = np.asarray(report["branch_thresholds"])
        offset = np.log(t / (1 - t)).astype(np.float32)
        limit = np.log((1 - 1e-6) / 1e-6)
        fused = (np.clip(np.stack([z[k] for k in keys]), -limit, limit) -
                 offset[:, None, None]).mean(0)
        shortlist = np.argsort(-z[keys[0]], axis=1, kind="stable")[:, :10]
        scores = np.take_along_axis(fused, shortlist, axis=1)
        priority = shortlist + y.shape[1] * (np.arange(10) >= 5)
        label = np.where(scores == scores.max(1, keepdims=True), priority,
                         2 * y.shape[1]).min(1) % y.shape[1]
        predicted = np.zeros_like(y)
        predicted[np.arange(len(y)), label] = True
        np.testing.assert_array_equal(predicted[empty[-1]], final[-1][empty[-1]])
        chosen.append(predicted)

    def jac(pred):
        return (y & pred).sum(1) / np.maximum(1, (y | pred).sum(1))

    raw_score = list(map(jac, raw))
    final_delta = jac(final[1]) - jac(final[0])
    rankings = []
    for z in (a, b):
        rankings.append(np.argsort(-z["probabilities"], axis=1, kind="stable"))
    counts = [p.sum(1) for p in raw]
    positions = np.arange(y.shape[1])
    crossed = {}
    for index, name in enumerate(("baseline", "candidate")):
        predicted = np.zeros_like(y)
        np.put_along_axis(predicted, rankings[index], positions < counts[1 - index][:, None], axis=1)
        crossed[name + "_rank_other_count"] = jac(predicted)
    added, removed = raw[1] & ~raw[0], raw[0] & ~raw[1]
    groups = {"both_empty": empty[0] & empty[1], "baseline_only_empty": empty[0] & ~empty[1],
              "candidate_only_empty": ~empty[0] & empty[1], "neither_empty": ~empty[0] & ~empty[1]}
    report = {}
    for name, subset in ({"all": np.ones(len(y), bool)} | strata).items():
        if not subset.any():
            continue
        report[name] = {
            "groups": {key: {"rows": int((mask & subset).sum()),
                             "final_delta_pp_contribution": float(100 * (final_delta * mask)[subset].mean())}
                       for key, mask in groups.items()},
            "fallback_gain_pp_by_empty_mask_then_ranking": {
                mask_name: {rank_name: float(100 * (mask * jac(ranking))[subset].mean())
                            for rank_name, ranking in zip(("baseline", "candidate"), chosen)}
                for mask_name, mask in zip(("baseline", "candidate"), empty)},
            "raw_delta_pp": float(100 * (raw_score[1] - raw_score[0])[subset].mean()),
        }
        paired = subset & groups["neither_empty"]
        if paired.any():
            report[name]["nonempty_in_both"] = {
                "rows": int(paired.sum()),
                "changed_rows": int(((added | removed).any(1) & paired).sum()),
                "added_annotated": int((added[paired] & y[paired]).sum()),
                "removed_annotated": int((removed[paired] & y[paired]).sum()),
                "added_unannotated": int((added[paired] & ~y[paired]).sum()),
                "removed_unannotated": int((removed[paired] & ~y[paired]).sum()),
                "acc_pct": {key: float(100 * score[paired].mean()) for key, score in (
                    {"baseline": raw_score[0], "candidate": raw_score[1]} | crossed).items()},
            }
    return report


def compare(baseline, candidate, nearest=None, structure=None, diagnose_fallback=False):
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
        fusion_delta = (float(np.abs(a["fusion_logits"] - b["fusion_logits"]).max())
                        if "fusion_logits" in a and "fusion_logits" in b else None)
        fallback_top10_changed = None
        if "theraset_logits" in a and "theraset_logits" in b:
            empty = ~a["predicted_raw"].any(1) | ~b["predicted_raw"].any(1)
            first, second = (np.argsort(-z["theraset_logits"][empty], axis=1, kind="stable")[:, :10]
                             for z in (a, b))
            fallback_top10_changed = int(np.any(first != second, axis=1).sum())
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
        complement = {}
        if (all("branch_thresholds" in report for report in (base, other))
                and all(key in z for z in (a, b) for key in ("theraset_logits", "aa34_logits", "latent_logits"))):
            limit = np.log((1 - 1e-6) / 1e-6)
            for name, z, report in (("baseline", a, base), ("candidate", b, other)):
                thresholds = np.asarray(report["branch_thresholds"])
                offsets = np.log(thresholds / (1 - thresholds)).astype(np.float32)
                branches = np.clip(np.stack([z[k] for k in ("theraset_logits", "aa34_logits", "latent_logits")]),
                                   -limit, limit) - offsets[:, None, None]
                full = z["predicted_raw"]
                full_hit = (full & y).any(1)
                full_score = (full & y).sum(1) / np.maximum((full | y).sum(1), 1)
                pair_results = {}
                for branch, pair in (("aa34", (0, 2)), ("latent", (0, 1))):
                    pred = branches[list(pair)].mean(0) >= 0
                    hit = (pred & y).any(1)
                    score = (pred & y).sum(1) / np.maximum((pred | y).sum(1), 1)
                    groups = {"all": np.ones(len(y), bool)}
                    if "nearest_below_070" in strata:
                        groups["nearest_below_070"] = strata["nearest_below_070"]
                    pair_results[branch] = {
                        key: {"rows": int(mask.sum()), "adding_delta_pp": 100 * float((full_score - score)[mask].mean()),
                              "rescued": int((mask & ~hit & full_hit).sum()),
                              "lost": int((mask & hit & ~full_hit).sum())}
                        for key, mask in groups.items() if mask.any()
                    }
                    if branch == "aa34":
                        pair_results["aa34_rescue_mask"] = ~hit & full_hit
                complement[name] = pair_results
            rescues = [complement[name].pop("aa34_rescue_mask") for name in ("baseline", "candidate")]
            complement["aa34_rescue_overlap"] = {
                "retained": int((rescues[0] & rescues[1]).sum()),
                "baseline_only": int((rescues[0] & ~rescues[1]).sum()),
                "candidate_only": int((~rescues[0] & rescues[1]).sum()),
            }
        fallback = fallback_diagnostics(a, b, base, other, y, strata) if diagnose_fallback else {}
    return {"split": split, "rows": len(y), "metrics": metrics,
            "fallback_diagnostics": fallback,
            "branch_logits_max_abs_delta": logits_delta,
            "fusion_logits_max_abs_delta": fusion_delta,
            "fallback_top10_changed_rows": fallback_top10_changed,
            "branch_complementarity": complement,
            "both_acc_at_least_baseline": all(v["delta_pp"] >= 0 for v in metrics.values()),
            "parameters": [r.get("parameters") for r in (base, other)],
            "cnn_parameters": [r.get("cnn_parameters") for r in (base, other)],
            "inference": [r.get("inference") for r in (base, other)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--nearest", type=Path, help="Fine39 nearest-Train Validation export")
    parser.add_argument("--structure", type=Path, help="Dataset structure_primary.npz")
    parser.add_argument("--diagnose-fallback", action="store_true",
                        help="Cross empty-row masks and fallback rankings for original TriEvidence")
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate, args.nearest, args.structure,
                             args.diagnose_fallback), indent=2))
