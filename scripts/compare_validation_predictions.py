"""Compare two saved fine39 Validation runs on the same rows and labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from mfp_nova.data import PeptideData
from mfp_nova.multimodal import function_projection, independent_functions


def read_validation(run, data, rows):
    path = run / "best_validation.npz"
    with np.load(path, allow_pickle=False) as saved:
        order = np.argsort(saved["row_id"])
        if (saved["split"].item() != "val" or saved["task"].item() != data.track
                or saved["label_names"].tolist() != data.labels
                or not np.array_equal(saved["row_id"][order], data.rows[rows])
                or not np.array_equal(saved["ID"][order], data.ids[rows])
                or not np.array_equal(saved["targets"][order], data.y[rows])):
            raise ValueError(f"{path} does not match the original {data.track} Validation rows and labels")
        probabilities = saved["probabilities"][order]
        threshold = saved["threshold"].item()
        if "prediction_rule" in saved and saved["prediction_rule"].item() == "joint_expected_jaccard":
            return probabilities, saved["predicted"][order]
    return probabilities, probabilities >= threshold


def percent_pair(baseline, candidate):
    return {"baseline_pct": round(100 * baseline, 3), "candidate_pct": round(100 * candidate, 3),
            "delta_pp": round(100 * (candidate - baseline), 3)}


def sample_jaccard(targets, prediction):
    intersection = (targets & prediction).sum(1)
    union = (targets | prediction).sum(1)
    return np.divide(intersection, union, out=np.zeros(len(targets)), where=union != 0)


def rank_diagnostics(targets, a_probs, a_pred, b_probs, b_pred, projection):
    a_score = sample_jaccard(targets, a_pred)
    b_score = sample_jaccard(targets, b_pred)
    singleton = (targets.sum(1) == 1) & (a_pred.sum(1) == 1) & (a_score == 0)
    true_axis = (targets @ np.asarray(projection)) > 0
    called_axis = (a_pred @ np.asarray(projection)) > 0
    same_axis = (true_axis & called_axis).any(1)
    slices = {"all": singleton, "same_axis": singleton & same_axis,
              "different_axis": singleton & ~same_axis}

    def prefix_hits(probabilities):
        order = np.argsort(-probabilities, axis=1, kind="stable")[:, :6]
        hits = np.take_along_axis(targets, order, axis=1).cumsum(1)
        sizes = np.arange(1, 7)
        prefix_jaccard = hits / (targets.sum(1)[:, None] + sizes - hits)
        return hits, prefix_jaccard.max(1)

    a_hits, a_best_prefix = prefix_hits(a_probs)
    b_hits, b_best_prefix = prefix_hits(b_probs)
    original = np.maximum(a_score, a_best_prefix)
    expanded = np.maximum.reduce([original, b_score, b_best_prefix])
    coverage = {}
    for name, mask in slices.items():
        counts = {"rows": int(mask.sum()), "candidate_exact_set": int(((b_score == 1) & mask).sum())}
        for k in (1, 3, 5):
            a_hit, b_hit = a_hits[:, k - 1] > 0, b_hits[:, k - 1] > 0
            counts[f"top_{k}"] = {
                "baseline_hits": int((mask & a_hit).sum()),
                "candidate_hits": int((mask & b_hit).sum()),
                "candidate_only_hits": int((mask & b_hit & ~a_hit).sum()),
                "baseline_only_hits": int((mask & a_hit & ~b_hit).sum()),
            }
        coverage[name] = counts
    return {
        "baseline_wrong_singletons": coverage,
        "top_1_annotated_hit": {"baseline": int((a_hits[:, 0] > 0).sum()),
                                "candidate": int((b_hits[:, 0] > 0).sum())},
        "top_5_at_least_two_annotated_hits": {"baseline": int((a_hits[:, 4] >= 2).sum()),
                                             "candidate": int((b_hits[:, 4] >= 2).sum())},
        "top_5_at_least_three_annotated_hits": {"baseline": int((a_hits[:, 4] >= 3).sum()),
                                               "candidate": int((b_hits[:, 4] >= 3).sum())},
        "exact_set_matches": {"baseline": int((a_score == 1).sum()), "candidate": int((b_score == 1).sum()),
                              "preserved": int(((a_score == 1) & (b_score == 1)).sum())},
        "baseline_zero_overlap_recovered": int(((a_score == 0) & (b_score > 0)).sum()),
        "baseline_positive_overlap_lost": int(((a_score > 0) & (b_score == 0)).sum()),
        "oracle_fixed_sets_and_top_1_to_6": percent_pair(original.mean(), expanded.mean()),
        "oracle_improved_rows": int((expanded > original).sum()),
    }


def compare(data, baseline_run, candidate_run):
    rows = data.indices("val")
    rows = rows[np.argsort(data.rows[rows])]
    targets = data.y[rows].astype(bool)
    a_probs, a_pred = read_validation(baseline_run, data, rows)
    b_probs, b_pred = read_validation(candidate_run, data, rows)
    a_score, b_score = sample_jaccard(targets, a_pred), sample_jaccard(targets, b_pred)
    length = np.fromiter(map(len, data.sequences[rows]), dtype=int, count=len(rows))
    counts = targets.sum(1)
    groups = {"length_up_to_8": length <= 8, "length_9_16": (length >= 9) & (length <= 16),
              "length_17_32": (length >= 17) & (length <= 32), "length_33_plus": length >= 33,
              "one_annotated_label": counts == 1, "two_annotated_labels": counts == 2,
              "three_or_more_annotated_labels": counts >= 3}
    by_group = {name: {"rows": int(mask.sum()), **percent_pair(a_score[mask].mean(), b_score[mask].mean())}
                for name, mask in groups.items() if mask.any()}
    per_label = []
    a_label_ap, b_label_ap = [], []
    label_support = targets.sum(0)
    for col in np.flatnonzero(label_support):
        a_ap = average_precision_score(targets[:, col], a_probs[:, col])
        b_ap = average_precision_score(targets[:, col], b_probs[:, col])
        a_label_ap.append(a_ap)
        b_label_ap.append(b_ap)
        per_label.append({"label": data.labels[col], "positives": int(label_support[col]), **percent_pair(a_ap, b_ap)})
    per_label.sort(key=lambda item: item["delta_pp"], reverse=True)
    projection = function_projection(data.labels, data.track)
    actual_mfp = independent_functions(targets, projection).sum(1) >= 2
    a_mfp = independent_functions(a_pred, projection).sum(1) >= 2
    b_mfp = independent_functions(b_pred, projection).sum(1) >= 2
    return {"baseline": str(baseline_run), "candidate": str(candidate_run), "rows": len(rows),
            "jaccard": percent_pair(a_score.mean(), b_score.mean()),
            "macro_ap": percent_pair(np.mean(a_label_ap), np.mean(b_label_ap)),
            "micro_f1": percent_pair(f1_score(targets, a_pred, average="micro", zero_division=0),
                                     f1_score(targets, b_pred, average="micro", zero_division=0)),
            "sample_jaccard_improved": int((b_score > a_score).sum()),
            "sample_jaccard_declined": int((b_score < a_score).sum()),
            "independent_multifunction_detection_f1": percent_pair(f1_score(actual_mfp, a_mfp, zero_division=0),
                                                                  f1_score(actual_mfp, b_mfp, zero_division=0)),
            "groups_jaccard": by_group,
            "candidate_ranking": rank_diagnostics(targets, a_probs, a_pred, b_probs, b_pred, projection),
            "label_ap_largest_gains": [row for row in per_label if row["delta_pp"] > 0][:5],
            "label_ap_largest_drops": [row for row in reversed(per_label) if row["delta_pp"] < 0][:5]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    args = parser.parse_args()
    print(json.dumps(compare(PeptideData(args.data_root, "fine39"), args.baseline, args.candidate), indent=2))


if __name__ == "__main__":
    main()
