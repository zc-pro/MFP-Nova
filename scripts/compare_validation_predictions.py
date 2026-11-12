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
                or not np.array_equal(saved["targets"][order], data.y[rows])):
            raise ValueError(f"{path} does not match the original {data.track} Validation rows and labels")
        probabilities = saved["probabilities"][order]
        threshold = saved["threshold"].item()
    return probabilities, probabilities >= threshold


def percent_pair(baseline, candidate):
    return {"baseline_pct": round(100 * baseline, 3), "candidate_pct": round(100 * candidate, 3),
            "delta_pp": round(100 * (candidate - baseline), 3)}


def sample_jaccard(targets, prediction):
    intersection = (targets & prediction).sum(1)
    union = (targets | prediction).sum(1)
    return np.divide(intersection, union, out=np.zeros(len(targets)), where=union != 0)


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
    label_support = targets.sum(0)
    for col in np.flatnonzero(label_support):
        a_ap = average_precision_score(targets[:, col], a_probs[:, col])
        b_ap = average_precision_score(targets[:, col], b_probs[:, col])
        per_label.append({"label": data.labels[col], "positives": int(label_support[col]), **percent_pair(a_ap, b_ap)})
    per_label.sort(key=lambda item: item["delta_pp"], reverse=True)
    projection = function_projection(data.labels, data.track)
    actual_mfp = independent_functions(targets, projection).sum(1) >= 2
    a_mfp = independent_functions(a_pred, projection).sum(1) >= 2
    b_mfp = independent_functions(b_pred, projection).sum(1) >= 2
    return {"baseline": str(baseline_run), "candidate": str(candidate_run), "rows": len(rows),
            "jaccard": percent_pair(a_score.mean(), b_score.mean()),
            "sample_jaccard_improved": int((b_score > a_score).sum()),
            "sample_jaccard_declined": int((b_score < a_score).sum()),
            "independent_multifunction_detection_f1": percent_pair(f1_score(actual_mfp, a_mfp, zero_division=0),
                                                                  f1_score(actual_mfp, b_mfp, zero_division=0)),
            "groups_jaccard": by_group,
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
