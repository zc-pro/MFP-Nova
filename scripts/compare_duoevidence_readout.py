"""Explain per-row Jaccard changes between aligned DuoEvidence Validation exports."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference, candidate):
    with np.load(reference) as before, np.load(candidate) as after:
        if any(str(export["split"]) != "val" for export in (before, after)):
            raise ValueError("Use Validation predictions for both readouts")
        for key in ("row_id", "cluster_id", "label_names", "targets"):
            np.testing.assert_array_equal(before[key], after[key])
        targets = before["targets"].astype(bool)
        raw = [export["predicted_raw"].astype(bool) for export in (before, after)]
        final = [export["predicted"].astype(bool) for export in (before, after)]

    def sample_jaccard(pred):
        return (pred & targets).sum(1) / np.maximum((pred | targets).sum(1), 1)

    start, end = map(sample_jaccard, final)
    delta = end - start
    old_empty, new_empty = (~p.any(1) for p in raw)
    added = final[1] & ~final[0]
    removed = final[0] & ~final[1]

    def summarize(mask):
        return {
            "rows": int(mask.sum()),
            "acc_contribution_pp": float(100 * delta[mask].sum() / len(delta)),
            "better": int((mask & (delta > 0)).sum()),
            "worse": int((mask & (delta < 0)).sum()),
            "zero_rescued": int((mask & (start == 0) & (end > 0)).sum()),
            "zero_created": int((mask & (start > 0) & (end == 0)).sum()),
            "added_observed": int((added & targets)[mask].sum()),
            "added_unrecorded": int((added & ~targets)[mask].sum()),
            "removed_observed": int((removed & targets)[mask].sum()),
            "removed_unrecorded": int((removed & ~targets)[mask].sum()),
        }

    return {
        "rows": len(delta), "reference_acc": float(start.mean()), "candidate_acc": float(end.mean()),
        "delta_pp": float(100 * delta.mean()),
        "raw_empty_transitions": {
            "empty_to_empty": summarize(old_empty & new_empty),
            "empty_to_nonempty": summarize(old_empty & ~new_empty),
            "nonempty_to_empty": summarize(~old_empty & new_empty),
            "nonempty_to_nonempty": summarize(~old_empty & ~new_empty),
        },
        "reference_nonempty_zero_hit": summarize(~old_empty & (start == 0)),
        "reference_nonempty_already_hit": summarize(~old_empty & (start > 0)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.reference, args.candidate), indent=2))
