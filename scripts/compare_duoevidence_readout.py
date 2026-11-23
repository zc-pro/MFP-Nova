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


def screen_additions(reference, candidate, seed=42):
    with np.load(reference) as before, np.load(candidate) as after:
        targets = before["targets"].astype(bool)
        base = before["predicted"].astype(bool)
        added = after["predicted_raw"] & ~base & before["predicted_raw"].any(1, keepdims=True)
        def logits(export):
            branches = np.stack([export[key] for key in ("theraset_logits", "aa34_logits")])
            limit = np.log(999999.)
            return (branches.clip(-limit, limit) - export["offsets"][:, None, None]).mean(0)
        original, revised = map(logits, (before, after))
        _, clusters = np.unique(before["cluster_id"], return_inverse=True)
    fold = np.random.default_rng(seed).permutation(clusters.max() + 1)[clusters] % 5
    hits, union = (base & targets).sum(1), (base | targets).sum(1)
    settings = [(0., 0.)] + [(scale, margin) for scale in (.25, .5, .75, 1.)
                              for margin in (0., .05, .1, .2, .4)]
    masks = [added & (original + scale * (revised - original) >= margin)
             if scale else np.zeros_like(added) for scale, margin in settings]
    baseline = hits / np.maximum(1, union)
    def jaccard(extra):
        return (hits + (extra & targets).sum(1)) / np.maximum(1, union + (extra & ~targets).sum(1))
    values = np.stack([jaccard(extra) for extra in masks])
    held = np.zeros_like(added)
    selected = []
    for k in range(5):
        test = fold == k
        winner = values[:, ~test].mean(1).argmax()
        held[test] = masks[winner][test]
        selected.append(settings[winner])
    def summarize(extra):
        delta = jaccard(extra) - baseline
        return {"acc": float((baseline + delta).mean()), "delta_pp": float(100 * delta.mean()),
                "zero_rescued": int(((hits == 0) & (delta > 0)).sum()),
                "added_observed": int((extra & targets).sum()),
                "added_unrecorded": int((extra & ~targets).sum())}
    return {"baseline_acc": float(baseline.mean()), "unfiltered": summarize(added),
            "cluster_crossfit": summarize(held), "fold_scale_margin": selected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--screen-additions", action="store_true")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    if args.screen_additions:
        result["addition_screen"] = screen_additions(args.reference, args.candidate)
    print(json.dumps(result, indent=2))
