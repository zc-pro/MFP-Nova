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


def context_diagnostics(reference, candidate, nearest=None):
    with np.load(reference) as before, np.load(candidate) as after:
        y = before["targets"].astype(bool)
        rows = before["row_id"]
        raw = [z["predicted_raw"].astype(bool) for z in (before, after)]
        final = [z["predicted"].astype(bool) for z in (before, after)]
        branches = [(z["theraset_logits"], z["aa34_logits"]) for z in (before, after)]
        thresholds = [z["offsets"].tolist() for z in (before, after)]
        _, cluster, sizes = np.unique(before["cluster_id"], return_inverse=True, return_counts=True)
    original_hit = (y & final[0]).any(1)
    hard = raw[0].any(1) & ~original_hit
    strata = {"all": np.ones(len(y), dtype=bool), "original_nonempty_zero": hard,
              "original_nonempty_hit": raw[0].any(1) & original_hit,
              "original_empty": ~raw[0].any(1), "multilabel_2": y.sum(1) == 2,
              "multilabel_3plus": y.sum(1) >= 3}
    if nearest is not None:
        with np.load(nearest) as z:
            order = np.argsort(z["row_id"])
            np.testing.assert_array_equal(rows, z["row_id"][order])
            strata["similarity_lt70"] = z["similarity"][order] < .7

    def jaccard(pred):
        return (pred & y).sum(1) / np.maximum(1, (pred | y).sum(1))

    delta = jaccard(final[1]) - jaccard(final[0])
    totals = np.bincount(cluster, weights=delta)
    rng = np.random.default_rng(42)
    bootstrap = []
    for _ in range(2000):
        sample = rng.integers(len(sizes), size=len(sizes))
        bootstrap.append(100 * totals[sample].sum() / sizes[sample].sum())

    def metrics(index, mask):
        thera, reader = branches[index]
        main_order = np.argsort(-thera, axis=1, kind="stable")
        read_order = np.argsort(-reader, axis=1, kind="stable")
        main_hit = np.take_along_axis(y, main_order, axis=1)
        read_hit = np.take_along_axis(y, read_order, axis=1)
        p = final[index]
        hits = (p & y).sum(1)
        return {
            "raw_acc": float(jaccard(raw[index])[mask].mean()),
            "final_acc": float(jaccard(p)[mask].mean()),
            "empty_before_fallback": int((mask & ~raw[index].any(1)).sum()),
            "zero_hit": int((mask & (hits == 0)).sum()),
            "mean_labels": float(p[mask].sum(1).mean()),
            "sample_precision": float((hits / np.maximum(1, p.sum(1)))[mask].mean()),
            "sample_recall": float((hits / np.maximum(1, y.sum(1)))[mask].mean()),
            "reader_top_hits": {k: int((mask & read_hit[:, :k].any(1)).sum()) for k in (1, 3, 5, 10)},
            "reader_only_top_hits": {k: int((mask & read_hit[:, :k].any(1) & ~main_hit[:, :k].any(1)).sum())
                                     for k in (1, 3, 5, 10)},
            "branches_top1_agree": int((mask & (main_order[:, 0] == read_order[:, 0])).sum()),
        }

    return {
        "paired_cluster_ci95_pp": np.quantile(bootstrap, [.025, .975]).tolist(),
        "theraset_logits_max_abs_delta": float(np.abs(branches[1][0] - branches[0][0]).max()),
        "thresholds": [[float(1 / (1 + np.exp(-value))) for value in offsets] for offsets in thresholds],
        "groups": {key: {"rows": int(mask.sum()), "reference": metrics(0, mask), "candidate": metrics(1, mask)}
                   for key, mask in strata.items() if mask.any()},
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


def two_label_diagnostics(reference, candidate):
    with np.load(reference) as before, np.load(candidate) as after:
        mask = before["targets"].sum(1) == 2
        truth = before["targets"][mask].astype(bool)
        def read(export):
            pred = export["predicted"][mask].astype(bool)
            scores = export["probabilities"][mask]
            order = np.argsort(-scores, axis=1, kind="stable")
            ranks = np.argsort(order, axis=1) + 1
            return pred, ranks
        old, old_ranks = read(before)
        new, new_ranks = read(after)

    lost = truth & old & ~new
    gained = truth & new & ~old
    added = ~truth & new & ~old
    old_hits = (old & truth).sum(1)
    new_hits = (new & truth).sum(1)
    delta = new_hits / (2 + (new & ~truth).sum(1)) - old_hits / (2 + (old & ~truth).sum(1))
    groups = {
        "fewer_hits": new_hits < old_hits,
        "more_hits": new_hits > old_hits,
        "same_hits_more_unrecorded": (new_hits == old_hits) & ((new & ~truth).sum(1) > (old & ~truth).sum(1)),
        "same_hits_fewer_unrecorded": (new_hits == old_hits) & ((new & ~truth).sum(1) < (old & ~truth).sum(1)),
    }
    return {
        "rows": int(mask.sum()),
        "hit_transitions_0_1_2": np.bincount(3 * old_hits + new_hits, minlength=9).reshape(3, 3).tolist(),
        "lost_observed": int(lost.sum()),
        "gained_observed": int(gained.sum()),
        "added_unrecorded": int(added.sum()),
        "lost_observed_candidate_rank_1_5_6_10_11plus": [
            int((lost & (new_ranks >= lo) & (new_ranks <= hi)).sum())
            for lo, hi in ((1, 5), (6, 10), (11, truth.shape[1]))
        ],
        "lost_observed_crossed_out_of_top5": int((lost & (old_ranks <= 5) & (new_ranks > 5)).sum()),
        "lost_observed_crossed_out_of_top10": int((lost & (old_ranks <= 10) & (new_ranks > 10)).sum()),
        "one_hit_other_observed_in_top5": [
            int(((hits == 1) & (truth & ~pred & (ranks <= 5)).any(1)).sum())
            for hits, pred, ranks in ((old_hits, old, old_ranks), (new_hits, new, new_ranks))
        ],
        "groups": {name: {"rows": int(group.sum()), "two_label_delta_contribution_pp":
                         float(100 * delta[group].sum() / max(1, len(delta)))}
                   for name, group in groups.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--screen-additions", action="store_true")
    parser.add_argument("--context-diagnostics", action="store_true")
    parser.add_argument("--two-label-diagnostics", action="store_true")
    parser.add_argument("--nearest", type=Path, help="Optional nearest-Train similarities for context diagnostics")
    args = parser.parse_args()
    result = compare(args.reference, args.candidate)
    if args.two_label_diagnostics:
        result["two_label_diagnostics"] = two_label_diagnostics(args.reference, args.candidate)
    if args.screen_additions:
        result["addition_screen"] = screen_additions(args.reference, args.candidate)
    if args.context_diagnostics:
        result["context_diagnostics"] = context_diagnostics(args.reference, args.candidate, args.nearest)
    print(json.dumps(result, indent=2))
