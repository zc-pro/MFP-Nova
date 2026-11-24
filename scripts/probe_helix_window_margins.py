"""Attribute frozen HelixSet motif ablation changes on fixed Validation labels."""

import argparse
import json
from pathlib import Path

import numpy as np


def audit(base, full, ablated, aa34):
    for export in (full, ablated, aa34):
        for key in ("row_id", "cluster_id", "targets", "label_names"):
            np.testing.assert_array_equal(base[key], export[key])
    y = base["targets"].astype(bool)
    old = base["predicted"].astype(bool)
    nonempty = base["predicted_raw"].any(1)
    limit = np.log(999999.)
    a, b = base["offsets"]
    t = base["theraset_logits"].clip(-limit, limit)
    c = base["aa34_logits"].clip(-limit, limit)
    score = (t - a + c - b) / 2
    offset = np.log(.64 / .36)
    complete = (2 * score + full["latent_logits"].clip(-limit, limit) - offset) / 3
    removed = (2 * score + ablated["latent_logits"].clip(-limit, limit) - offset) / 3

    def predict(logits):
        raw = logits >= 0
        return np.where((nonempty & raw.any(1))[:, None], raw, old)

    old_hit, full_hit, removed_hit = ((p & y).any(1) for p in (old, predict(complete), predict(removed)))
    rows = np.arange(len(y))
    old_correct = np.where(y, score, -np.inf).argmax(1)
    old_wrong = np.where(old & ~y, score, -np.inf).argmax(1)
    ranks = np.argsort(np.argsort(-score, axis=1, kind="stable"), axis=1)
    common = (nonempty & ~old_hit & (ranks[rows, old_correct] < 5)
              & (t[rows, old_wrong] > t[rows, old_correct])
              & (c[rows, old_wrong] > c[rows, old_correct]))
    aa_hit = (aa34["predicted"] & y).any(1)

    def summarize(mask, preservation=False):
        i = np.flatnonzero(mask)
        if not len(i):
            return {"rows": 0}
        # Freeze the labels across both forwards. For already-hit rows use the
        # strongest originally reported true label and highest unrecorded rival.
        good_source = score if preservation else complete
        good_mask = old & y if preservation else predict(complete) & y
        good = np.where(good_mask[i], good_source[i], -np.inf).argmax(1)
        bad = (np.where(~y[i], score[i], -np.inf).argmax(1) if preservation else old_wrong[i])
        d_good = complete[i, good] - removed[i, good]
        d_bad = complete[i, bad] - removed[i, bad]
        def outcomes(pred):
            pred = pred[i]
            hit = (pred & y[i]).any(1)
            jaccard = (pred & y[i]).sum(1) / (pred | y[i]).sum(1).clip(min=1)
            return {"hit_rows": int(hit.sum()), "mean_jaccard": float(jaccard.mean())}
        def replay(columns):
            copy = removed.copy()
            for col in columns:
                copy[i, col] = complete[i, col]
            return outcomes(predict(copy))
        return {
            "rows": len(i),
            "median_correct_logit_change": float(np.median(d_good)),
            "median_wrong_logit_change": float(np.median(d_bad)),
            "median_margin_change": float(np.median(d_good - d_bad)),
            "correct_crossed_zero_up": int(((removed[i, good] < 0) & (complete[i, good] >= 0)).sum()),
            "wrong_crossed_zero_down": int(((removed[i, bad] >= 0) & (complete[i, bad] < 0)).sum()),
            "replays": {"ablated": outcomes(predict(removed)), "full": outcomes(predict(complete)),
                        "restore_correct": replay((good,)), "restore_wrong": replay((bad,)),
                        "restore_both": replay((good, bad))},
        }

    lost = common & full_hit & ~removed_hit
    return {"common_errors": int(common.sum()), "full_rescues": int((common & full_hit).sum()),
            "retained": int((common & full_hit & removed_hit).sum()),
            "lost": summarize(lost), "lost_and_aa34_still_wrong": summarize(lost & ~aa_hit),
            "original_nonempty_already_hit": summarize(nonempty & old_hit, preservation=True)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "full", "ablated", "aa34"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    with np.load(args.base) as base, np.load(args.full) as full, \
            np.load(args.ablated) as ablated, np.load(args.aa34) as aa34:
        print(json.dumps(audit(base, full, ablated, aa34), indent=2))
