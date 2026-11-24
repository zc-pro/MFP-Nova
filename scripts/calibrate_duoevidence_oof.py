"""Fit a low-capacity fusion on Train OOF scores and evaluate it on Validation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mfp_nova.data import PeptideData
from mfp_nova.models.trievidence import center_evidence, empty_top10
from mfp_nova.reference_fallback import paired_cluster_interval


def centered(export, offsets):
    logits = np.stack([export["theraset_logits"], export["aa34_logits"]])
    return center_evidence(torch.from_numpy(logits), torch.from_numpy(offsets))


def decide(scores, thera):
    return empty_top10(scores, torch.from_numpy(thera)).numpy()


def jaccard(y, p):
    return (y & p).sum(1) / (y | p).sum(1).clip(min=1)


def evaluate(train, val):
    offsets = val["offsets"]
    scores = centered(train, offsets)
    # Divide centered margins by a robust spread without changing their sign.
    quartiles = np.quantile(scores.numpy().reshape(2, -1), [.25, .75], axis=1)
    spread = np.maximum(quartiles[1] - quartiles[0], 1e-6).astype(np.float32)
    scaled_train = scores / torch.from_numpy(spread)[:, None, None]
    y_train = train["targets"].astype(bool)
    def fuse(values, weight):
        return (1 - weight) * values[0] + weight * values[1]
    weights = (.5, .375, .625, .25, .75)
    weight = max(weights, key=lambda w: jaccard(y_train, decide(fuse(scaled_train, w), train["theraset_logits"])).mean())

    y, old = val["targets"].astype(bool), val["predicted"].astype(bool)
    test_scores = centered(val, offsets)
    scaled_val = test_scores / torch.from_numpy(spread)[:, None, None]
    thera = val["theraset_logits"]
    np.testing.assert_array_equal(decide(test_scores.mean(0), thera), old)
    old_score, old_hit = jaccard(y, old), (y & old).any(1)
    report = {"baseline_acc": float(old_score.mean()), "train_oof_rows": len(y_train),
              "margin_iqr": spread.tolist(), "fitted_aa34_weight": weight, "arms": {}}
    for name, logits in (("scaled_equal", scaled_val.mean(0)), ("scaled_weighted", fuse(scaled_val, weight))):
        pred = decide(logits, thera)
        score, hit = jaccard(y, pred), (y & pred).any(1)
        delta = score - old_score
        report["arms"][name] = {"acc": float(score.mean()), "delta_pp": float(100 * delta.mean()),
                                "ci95_pp": paired_cluster_interval(delta, val["cluster_id"]),
                                "rescued": int((~old_hit & hit).sum()), "new_zero": int((old_hit & ~hit).sum())}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_oof", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    data = PeptideData(args.data_root)
    with np.load(args.train_oof) as train, np.load(args.validation) as val:
        for export, split, scope in ((train, "train", "train_oof"), (val, "val", "val")):
            if str(export["split"]) != scope:
                raise ValueError(f"Expected {scope} predictions")
            idx = data.indices(split)
            for name, expected in (("row_id", data.rows[idx]), ("cluster_id", data.clusters[idx]),
                                   ("targets", data.y[idx]), ("label_names", data.labels)):
                np.testing.assert_array_equal(export[name], expected)
        print(json.dumps(evaluate(train, val), indent=2))
