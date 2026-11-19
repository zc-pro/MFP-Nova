"""Evaluate the fixed Fine39 ensemble and its two empty-output rules."""

import argparse
import io
import json
import tarfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def from_zip(path, suffix):
    with zipfile.ZipFile(path) as archive:
        member, = [name for name in archive.namelist() if "fine39" in name and name.endswith(suffix)]
        with np.load(io.BytesIO(archive.read(member))) as data:
            return dict(data)


def read_theraset(path, split):
    if split == "val" and path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            with np.load(io.BytesIO(archive.extractfile("baseline/best_validation.npz").read())) as data:
                return dict(data)
    with np.load(path) as data:
        return dict(data)


def align(data: Mapping, reference: Mapping):
    order = np.argsort(data["row_id"])
    matches = np.searchsorted(data["row_id"][order], reference["row_id"])
    if len(order) != len(matches) or (matches >= len(order)).any():
        raise ValueError("Prediction exports cover different rows")
    index = order[matches]
    for key in ("row_id", "targets"):
        np.testing.assert_array_equal(data[key][index], reference[key])
    np.testing.assert_array_equal(data["ID"][index].astype(str), reference["ID"].astype(str))
    np.testing.assert_array_equal(data["label_names"], reference["label_names"])
    return data["probabilities"][index]


def logit(value):
    value = np.clip(np.asarray(value, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(value) - np.log1p(-value)


def summarize(y, predicted, baseline, clusters, missed, fallback_rows):
    tp = (y & predicted).sum(1)
    n_pred = predicted.sum(1)
    n_true = y.sum(1)
    score = tp / (n_pred + n_true - tp)
    delta = score - baseline
    _, group = np.unique(clusters, return_inverse=True)
    counts = np.bincount(group)
    sums = np.bincount(group, weights=delta)
    rng = np.random.default_rng(42)
    draws = []
    for _ in range(2000):
        sample = rng.integers(len(counts), size=len(counts))
        draws.append(100 * sums[sample].sum() / counts[sample].sum())
    precision = np.divide(tp, n_pred, out=np.zeros(len(y)), where=n_pred > 0)
    return {
        "acc_pct": float(100 * score.mean()),
        "delta_vs_theraset_pp": float(100 * delta.mean()),
        "paired_cluster_ci95_pp": np.quantile(draws, [.025, .975]).tolist(),
        "sample_precision_pct": float(100 * precision.mean()),
        "sample_recall_pct": float(100 * (tp / n_true).mean()),
        "zero_hit_rows": int((tp == 0).sum()),
        "output_empty_rows": int((n_pred == 0).sum()),
        "hits_in_theraset_top5_misses": int((missed & (tp > 0)).sum()),
        "fallback_additions": int(fallback_rows.sum()),
        "fallback_observed_hits": int((fallback_rows & (tp > 0)).sum()),
    }


def evaluate(theraset, cnn, helix, split):
    for data, cutoff in ((theraset, .5), (cnn, .28), (helix, .64)):
        if float(data["threshold"]) != cutoff:
            raise ValueError(f"Expected historical threshold {cutoff}")
    if str(theraset["split"]) != split or str(helix["split"]) != split:
        raise ValueError(f"Expected {split} predictions")
    a = align(theraset, cnn)
    b = cnn["probabilities"]
    c = align(helix, cnn)
    if len(a) != {"val": 8529, "test": 8180}[split] or len(np.unique(cnn["row_id"])) != len(a):
        raise ValueError("Expected the complete original Fine39 split with unique row IDs")
    y = cnn["targets"].astype(bool)
    order = np.argsort(-a, axis=1, kind="stable")
    rank = np.argsort(order, axis=1) + 1
    missed = ~(y & (rank <= 5)).any(1)
    thera_pred = a >= .5
    baseline = (y & thera_pred).sum(1) / (y | thera_pred).sum(1)
    mix = ((logit(a) - logit(.5)) + (logit(b) - logit(.28)) + (logit(c) - logit(.64))) / 3
    equal_pred = mix >= 0
    empty = ~equal_pred.any(1)
    output = {
        "split": split, "rows": len(a), "theraset_top5_misses": int(missed.sum()),
        "decision_rules": {"weights": [1 / 3] * 3, "thresholds": [.5, .28, .64], "low_rank_bonus": 2.},
        "results": {},
    }
    for key, pred in (("theraset", thera_pred), ("equal", equal_pred)):
        output["results"][key] = summarize(y, pred, baseline, cnn["cluster_id"], missed, np.zeros_like(empty))
    high = np.where(rank <= 5, mix, -np.inf).argmax(1)
    low = np.where((rank >= 6) & (rank <= 10), mix, -np.inf).argmax(1)
    row = np.arange(len(a))
    for key, bonus in (("empty_top10", 0.), ("empty_top10_low_rank_bonus", 2.)):
        label = np.where(mix[row, low] + bonus > mix[row, high], low, high)
        pred = equal_pred.copy()
        pred[row[empty], label[empty]] = True
        output["results"][key] = summarize(y, pred, baseline, cnn["cluster_id"], missed, empty)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("theraset", "therapep", "helixset"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    thera = read_theraset(args.theraset, args.split)
    cnn = from_zip(args.therapep, f"/fine39/traditional5/{args.split}.predictions.npz")
    helix = from_zip(args.helixset, "/best_validation.npz" if args.split == "val" else "/test_predictions.npz")
    report = evaluate(thera, cnn, helix, args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
