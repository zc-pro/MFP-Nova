"""Offline validation of independent Train-cluster evidence for DuoEvidence."""

import argparse
from datetime import datetime
from io import BytesIO
from pathlib import Path
import tarfile

import numpy as np
from threadpoolctl import threadpool_limits

from mfp_nova.data import CachedPeptides, PeptideData
from mfp_nova.reference_fallback import aa34_summary, fallback_comparison, select_fallback, support_scores
from mfp_nova.trainer import write_json


def pool_esm(data, root, split):
    cache = CachedPeptides(data, root, split)
    offsets = cache.offsets
    return np.stack([cache.values[offsets[i]:offsets[i + 1]].mean(0, dtype=np.float32)
                     for i in cache.positions])


def diagnose(y, raw, final, logits, t_logits, a_logits, support, low, clusters):
    order = np.argsort(-logits, axis=1, kind="stable")
    in_top5 = np.argsort(order, axis=1) < 5
    rows = np.arange(len(y))
    hit = (y & final).any(1)
    correct = np.where(y, logits, -np.inf).argmax(1)
    wrong = np.where(~y, logits, -np.inf).argmax(1)
    correct5 = (in_top5 & y).any(1)
    both_wrong = (t_logits[rows, correct] < t_logits[rows, wrong]) & (a_logits[rows, correct] < a_logits[rows, wrong])
    nonempty = raw.any(1)
    margin = support[rows, correct] - support[rows, wrong]
    groups = {"nonempty_zero": ~hit & nonempty, "nonempty_zero_true_top5": ~hit & nonempty & correct5,
              "nonempty_zero_true_top5_both_prefer_error": ~hit & nonempty & correct5 & both_wrong,
              "fallback_zero_true_top5": ~hit & ~nonempty & correct5}
    if low is not None:
        groups["nonempty_zero_true_top5_both_prefer_error_low"] = groups["nonempty_zero_true_top5_both_prefer_error"] & low
    pair = {}
    for key, mask in groups.items():
        valid = mask & np.isfinite(margin)
        pair[key] = {"rows": int(mask.sum()), "eligible": int(valid.sum()),
                     "references_prefer_observed": int((valid & (margin > 0)).sum()),
                     "median_support_margin": float(np.median(margin[valid])) if valid.any() else None}

    # Fixed candidate pool and one prediction per row. Measure both wins and losses.
    original_top = order[:, 0]
    eligible = (np.isfinite(support) | ~in_top5).all(1)
    reference_top = np.where(in_top5 & np.isfinite(support), support, -np.inf).argmax(1)
    original_hit, reference_hit = y[rows, original_top], y[rows, reference_top]
    eligible_rows, updated = select_fallback(raw, final, t_logits, support)
    return {"pair_diagnostics": pair,
            "fixed_theraset_top10_fallback": fallback_comparison(y, final, updated, eligible_rows, clusters),
            "fixed_fused_top5_probe": {
                "eligible": int(eligible.sum()),
                "top1_hit_before": int((eligible & original_hit).sum()),
                "top1_hit_after": int((eligible & reference_hit).sum()),
                "top1_gained": int((eligible & ~original_hit & reference_hit).sum()),
                "top1_lost": int((eligible & original_hit & ~reference_hit).sum()),
                "original_final_hit_but_reference_top1_wrong": int((eligible & hit & ~reference_hit).sum()),
                "rescued_original_final_zero": int((eligible & ~hit & reference_hit).sum()),
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--member", help="Prediction member within a tar archive")
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--nearest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--aa34-only", action="store_true", help="Only evaluate fixed AA34 and dipeptide references")
    args = parser.parse_args()
    data = PeptideData(args.data_root)
    train, val = (data.indices(split) for split in ("train", "val"))
    if np.intersect1d(data.clusters[train], data.clusters[val]).size:
        raise ValueError("Train and Validation clusters overlap")
    source = args.predictions
    if args.member:
        with tarfile.open(source) as archive:
            source = BytesIO(archive.extractfile(args.member).read())
    with np.load(source) as saved:
        if str(saved["split"]) != "val":
            raise ValueError("Expected Fine39 Validation predictions")
        for key, expected in (("row_id", data.rows[val]), ("targets", data.y[val]),
                              ("label_names", data.labels), ("cluster_id", data.clusters[val])):
            np.testing.assert_array_equal(saved[key], expected)
        thera, aa34, offsets = (saved[key] for key in ("theraset_logits", "aa34_logits", "offsets"))
        y, raw, final = (saved[key].astype(bool) for key in ("targets", "predicted_raw", "predicted"))
    limit = np.log(999999.)
    logits = ((np.clip(thera, -limit, limit) - offsets[0]) + (np.clip(aa34, -limit, limit) - offsets[1])) / 2
    low = None
    if args.nearest:
        with np.load(args.nearest) as saved:
            order = np.argsort(saved["row_id"])
            np.testing.assert_array_equal(saved["row_id"][order], data.rows[val])
            low = saved["similarity"][order] < .7
    output = args.output or Path("runs") / f"duoevidence_cluster_references_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    report = {"train_rows": len(train), "validation_rows": len(val), "member": args.member,
              "scope": "Exploratory Validation comparison using fixed Train references", "features": {}}
    with threadpool_limits(limits=args.threads):
        for name, feature in (
            ("aa34_input_and_dipeptides", lambda split, idx: aa34_summary(data.sequences[idx])),
            ("esm2_650m", lambda split, idx: pool_esm(data, args.features, split)),
            ("esm2_peptide_150m", lambda split, idx: pool_esm(data, args.peptide_features, split)),
        ):
            if args.aa34_only and name != "aa34_input_and_dipeptides":
                continue
            print(f"Compare {name}", flush=True)
            support = support_scores(feature("train", train), feature("val", val),
                                     data.y[train].astype(bool), data.clusters[train])
            report["features"][name] = diagnose(y, raw, final, logits, thera, aa34, support, low, data.clusters[val])
            write_json(output / "results.json", report)
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
