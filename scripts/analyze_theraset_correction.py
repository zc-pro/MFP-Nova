"""Attribute Fine39 ranking changes to the direct head and evidence correction."""

import argparse
import json
from pathlib import Path

import numpy as np

from mfp_nova.hypothesis_data import family_folds


def ranks(scores):
    return np.argsort(np.argsort(-scores, axis=1, kind="stable"), axis=1) + 1


def row_jaccard(predicted, targets):
    return (predicted & targets).sum(1) / (predicted | targets).sum(1)


def cohort(selected, direct_rank, final_rank, targets, final_hits, correction):
    before = (targets & (direct_rank <= 5)).any(1)
    after = (targets & (final_rank <= 5)).any(1)
    dropped = selected[:, None] & targets & (direct_rank <= 5) & (final_rank > 5)
    return {
        "rows": int(selected.sum()),
        "direct_top5_hits": int((selected & before).sum()),
        "final_top5_hits": int((selected & after).sum()),
        "direct_hit_final_miss": int((selected & before & ~after).sum()),
        "both_miss": int((selected & ~before & ~after).sum()),
        "direct_miss_final_hit": int((selected & ~before & after).sum()),
        "final_threshold_hits": int((selected & final_hits).sum()),
        "known_labels_that_fell_out_of_top5": int(dropped.sum()),
        "dropped_labels_with_negative_logit_correction": int((dropped & (correction < 0)).sum()),
    }


def probe(direct_rank, final_logits, correction, targets, clusters, baseline, missed):
    """Choose a fixed rule on four folds and assess on the held-out cluster fold."""
    scores = {"unchanged": baseline}
    for k in (1, 3, 5):
        protected = (direct_rank <= k) & (correction < 0)
        for strength in (.25, .5, 1.):
            adjusted = final_logits + np.where(protected, -strength * correction, 0)
            scores[f"top{k}_undo_{strength:g}"] = row_jaccard(adjusted >= 0, targets)
    folds = family_folds(clusters, 5)
    selected = {}
    result = np.zeros_like(baseline)
    for fold in np.unique(folds):
        held = folds == fold
        choice = max(scores, key=lambda key: scores[key][~held].mean())
        result[held] = scores[choice][held]
        selected[str(fold)] = choice
    delta = result - baseline
    _, groups = np.unique(clusters, return_inverse=True)
    counts = np.bincount(groups)
    sums = np.bincount(groups, weights=delta)
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(2000):
        draw = rng.integers(len(counts), size=len(counts))
        samples.append(100 * sums[draw].sum() / counts[draw].sum())
    return {
        "rule_by_fold": selected,
        "acc_pct": float(100 * result.mean()),
        "delta_pp": float(100 * delta.mean()),
        "paired_cluster_ci95_pp": np.quantile(samples, [.025, .975]).tolist(),
        "gained_threshold_hits_in_reference_misses": int(((baseline == 0) & (result > 0) & missed).sum()),
        "lost_threshold_hits_in_reference_misses": int(((baseline > 0) & (result == 0) & missed).sum()),
        "better_rows": int((delta > 0).sum()),
        "worse_rows": int((delta < 0).sum()),
    }


def analyze(evidence, reference):
    rows = evidence["row_id"]
    index = np.argsort(reference["row_id"])
    if len(rows) != len(index) or len(np.unique(rows)) != len(rows):
        raise ValueError("Expected matching validation samples and unique row IDs")
    index = index[np.searchsorted(reference["row_id"][index], rows)]
    for key in ("row_id", "targets", "cluster_id"):
        np.testing.assert_array_equal(evidence[key], reference[key][index])
    np.testing.assert_array_equal(evidence["label_names"], reference["label_names"])
    targets = evidence["targets"].astype(bool)
    direct = evidence["direct_probabilities"]
    final = evidence["probabilities"]
    ref = reference["probabilities"][index]
    correction = evidence["correction"]
    if "logits" in evidence:
        logits = evidence["logits"]
        np.testing.assert_allclose(logits, evidence["direct_logits"] + correction, atol=1e-5, rtol=1e-6)
    else:
        # Historical exports omit raw logits. Verify the reconstructed cutoff.
        probability = np.clip(direct.astype(np.float64), 1e-7, 1 - 1e-7)
        logits = np.log(probability) - np.log1p(-probability) + correction
    np.testing.assert_array_equal(logits >= 0, final >= .5)
    direct_rank, final_rank, ref_rank = (ranks(x) for x in (direct, final, ref))
    own_miss = ~(targets & (final_rank <= 5)).any(1)
    ref_miss = ~(targets & (ref_rank <= 5)).any(1)
    rank_6_to_10 = ref_miss & (targets & (ref_rank >= 6) & (ref_rank <= 10)).any(1)
    base = row_jaccard(final >= .5, targets)
    groups = {"all": np.ones(len(rows), dtype=bool), "own_top5_miss": own_miss,
              "reference_top5_miss": ref_miss, "reference_top5_miss_with_ranks_6_to_10": rank_6_to_10}
    return {
        "reference_probabilities_match": bool(np.allclose(final, ref, atol=1e-5, rtol=1e-5)),
        "reference_acc_pct_at_05": float(100 * row_jaccard(ref >= .5, targets).mean()),
        "direct_acc_pct_at_05": float(100 * row_jaccard(direct >= .5, targets).mean()),
        "final_acc_pct_at_05": float(100 * base.mean()),
        "cohorts": {key: cohort(mask, direct_rank, final_rank, targets, base > 0, correction)
                    for key, mask in groups.items()},
        "protect_direct_candidates_crossfit": probe(
            direct_rank, logits, correction, targets, evidence["cluster_id"], base, ref_miss),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.evidence) as evidence, np.load(args.reference) as reference:
        result = analyze(evidence, reference)
    result.update(evidence=str(args.evidence), reference=str(args.reference))
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
