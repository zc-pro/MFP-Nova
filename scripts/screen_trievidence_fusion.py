"""Cluster-held-out screen of late fusion rules for fixed TriEvidence Validation logits."""

import argparse
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import torch

from evaluate_fine39_fixed_fusion import summarize
from mfp_nova.hypothesis_data import family_folds
from mfp_nova.models.trievidence import center_evidence, empty_top10
from mfp_nova.trainer import write_json
from run_trievidence import row_jaccard


def best_label(score, eligible):
    label = score.masked_fill(~eligible, -torch.inf).argmax(1, keepdim=True)
    return torch.zeros_like(eligible).scatter_(1, label, eligible.any(1, keepdim=True))


def action_trials(score, cnn_margin, reference, eligible, mode):
    if mode == "add":
        size = (score >= 0).sum(1, keepdim=True)
        eligible = eligible & (size > 0) & ~reference
        gap = -score
    else:
        size = (score >= 0).sum(1, keepdim=True)
        eligible = eligible & (size == 0) & ~reference
        gap = score.masked_fill(~reference, -torch.inf).amax(1, keepdim=True) - score
    for confidence in (0., .5, 1., 2.):
        strong = eligible & (cnn_margin >= confidence)
        for width in (.125, .25, .5, 1., 2., 4.):
            allowed = strong & (gap <= width)
            for single in ((False, True) if mode == "add" else (False,)):
                chosen = best_label(score, allowed & (size == 1) if single else allowed)
                rule = {"min_cnn_logit_above_cutoff": confidence, "max_fusion_gap": width,
                        "one_label_base_only": single}
                if mode == "add":
                    yield rule, reference | chosen
                else:
                    yield rule, torch.where(chosen.any(1, keepdim=True), chosen, reference)


def evaluate_trials(y, reference, groups, trials):
    baseline = row_jaccard(y, reference)
    choices = [(0., 0, {"rule": "off"}) for _ in range(5)]
    decisions = reference.clone()
    for rule, candidate in trials:
        delta = (row_jaccard(y, candidate) - baseline).numpy()
        edited = (reference != candidate).any(1).numpy()
        sums = np.bincount(groups, weights=delta, minlength=5)
        edits = np.bincount(groups, weights=edited, minlength=5)
        for i in range(5):
            score = (float(sums.sum() - sums[i]), -int(edits.sum() - edits[i]))
            if score > choices[i][:2]:
                choices[i] = (*score, rule)
                held = groups == i
                decisions[held] = candidate[held]
    return decisions, [rule for _, _, rule in choices]


def screen(path, output):
    params = json.loads(path.with_name("results.json").read_text(encoding="utf-8"))
    with np.load(path) as data:
        if str(data["split"]) != "val":
            raise ValueError("This script screens rules on original Validation only")
        rows, clusters = data["row_id"], data["cluster_id"]
        if len(np.unique(rows)) != len(rows) or data["targets"].shape != (8529, 39):
            raise ValueError("Expected the complete original Fine39 Validation")
        y = torch.from_numpy(data["targets"].astype(bool))
        thera, cnn, latent = (torch.from_numpy(data[key]) for key in
                             ("theraset_logits", "aa34_logits", "latent_logits"))
        raw_expected, fallback_expected = data["predicted_raw"], data["predicted"]
    offsets = torch.tensor(params["branch_thresholds"]).logit()
    a, c, h = center_evidence(torch.stack((thera, cnn, latent)), offsets)
    score = torch.stack((a, c, h)).mean(0)
    raw = score >= 0
    fallback = empty_top10(score, thera)
    np.testing.assert_array_equal(raw.numpy(), raw_expected)
    np.testing.assert_array_equal(fallback.numpy(), fallback_expected)
    rank = thera.argsort(dim=1, descending=True, stable=True)
    top = cnn.argsort(dim=1, descending=True, stable=True)[:, :5]
    low = torch.zeros_like(y).scatter_(1, rank[:, 5:10], True)
    in_cnn_top5 = torch.zeros_like(y).scatter_(1, top, True)
    eligible = low & in_cnn_top5
    groups = family_folds(clusters, 5)
    thera_top5_miss = ~y.gather(1, rank[:, :5]).any(1).numpy()

    weights = []
    for w in (.5, .75, 1.25, 1.5, 2.):
        mixed = (a + w * c + h) / (2 + w)
        weights.append(({"cnn_weight": w}, empty_top10(mixed, thera)))
    trials = {
        "global_cnn_weight": weights,
        "nonempty_add_rank6_10": action_trials(score, c, fallback, eligible, "add"),
        "empty_rerank_rank6_10": action_trials(score, c, fallback, eligible, "rerank"),
    }
    report = {"scope": "Cluster-held-out fusion rules, conditional on the fixed Validation-selected reader",
              "rows": len(rows), "source": str(path), "baseline_raw_acc_pct": 100 * float(row_jaccard(y, raw).mean()),
              "baseline_fallback_acc_pct": 100 * float(row_jaccard(y, fallback).mean()), "rules": {}}
    out = {}
    baseline = ((y & fallback).sum(1).double() / (y | fallback).sum(1).clamp_min(1)).numpy()
    selected = {name: evaluate_trials(y, fallback, groups, candidates) for name, candidates in trials.items()}
    fixed_gap = 1.
    fixed_rule, fixed_pred = next((rule, pred) for rule, pred in action_trials(score, c, fallback, eligible, "add")
                                  if rule["min_cnn_logit_above_cutoff"] == 0.
                                  and rule["max_fusion_gap"] == fixed_gap and not rule["one_label_base_only"])
    selected["nonempty_add_fixed"] = (fixed_pred, [fixed_rule])
    for name, (pred, choices) in selected.items():
        added, removed = pred & ~fallback, fallback & ~pred
        summary = summarize(y.numpy(), pred.numpy(), baseline, clusters,
                            thera_top5_miss, np.zeros(len(rows), dtype=bool))
        summary["delta_pp"] = summary.pop("delta_vs_theraset_pp")
        summary.pop("fallback_additions")
        summary.pop("fallback_observed_hits")
        summary.update(fold_choices=choices, changed_rows=int((pred != fallback).any(1).sum()),
                       added_true=int((added & y).sum()), added_unannotated=int((added & ~y).sum()),
                       removed_true=int((removed & y).sum()), removed_unannotated=int((removed & ~y).sum()),
                       added_true_rank6_10=int((added & y & low).sum()))
        report["rules"][name] = summary
        out[name] = pred.numpy()
        print(f"{name} | ACC={summary['acc_pct']:.4f}% | delta={summary['delta_pp']:+.4f}pp "
              f"| edited={summary['changed_rows']}", flush=True)
    write_json(output / "results.json", report)
    np.savez_compressed(output / "val_decisions.npz", row_id=rows, cluster_id=clusters, folds=groups,
                        baseline=fallback.numpy(), **out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("runs") / datetime.now().strftime("trievidence_fusion_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    screen(args.predictions, output)
    print(output, flush=True)


if __name__ == "__main__":
    main()
