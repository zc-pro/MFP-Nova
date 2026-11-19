"""Paired 2x2 branch replacement audit on the original Fine39 Validation."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from evaluate_fine39_fixed_fusion import from_zip
from mfp_nova.models.trievidence import center_evidence, empty_top10, fuse_evidence
from mfp_nova.multimodal import BackbonePeptides, MultimodalData, independent_functions
from mfp_nova.trainer import write_json
from run_trievidence import align_saved, row_jaccard


def audit(args):
    source = MultimodalData(args.data_root, "fine39", sequence_only=True)
    val = BackbonePeptides(source, "val", 80)
    idx = val.indices
    truth = torch.from_numpy(source.data.y[idx].astype(bool))
    with np.load(args.single) as data:
        if str(data["split"]) != "val":
            raise ValueError("Use original Validation predictions")
        for field, expected in (("row_id", source.data.rows[idx]), ("targets", truth.numpy()),
                                ("label_names", source.data.labels), ("cluster_id", source.data.clusters[idx])):
            np.testing.assert_array_equal(data[field], expected)
        a, single, latent = (torch.from_numpy(data[key]) for key in
                             ("theraset_logits", "aa34_logits", "latent_logits"))
    offsets = torch.tensor(json.loads(args.single.with_name("results.json").read_text(encoding="utf-8"))["branch_thresholds"])
    old = from_zip(args.five, "/fine39/traditional5/val.predictions.npz")
    five = align_saved(old, source, val, "probabilities")
    with np.load(args.helix) as exits:
        full = align_saved(exits, source, val, "decoder_3")
        archived = align_saved(exits, source, val, "direct")
    torch.testing.assert_close(latent, archived, atol=1e-5, rtol=1e-5)

    def top_mask(logits, k):
        top = logits.argsort(dim=1, descending=True, stable=True)[:, :k]
        return torch.zeros_like(truth).scatter_(1, top, True)

    multi = torch.from_numpy(independent_functions(truth.numpy(), source.statistics["function_projection"]).sum(1) >= 2)
    thera10 = top_mask(a, 10)
    arms, outputs = {}, {}
    for cnn_name, cnn, cut in (("five", five, float(old["threshold"])), ("single", single, float(offsets[1]))):
        for helix_name, helix in (("full", full), ("latent", latent)):
            name = f"{cnn_name}_{helix_name}"
            threshold = offsets.clone()
            threshold[1] = cut
            fused = fuse_evidence(a, cnn, helix, threshold.logit())
            raw = fused >= 0
            final = empty_top10(fused, a)
            empty = ~raw.any(1)
            correct = (truth & final).sum(1)
            overlap = (thera10 & truth).any(1)
            expanded = (truth & (thera10 | top_mask(cnn, 5) | top_mask(helix, 5))).any(1)
            potential = (empty & overlap).double() / truth.sum(1).clamp_min(1)
            expansion = (empty & expanded & ~overlap).double() / truth.sum(1).clamp_min(1)
            arms[name] = {
                "raw_acc_pct": 100 * float(row_jaccard(truth, raw).mean()),
                "final_acc_pct": 100 * float(row_jaccard(truth, final).mean()),
                "multifunction_final_acc_pct": 100 * float(row_jaccard(truth, final)[multi].mean()),
                "final_zero_hit": int((correct == 0).sum()),
                "empty": int(empty.sum()),
                "empty_fallback_hits": int((empty & (correct > 0)).sum()),
                "empty_has_true_theraset_top10": int((empty & overlap).sum()),
                "empty_has_true_expanded": int((empty & expanded).sum()),
                "empty_top10_oracle_gain_pp": 100 * float(potential.mean()),
                "empty_extra_oracle_expanded_pp": 100 * float(expansion.mean()),
            }
            outputs[name] = (raw, final, empty)
    raw_old, final_old, empty_old = outputs["five_full"]
    raw_new, final_new, empty_new = outputs["single_latent"]
    centered = center_evidence(torch.stack((a, single, latent)), offsets.logit())
    removals = {}
    for name, keep in (("theraset_only", [0]), ("remove_aa34", [0, 2]), ("remove_helix", [0, 1])):
        score = centered[keep].mean(0)
        raw = score >= 0
        final = empty_top10(score, a)
        removals[name] = {
            "raw_acc_pct": 100 * float(row_jaccard(truth, raw).mean()),
            "final_acc_pct": 100 * float(row_jaccard(truth, final).mean()),
            "raw_delta_pp_vs_three": 100 * float((row_jaccard(truth, raw) - row_jaccard(truth, raw_new)).mean()),
            "raw_zero_hits": int((~(truth & raw).any(1)).sum()),
        }
    paired = {}
    delta_raw = row_jaccard(truth, raw_new) - row_jaccard(truth, raw_old)
    delta_final = row_jaccard(truth, final_new) - row_jaccard(truth, final_old)
    for name, mask in (("both_empty", empty_old & empty_new), ("old_only_empty", empty_old & ~empty_new),
                       ("new_only_empty", empty_new & ~empty_old), ("neither_empty", ~empty_new & ~empty_old)):
        paired[name] = {"rows": int(mask.sum()),
                        "raw_delta_pp_contribution": 100 * float((delta_raw * mask).mean()),
                        "final_delta_pp_contribution": 100 * float((delta_final * mask).mean()),
                        "multifunction_delta_pp_contribution": 100 * float((delta_final * mask * multi).mean()),
                        "gained_observed_labels": int((truth & final_new & ~final_old & mask[:, None]).sum()),
                        "lost_observed_labels": int((truth & final_old & ~final_new & mask[:, None]).sum()),
                        "gained_unannotated_labels": int((~truth & final_new & ~final_old & mask[:, None]).sum()),
                        "removed_unannotated_labels": int((~truth & final_old & ~final_new & mask[:, None]).sum())}
    report = {"split": "val", "rows": len(val), "source_single": str(args.single),
              "source_five": str(args.five), "source_helix_exits": str(args.helix),
              "arms": arms, "paired": paired, "fixed_threshold_branch_removal": removals}
    output = Path("runs") / datetime.now().strftime("trievidence_replacements_%Y%m%d_%H%M%S_%f")
    output.mkdir()
    write_json(output / "results.json", report)
    print(json.dumps(report, indent=2), output, sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("single", "five", "helix"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    torch.set_num_threads(4)
    audit(parser.parse_args())
