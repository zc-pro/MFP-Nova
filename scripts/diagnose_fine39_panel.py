"""Unified Fine39 diagnostic panel with paired cluster bootstrap 95% CI."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from mfp_nova.multimodal import function_projection, independent_functions


def load_export(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    probs = data["probabilities"] if "probabilities" in data else 1.0 / (1.0 + np.exp(-data["logits"]))
    out = {
        "row_id": data["row_id"],
        "cluster_id": data["cluster_id"],
        "targets": data["targets"].astype(bool),
        "probabilities": probs.astype(np.float64),
        "label_names": [str(x) for x in data["label_names"]],
    }
    for key in ("path_a_probabilities", "path_b_probabilities"):
        if key in data:
            out[key] = data[key].astype(np.float64)
    return out


def sample_jaccard(y, pred):
    inter = (y & pred).sum(axis=1)
    union = (y | pred).sum(axis=1)
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def apply_empty_top1(pred, probs):
    out = pred.copy()
    empty = ~out.any(axis=1)
    if empty.any():
        out[empty, probs[empty].argmax(axis=1)] = True
    return out


def paired_cluster_bootstrap_ci(delta, clusters, n_boot=2000, seed=42):
    uniq, inv = np.unique(clusters, return_inverse=True)
    c_sum = np.bincount(inv, weights=delta)
    c_cnt = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = c_sum[idx].sum(axis=1) / c_cnt[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return [float(lo * 100.0), float(hi * 100.0)]


def evaluate_panel(export, threshold=0.5, low_mask=None):
    y = export["targets"]
    p = export["probabilities"]
    labels = export["label_names"]
    raw_pred = p >= threshold
    fb_pred = apply_empty_top1(raw_pred, p)

    raw_j = sample_jaccard(y, raw_pred)
    fb_j = sample_jaccard(y, fb_pred)
    raw_empty = ~raw_pred.any(axis=1)
    raw_zero = ~(raw_pred & y).any(axis=1)
    nonempty_zero = raw_zero & ~raw_empty
    fb_zero = ~(fb_pred & y).any(axis=1)

    order = np.argsort(-p, axis=1, kind="stable")
    top1_hit = y[np.arange(len(y)), order[:, 0]]
    top5_hit = np.take_along_axis(y, order[:, :5], axis=1).any(axis=1)

    ap_list = [average_precision_score(y[:, c], p[:, c]) for c in range(y.shape[1]) if y[:, c].any()]
    macro_ap = float(np.mean(ap_list)) if ap_list else 0.0
    micro_f1 = float(f1_score(y.ravel(), raw_pred.ravel(), zero_division=0))

    proj = function_projection(labels, "fine39")
    axes_y = independent_functions(y, proj)
    axes_p = independent_functions(raw_pred, proj)
    multi_mask = axes_y.sum(axis=1) >= 2
    multi_pred = axes_p.sum(axis=1) >= 2
    mfp_det_f1 = float(f1_score(multi_mask, multi_pred, zero_division=0))
    multi_jaccard = float(raw_j[multi_mask].mean()) if multi_mask.any() else 0.0
    ge2_recovered = float((axes_p[multi_mask] & axes_y[multi_mask]).sum(axis=1).ge(2).mean()) if False else (
        float(((axes_p[multi_mask] & axes_y[multi_mask]).sum(axis=1) >= 2).mean()) if multi_mask.any() else 0.0
    )

    result = {
        "rows": int(len(y)),
        "clusters": int(len(np.unique(export["cluster_id"]))),
        "threshold": float(threshold),
        "raw_acc_pct": float(raw_j.mean() * 100.0),
        "fallback_acc_pct": float(fb_j.mean() * 100.0),
        "macro_ap_pct": macro_ap * 100.0,
        "micro_f1_pct": micro_f1 * 100.0,
        "labels_per_sample": float(raw_pred.sum(axis=1).mean()),
        "raw_empty": int(raw_empty.sum()),
        "nonempty_zero": int(nonempty_zero.sum()),
        "final_zero": int(fb_zero.sum()),
        "top1_hit_pct": float(top1_hit.mean() * 100.0),
        "top5_hit_pct": float(top5_hit.mean() * 100.0),
        "zero_top5_coverage_pct": float(top5_hit[fb_zero].mean() * 100.0) if fb_zero.any() else 0.0,
        "multifunction_rows": int(multi_mask.sum()),
        "multifunction_acc_pct": multi_jaccard * 100.0,
        "mfp_detection_f1_pct": mfp_det_f1 * 100.0,
        "ge2_axes_recovered_pct": ge2_recovered * 100.0,
    }
    if low_mask is not None and low_mask.any():
        result["low_sim_rows"] = int(low_mask.sum())
        result["low_sim_raw_acc_pct"] = float(raw_j[low_mask].mean() * 100.0)
        result["low_sim_fallback_acc_pct"] = float(fb_j[low_mask].mean() * 100.0)
    if "path_a_probabilities" in export and "path_b_probabilities" in export:
        pa, pb = export["path_a_probabilities"], export["path_b_probabilities"]
        pa_raw = pa >= threshold
        pb_raw = pb >= threshold
        pa_fb = apply_empty_top1(pa_raw, pa)
        pb_fb = apply_empty_top1(pb_raw, pb)
        a_hit = (pa_fb & y).any(axis=1)
        b_hit = (pb_fb & y).any(axis=1)
        result["dualpath_branches"] = {
            "path_a_raw_acc_pct": float(sample_jaccard(y, pa_raw).mean() * 100.0),
            "path_a_fallback_acc_pct": float(sample_jaccard(y, pa_fb).mean() * 100.0),
            "path_b_raw_acc_pct": float(sample_jaccard(y, pb_raw).mean() * 100.0),
            "path_b_fallback_acc_pct": float(sample_jaccard(y, pb_fb).mean() * 100.0),
            "path_b_unique_rescues_over_a": int((~a_hit & b_hit).sum()),
            "path_a_unique_rescues_over_b": int((a_hit & ~b_hit).sum()),
        }
    return result, raw_j, fb_j, raw_pred, fb_pred


def compare_exports(cand, base, threshold=0.5, low_mask=None):
    for key in ("row_id", "cluster_id", "targets", "label_names"):
        if not np.array_equal(cand[key], base[key]):
            raise ValueError(f"Mismatch between candidate and baseline on '{key}'")
    cand_res, c_raw_j, c_fb_j, c_raw, c_fb = evaluate_panel(cand, threshold, low_mask)
    base_res, b_raw_j, b_fb_j, b_raw, b_fb = evaluate_panel(base, threshold, low_mask)
    y = cand["targets"]
    clusters = cand["cluster_id"]
    raw_delta = c_raw_j - b_raw_j
    fb_delta = c_fb_j - b_fb_j
    c_hit = (c_fb & y).any(axis=1)
    b_hit = (b_fb & y).any(axis=1)
    comp = {
        "raw_delta_pp": float(raw_delta.mean() * 100.0),
        "raw_ci95_pp": paired_cluster_bootstrap_ci(raw_delta, clusters),
        "fallback_delta_pp": float(fb_delta.mean() * 100.0),
        "fallback_ci95_pp": paired_cluster_bootstrap_ci(fb_delta, clusters),
        "rescued_zero_rows": int((~b_hit & c_hit).sum()),
        "new_zero_rows": int((b_hit & ~c_hit).sum()),
        "gained_true_labels": int((~b_raw & c_raw & y).sum()),
        "lost_true_labels": int((b_raw & ~c_raw & y).sum()),
        "delta_unrecorded_preds": int((c_raw & ~y).sum() - (b_raw & ~y).sum()),
    }
    if low_mask is not None and low_mask.any():
        comp["low_sim_raw_delta_pp"] = float(raw_delta[low_mask].mean() * 100.0)
        comp["low_sim_raw_ci95_pp"] = paired_cluster_bootstrap_ci(raw_delta[low_mask], clusters[low_mask])
    return {"candidate": cand_res, "baseline": base_res, "paired_comparison": comp}


def main():
    parser = argparse.ArgumentParser(description="Fine39 unified diagnostic panel")
    parser.add_argument("candidate", type=Path, help="Candidate validation predictions NPZ")
    parser.add_argument("--baseline", type=Path, help="Optional baseline validation predictions NPZ")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    cand = load_export(args.candidate)
    if args.baseline is not None:
        base = load_export(args.baseline)
        report = compare_exports(cand, base, threshold=args.threshold)
    else:
        report, *_ = evaluate_panel(cand, threshold=args.threshold)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
