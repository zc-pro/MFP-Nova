"""Zero-cost CPU offline probe & 3-parameter prior-aware BCE calibration + Expected Jaccard decoder."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mfp_nova.data import PeptideData
from mfp_nova.decisions import apply_decision, expected_jaccard_decode
from mfp_nova.multimodal import label_statistics


def logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def sample_jaccard(y_true, y_pred):
    inter = (y_true & y_pred).sum(axis=1)
    union = (y_true | y_pred).sum(axis=1)
    return float(np.where(union > 0, inter / np.maximum(union, 1), 0.0).mean()) * 100.0


def eval_threshold(y_true, probs, th=0.5):
    pred = probs >= th
    raw = sample_jaccard(y_true, pred)
    fb = pred.copy()
    empty = ~fb.any(axis=1)
    if empty.any():
        fb[empty, probs[empty].argmax(axis=1)] = True
    return raw, sample_jaccard(y_true, fb), float(pred.sum(axis=1).mean()), int(empty.sum())


def cluster_folds(clusters, k=5, seed=42):
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    mapping = {int(c): int(i % k) for i, c in enumerate(uniq)}
    return np.array([mapping[int(c)] for c in clusters], dtype=np.int32)


def fit_prior_aware_bce(logits, y_true, log_pos_weights):
    """Fit 3 global scalars (alpha, beta0, beta1) minimizing proper BCE loss (zero tail overfitting)."""
    z = torch.from_numpy(logits.astype(np.float32))
    y = torch.from_numpy(y_true.astype(np.float32))
    lw = torch.from_numpy(log_pos_weights.astype(np.float32))[None, :]

    params = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([params], lr=0.5, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        alpha = params[0].clamp(0.2, 5.0)
        z_cal = alpha * z + params[1] - params[2] * lw
        loss = F.binary_cross_entropy_with_logits(z_cal, y) + 1e-4 * (params - torch.tensor([1.0, 0.0, 0.0])).square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        a = float(params[0].clamp(0.2, 5.0).item())
        b0 = float(params[1].item())
        b1 = float(params[2].item())
    return a, b0, b1


def evaluate_logits(name, logits, y_true, clusters, log_pos_weights):
    p_raw = sigmoid(logits)
    r50, f50, l50, e50 = eval_threshold(y_true, p_raw, 0.50)
    folds = cluster_folds(clusters, k=5, seed=42)
    p_bce = np.zeros_like(p_raw)
    for f in range(5):
        tr, va = folds != f, folds == f
        a, b0, b1 = fit_prior_aware_bce(logits[tr], y_true[tr], log_pos_weights)
        p_bce[va] = sigmoid(a * logits[va] + b0 - b1 * log_pos_weights[None, :])
    rbce, fbce, lbce, _ = eval_threshold(y_true, p_bce, 0.50)
    ej_pred = expected_jaccard_decode(p_bce, top_m=8)
    ej_acc = sample_jaccard(y_true, ej_pred)
    ej_lps = float(ej_pred.sum(axis=1).mean())
    a_all, b0_all, b1_all = fit_prior_aware_bce(logits, y_true, log_pos_weights)
    print(f"[{name}]")
    print(f"  @0.50 Raw={r50:.4f}% | Fallback={f50:.4f}% (lps={l50:.3f}, empty={e50})")
    print(f"  5-Fold Cluster-CV 3-Param BCE @0.50 -> Raw={rbce:.4f}% | Fallback={fbce:.4f}% (lps={lbce:.3f})")
    print(f"  5-Fold Cluster-CV 3-Param BCE + Top-8 Expected Jaccard -> ACC={ej_acc:.4f}% (lps={ej_lps:.3f})")
    print(f"  Full-Val 3-Param Fit: alpha={a_all:.4f}, beta0={b0_all:.4f}, beta1={b1_all:.4f}")
    return a_all, b0_all, b1_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym04", required=True, help="Primary DualPath best_validation.npz")
    parser.add_argument("--theraset", default=None, help="Optional standalone TheraSet best_validation.npz")
    parser.add_argument("--asl", default=None, help="Optional comparison DualPath best_validation.npz")
    parser.add_argument("--data-root", default="data/therapeutic_2025", help="Dataset root for Train positive_weights")
    parser.add_argument("--export-decision", default=None, help="Optional path to write decision.json for evaluate/predict")
    args = parser.parse_args()

    s04 = np.load(args.sym04, allow_pickle=False)
    y = s04["targets"].astype(bool)
    clusters = s04["cluster_id"]
    labels = s04["label_names"].tolist()

    data = PeptideData(args.data_root, "fine39")
    tr_idx = data.indices("train")
    stats = label_statistics(data.y[tr_idx], data.clusters[tr_idx], data.labels)
    log_pos_w = np.log(np.maximum(np.asarray(stats["positive_weights"], dtype=np.float64), 1e-6))

    z_main = logit(s04["probabilities"])
    a, b0, b1 = evaluate_logits("Primary Candidate", z_main, y, clusters, log_pos_w)

    if args.export_decision:
        decision = {
            "rule": "prior_calibrated_expected_jaccard",
            "task": "fine39",
            "labels": labels,
            "alpha": a,
            "beta0": b0,
            "beta1": b1,
            "log_positive_weights": log_pos_w.tolist(),
            "top_m": 8,
        }
        out_path = Path(args.export_decision)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        applied = apply_decision(s04["probabilities"], labels, "fine39", decision)
        print(f"Exported decision.json -> {out_path} (in-sample Expected Jaccard={sample_jaccard(y, applied):.4f}%)")

    if args.theraset and "path_b_probabilities" in s04:
        th = np.load(args.theraset, allow_pickle=False)
        assert np.array_equal(s04["row_id"], th["row_id"]), "row_id mismatch"
        evaluate_logits("TheraSet + Candidate-PathB [Undiluted PathA Proxy]",
                        0.5 * (logit(th["probabilities"]) + logit(s04["path_b_probabilities"])),
                        y, clusters, log_pos_w)

    if args.asl:
        asl = np.load(args.asl, allow_pickle=False)
        assert np.array_equal(s04["row_id"], asl["row_id"]), "row_id mismatch with asl"
        evaluate_logits("Comparison Arm", logit(asl["probabilities"]), y, clusters, log_pos_w)


if __name__ == "__main__":
    main()
