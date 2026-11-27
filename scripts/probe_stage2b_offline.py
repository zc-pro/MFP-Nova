"""Zero-cost CPU offline probe for Stage 2/2b: 5-fold cluster-CV calibration + Expected Jaccard decoding + cross-pairing."""

import argparse
import numpy as np
from pathlib import Path


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


def fit_temp_bias(logits, y_true, temps=(0.6, 0.75, 0.9, 1.0, 1.15, 1.3), biases=(-0.3, -0.15, 0.0, 0.15, 0.3)):
    best = (-1.0, 1.0, 0.0)
    for t in temps:
        for b in biases:
            p = sigmoid(logits / t + b)
            _, fb, _, _ = eval_threshold(y_true, p, 0.5)
            if fb > best[0]:
                best = (fb, t, b)
    return best[1], best[2]


def expected_jaccard_decode(probs, top_m=8):
    """Exact Bayes-optimal Expected Sample Jaccard over top-M candidates per sample (non-empty)."""
    n, c = probs.shape
    m = min(top_m, c)
    order = np.argsort(-probs, axis=1)
    top_idx = order[:, :m]
    p_top = np.take_along_axis(probs, top_idx, axis=1)
    rem_mass = np.take_along_axis(probs, order[:, m:], axis=1).sum(axis=1) if c > m else np.zeros(n)

    # All 2^m truth states Y in {0,1}^m and all non-empty candidate subsets S in {0,1}^m \ {0}
    states = np.array([[(s >> j) & 1 for j in range(m)] for s in range(1 << m)], dtype=np.float32)  # [2^m, m]
    cand_s = states[1:]  # [2^m - 1, m]
    inter = states @ cand_s.T  # [2^m, 2^m-1]
    union_base = states.sum(axis=1, keepdims=True) + cand_s.sum(axis=1, keepdims=True).T - inter

    pred = np.zeros((n, c), dtype=bool)
    for start in range(0, n, 1024):
        end = min(start + 1024, n)
        pb = p_top[start:end]  # [B, m]
        # P(Y | x) = prod_j p_j^{y_j} (1 - p_j)^{1 - y_j}
        log_p = np.log(np.clip(pb, 1e-6, 1.0 - 1e-6))
        log_1mp = np.log(np.clip(1.0 - pb, 1e-6, 1.0 - 1e-6))
        py = np.exp(log_p @ states.T + log_1mp @ (1.0 - states).T)  # [B, 2^m]
        rem = rem_mass[start:end, None, None]  # [B, 1, 1]
        j_mat = np.where(union_base[None] + rem > 0, inter[None] / np.maximum(union_base[None] + rem, 1e-6), 0.0)
        exp_j = np.einsum("by,bys->bs", py, j_mat)  # [B, 2^m-1]
        best_s = cand_s[exp_j.argmax(axis=1)] > 0.5  # [B, m]
        rows = np.arange(start, end)[:, None]
        pred[rows, top_idx[start:end]] = best_s
    return pred


def evaluate_logits(name, logits, y_true, clusters):
    p_raw = sigmoid(logits)
    r50, f50, l50, e50 = eval_threshold(y_true, p_raw, 0.50)
    folds = cluster_folds(clusters, k=5, seed=42)
    p_cal = np.zeros_like(p_raw)
    for f in range(5):
        tr, va = folds != f, folds == f
        t, b = fit_temp_bias(logits[tr], y_true[tr])
        p_cal[va] = sigmoid(logits[va] / t + b)
    rcal, fcal, lcal, _ = eval_threshold(y_true, p_cal, 0.50)
    ej_pred = expected_jaccard_decode(p_cal, top_m=8)
    ej_acc = sample_jaccard(y_true, ej_pred)
    ej_lps = float(ej_pred.sum(axis=1).mean())
    print(f"[{name}]")
    print(f"  @0.50 Raw={r50:.4f}% | Fallback={f50:.4f}% (lps={l50:.3f}, empty={e50})")
    print(f"  5-Fold Cluster-CV Calibrated @0.50 -> Raw={rcal:.4f}% | Fallback={fcal:.4f}% (lps={lcal:.3f})")
    print(f"  5-Fold Cluster-CV + Top-8 Expected Jaccard -> ACC={ej_acc:.4f}% (lps={ej_lps:.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym04", required=True, help="Path to dp2_2path_sym04 best_validation.npz")
    parser.add_argument("--theraset", required=True, help="Path to theraset_jacc_100ep best_validation.npz")
    parser.add_argument("--asl", default=None, help="Optional path to stage2 dualpath_asl best_validation.npz")
    args = parser.parse_args()

    s04 = np.load(args.sym04, allow_pickle=False)
    th = np.load(args.theraset, allow_pickle=False)
    assert np.array_equal(s04["row_id"], th["row_id"]), "row_id mismatch"
    y = s04["targets"].astype(bool)
    clusters = s04["cluster_id"]

    za_s04 = logit(s04["path_a_probabilities"])
    zb_s04 = logit(s04["path_b_probabilities"])
    z_th = logit(th["probabilities"])

    evaluate_logits("dp2_2path_sym04 (Current 2-Path)", 0.5 * (za_s04 + zb_s04), y, clusters)
    evaluate_logits("TheraSet(60.82%) + dp2_2path_sym04-PathB(59.20%) [Undiluted PathA Proxy]", 0.5 * (z_th + zb_s04), y, clusters)

    if args.asl:
        asl = np.load(args.asl, allow_pickle=False)
        assert np.array_equal(s04["row_id"], asl["row_id"]), "row_id mismatch with asl"
        za_asl = logit(asl["path_a_probabilities"])
        zb_asl = logit(asl["path_b_probabilities"])
        evaluate_logits("dualpath_asl (Stage 2)", 0.5 * (za_asl + zb_asl), y, clusters)


if __name__ == "__main__":
    main()
