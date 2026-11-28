"""CPU offline probe for Fine39-Bio taxonomy, Partial-Label Bio-Eval, and PharmaField Probe 1 (255-Subset Energy Head)."""

import argparse
from pathlib import Path

import numpy as np
import torch

from mfp_nova.bio_taxonomy import build_fine39_bio_targets, granularity_mutex_matrix, partial_label_bio_eval
from mfp_nova.data import PeptideData
from mfp_nova.decisions import expected_jaccard_decode
from mfp_nova.multimodal import label_statistics
from mfp_nova.pharmafield_probe import ExactSubsetEnergyHead


def _logit(p):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _sample_jaccard(y_true, y_pred):
    inter = (y_true & y_pred).sum(axis=1)
    union = (y_true | y_pred).sum(axis=1)
    return float(np.where(union > 0, inter / np.maximum(union, 1), 0.0).mean()) * 100.0


def _cluster_folds(clusters, k=5, seed=42):
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    mapping = {int(c): int(i % k) for i, c in enumerate(uniq)}
    return np.array([mapping[int(c)] for c in clusters], dtype=np.int32)


def fit_3param_bce(logits, y_true, log_pos_w):
    z = torch.from_numpy(logits.astype(np.float32))
    y = torch.from_numpy(y_true.astype(np.float32))
    lw = torch.from_numpy(log_pos_w.astype(np.float32))[None, :]
    params = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([params], lr=0.5, max_iter=60, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        a = params[0].clamp(0.2, 5.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * z + params[1] - params[2] * lw, y)
        loss = loss + 1e-4 * (params - torch.tensor([1.0, 0.0, 0.0])).square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(params[0].clamp(0.2, 5.0).item()), float(params[1].item()), float(params[2].item())


def fit_subset_energy_head(logits, y_true, labels, log_pos_w, init_cal, max_iter=35):
    head = ExactSubsetEnergyHead(labels, log_pos_w, top_m=8)
    head.init_from_calibration(*init_cal)
    z = torch.from_numpy(logits.astype(np.float32))
    y = torch.from_numpy(y_true.astype(np.float32))
    # Precompute top_idx / j_true once when ranking under init_cal? No, forward() handles dynamic ranking;
    # Adam 60 steps at lr=0.06 is clean, fast (~1s/fold), and avoids LBFGS topk non-smooth line-search stalls.
    opt = torch.optim.Adam(head.parameters(), lr=0.06)
    for _ in range(max_iter):
        opt.zero_grad()
        loss = head.expected_jaccard_loss(z, y, reg_weight=1e-3)
        loss.backward()
        opt.step()
    return head


def audit_fine39_bio(data_root):
    bio = build_fine39_bio_targets(data_root, include_omitted_parent_positives=True)
    data = PeptideData(data_root, "fine39")
    print("=== Fine39-Bio (Y_bio, M_obs) Dataset Audit ===")
    for split in ("train", "val", "test"):
        idx = data.indices(split)
        y39 = bio["y_fine39"][idx]
        yb = bio["y_bio"][idx]
        mo = bio["m_obs"][idx]
        added_pos = int((yb - y39).sum())
        masked_cells = int((mo == 0).sum())
        masked_rows = int((mo == 0).any(axis=1).sum())
        print(f"  {split:5s}: rows={len(idx):5d} | restored/closure_pos=+{added_pos:5d} | "
              f"masked_subtype_cells={masked_cells:6d} across {masked_rows:5d} rows")


def evaluate_file(name, logits, y_true, clusters, labels, log_pos_w):
    p_raw = _sigmoid(logits)
    pred_05 = p_raw >= 0.50
    fb_05 = pred_05.copy()
    emp = ~fb_05.any(axis=1)
    if emp.any():
        fb_05[emp, p_raw[emp].argmax(axis=1)] = True
    bio_05 = partial_label_bio_eval(y_true, pred_05, labels)

    folds = _cluster_folds(clusters, k=5, seed=42)
    p_bce = np.zeros_like(p_raw)
    pred_energy = np.zeros_like(y_true, dtype=bool)
    for f in range(5):
        tr, va = folds != f, folds == f
        cal = fit_3param_bce(logits[tr], y_true[tr], log_pos_w)
        p_bce[va] = _sigmoid(cal[0] * logits[va] + cal[1] - cal[2] * log_pos_w[None, :])
        head = fit_subset_energy_head(logits[tr], y_true[tr], labels, log_pos_w, cal, max_iter=60)
        pred_energy[va] = head.decode(torch.from_numpy(logits[va].astype(np.float32)))

    mutex_mat = granularity_mutex_matrix(labels)
    pred_ej = expected_jaccard_decode(p_bce, top_m=8, mutex_matrix=None)
    pred_mej = expected_jaccard_decode(p_bce, top_m=8, mutex_matrix=mutex_mat)

    bio_mej = partial_label_bio_eval(y_true, pred_mej, labels)
    bio_en = partial_label_bio_eval(y_true, pred_energy, labels)

    print(f"\n[{name}]")
    print(f"  @0.50 Raw={_sample_jaccard(y_true, pred_05):.4f}% | FB={_sample_jaccard(y_true, fb_05):.4f}% | "
          f"BioAxis={bio_05['bio_axis_jaccard_pct']:.2f}% | PartialLabel={bio_05['partial_label_jaccard_pct']:.2f}% | "
          f"GranZero={bio_05['granularity_only_zero_rows']} ({bio_05['granularity_only_zero_pct_of_zero']:.1f}%) | "
          f"CrossZero={bio_05['cross_axis_zero_rows']}")
    print(f"  5-Fold CV 3-Param BCE + EJ       : ACC={_sample_jaccard(y_true, pred_ej):.4f}% (lps={pred_ej.sum(1).mean():.3f})")
    print(f"  5-Fold CV 3-Param BCE + Mutex-EJ : ACC={_sample_jaccard(y_true, pred_mej):.4f}% (lps={pred_mej.sum(1).mean():.3f}) | "
          f"BioAxis={bio_mej['bio_axis_jaccard_pct']:.2f}% | PartialLabel={bio_mej['partial_label_jaccard_pct']:.2f}% | Illegal={bio_mej['illegal_mutex_rows']}")
    print(f"  5-Fold CV 255-Subset Energy Head : ACC={_sample_jaccard(y_true, pred_energy):.4f}% (lps={pred_energy.sum(1).mean():.3f}) | "
          f"BioAxis={bio_en['bio_axis_jaccard_pct']:.2f}% | PartialLabel={bio_en['partial_label_jaccard_pct']:.2f}% | Illegal={bio_en['illegal_mutex_rows']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="Validation prediction NPZ (best_validation.npz or val_predictions.npz)")
    parser.add_argument("--data-root", default="data/therapeutic_2025")
    parser.add_argument("--extra-logit-key", default=None, help="Optional extra logit key inside NPZ (e.g. theraset_logits)")
    args = parser.parse_args()

    audit_fine39_bio(args.data_root)

    data = PeptideData(args.data_root, "fine39")
    tr_idx = data.indices("train")
    stats = label_statistics(data.y[tr_idx], data.clusters[tr_idx], data.labels)
    lw = np.log(np.maximum(np.asarray(stats["positive_weights"], dtype=np.float64), 1e-6))

    with np.load(args.npz, allow_pickle=False) as z:
        y = z["targets"].astype(bool)
        clusters = z["cluster_id"]
        labels = [str(x) for x in z["label_names"]]
        if labels != data.labels:
            raise ValueError("Label order mismatch")
        if args.extra_logit_key and args.extra_logit_key in z:
            evaluate_file(args.extra_logit_key, z[args.extra_logit_key], y, clusters, labels, lw)
        logits = z["logits"] if "logits" in z else _logit(z["probabilities"])
        evaluate_file(args.npz.stem, logits, y, clusters, labels, lw)


if __name__ == "__main__":
    main()
