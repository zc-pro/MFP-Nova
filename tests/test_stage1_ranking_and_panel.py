"""Unit tests for Stage 0 diagnostic panel and Stage 1 anchored PU ranking loss."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

from mfp_nova.multimodal import label_statistics
from mfp_nova.research_layers import anchored_pu_ranking_loss, observed_label_loss


def _load_panel_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_fine39_panel.py"
    spec = importlib.util.spec_from_file_location("diagnose_fine39_panel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_anchored_pu_ranking_focuses_on_high_wrong_and_protects_cooccurrence():
    # 4 labels: 0=pos, 1=high co-occurring unannotated, 2=zero co-occurring high wrong, 3=low unannotated
    y_train = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    stats = label_statistics(y_train, np.arange(len(y_train)), ["a", "b", "c", "d"])
    assert np.isclose(stats["cond_cooccurrence"][0][1], 2 / 3)
    assert stats["cond_cooccurrence"][0][2] == 0.0

    # Sample with positive label 0; labels 1 and 2 both have high logit +2.5, label 3 has low logit -3.0.
    logits_plain = torch.tensor([[0.2, 2.5, 2.5, -3.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss_plain = anchored_pu_ranking_loss(logits_plain, targets, stats, cooccur_beta=0.0)
    loss_plain.backward()
    g_plain = logits_plain.grad[0]
    assert torch.isfinite(loss_plain)
    # Positive gets negative grad (push up); high wrong gets much larger positive grad than low unannotated.
    assert g_plain[0] < 0
    assert g_plain[2] > 10 * g_plain[3]
    assert torch.isclose(g_plain[1], g_plain[2], atol=1e-6)

    # With co-occurrence protection, label 1 (co-occurring with 0) gets smaller push-down than zero-cooccur label 2.
    logits_co = torch.tensor([[0.2, 2.5, 2.5, -3.0]], requires_grad=True)
    loss_co = anchored_pu_ranking_loss(logits_co, targets, stats, cooccur_beta=1.5)
    loss_co.backward()
    g_co = logits_co.grad[0]
    assert torch.isfinite(loss_co)
    assert g_co[1] < g_co[2]
    assert g_co[2] > 0


def test_anchored_pu_ranking_scale_comparable_to_asl():
    rng = np.random.default_rng(0)
    y_train = (rng.random((64, 39)) < 0.06).astype(np.float32)
    y_train[:, 0] = 1.0
    stats = label_statistics(y_train, np.arange(64), [f"l{i}" for i in range(39)])
    logits = torch.randn(16, 39, requires_grad=True)
    targets = torch.from_numpy(y_train[:16])
    asl = observed_label_loss(logits, targets, stats, unknown_weight=0.3)
    rank = anchored_pu_ranking_loss(logits, targets, stats, cooccur_beta=1.5)
    assert torch.isfinite(asl) and torch.isfinite(rank)
    # Scaled by 1/C, rank loss should be within an order of magnitude of ASL.
    assert 0.05 < float(rank / asl.clamp_min(1e-6)) < 20.0


def test_diagnostic_panel_single_and_paired():
    panel = _load_panel_module()
    labels = ["anti_gram+", "anti_gram_", "anticancer__unknown", "antifungal__unknown", "neuropeptide__unknown"]
    targets = np.array([
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
    ], dtype=bool)
    base_probs = np.array([
        [0.8, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.3, 0.1, 0.1],  # nonempty_zero
        [0.8, 0.1, 0.2, 0.1, 0.1],  # partial multifunction
        [0.2, 0.1, 0.1, 0.1, 0.3],  # raw_empty -> top1 rescues
    ], dtype=np.float64)
    cand_probs = base_probs.copy()
    cand_probs[1, 0] = 0.2
    cand_probs[1, 2] = 0.85  # rescues nonempty_zero

    base = {"row_id": np.arange(4), "cluster_id": np.array([10, 11, 12, 13]), "targets": targets,
            "probabilities": base_probs, "label_names": labels}
    cand = dict(base, probabilities=cand_probs)

    res = panel.compare_exports(cand, base, threshold=0.5)
    assert res["baseline"]["nonempty_zero"] == 1
    assert res["baseline"]["raw_empty"] == 1
    assert res["candidate"]["nonempty_zero"] == 0
    assert res["paired_comparison"]["rescued_zero_rows"] == 1
    assert res["paired_comparison"]["raw_delta_pp"] > 0
