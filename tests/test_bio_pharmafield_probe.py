"""Tests for Fine39-Bio taxonomy, Partial-Label eval, Mutex-EJ decoding, and Stage 4 DualPathMFP."""

from pathlib import Path

import numpy as np
import torch

from mfp_nova.bio_taxonomy import (
    build_fine39_bio_targets,
    cluster_balanced_positive_multipliers,
    granularity_mutex_matrix,
    partial_label_bio_eval,
)
from mfp_nova.data import PeptideData
from mfp_nova.decisions import apply_decision
from mfp_nova.models.dualpath_mfp import DualPathMFP
from mfp_nova.multimodal import collate_multimodal, label_statistics
from mfp_nova.pharmafield_probe import ExactSubsetEnergyHead, fine39_bio_supervision_loss

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "therapeutic_2025"


def _make_sample(length, input_dim=32, pep_dim=16, num_labels=39, seed=0):
    g = torch.Generator().manual_seed(seed)
    aa = torch.randint(1, 21, (length,), generator=g)
    d = torch.randint(0, 2, (length,), generator=g)
    mask = torch.ones(length, dtype=torch.bool)
    esm = torch.randn(length, input_dim, generator=g)
    esm_pep = torch.randn(length, pep_dim, generator=g)
    y = torch.zeros(num_labels)
    y[seed % num_labels] = 1.0
    y_bio = y.clone()
    m_obs = torch.ones(num_labels)
    # Mask a couple of subtypes as unobserved under __unknown.
    m_obs[(seed + 1) % num_labels] = 0.0
    return {
        "aa": aa,
        "d": d,
        "mask": mask,
        "esm": esm,
        "esm_peptide": esm_pep,
        "chemical": torch.tensor([2, 3], dtype=torch.int64),
        "residue_chem": torch.zeros(length, dtype=torch.int64),
        "terminal_caps": torch.tensor([True, False], dtype=torch.bool),
        "y": y,
        "y_bio": y_bio,
        "m_obs": m_obs,
        "row_id": torch.tensor(seed),
        "coords": torch.zeros(length, 3),
        "geometry": torch.zeros(length, 16),
        "struct_mask": torch.zeros(length, dtype=torch.bool),
        "dssp": torch.full((length,), 9, dtype=torch.int64),
        "edge_type": torch.zeros((length, length), dtype=torch.int64),
        "confidence_known": torch.zeros(length, dtype=torch.bool),
        "phys": torch.zeros(515),
        "phys_mask": torch.zeros(515, dtype=torch.bool),
        "struct_global": torch.zeros(25),
        "struct_global_mask": torch.zeros(25, dtype=torch.bool),
        "quality": torch.zeros(3, 6),
    }


def test_fine39_bio_targets_and_mutex_real_data():
    data = PeptideData(DATA_ROOT, "fine39")
    labels = data.labels
    mutex = granularity_mutex_matrix(labels)
    assert mutex.shape == (39, 39)
    assert int(mutex.sum() // 2) == 33

    # Verify strict 0 co-occurrence in all 33 mutex pairs across all original rows.
    orig_y = data.y[~data.augmented].astype(bool)
    co = (orig_y[:, :, None] & orig_y[:, None, :]) & mutex[None, :, :]
    assert not co.any(), "Unexpected co-occurrence in strict mutex pairs"

    bio = build_fine39_bio_targets(DATA_ROOT, include_omitted_parent_positives=True)
    ab_u = labels.index("antibacterial__unknown")
    ag_p = labels.index("anti_gram+")
    ag_m = labels.index("anti_gram_")
    am_u = labels.index("antimicrobial__unknown")

    only_ab_u = (data.y[:, ab_u] == 1) & (data.y[:, ag_p] == 0) & (data.y[:, ag_m] == 0)
    assert (bio["m_obs"][only_ab_u, ag_p] == 0.0).all()
    assert (bio["m_obs"][only_ab_u, ag_m] == 0.0).all()
    assert (bio["m_obs"][only_ab_u, ab_u] == 1.0).all()

    has_gp = data.y[:, ag_p] == 1
    assert (bio["y_bio"][has_gp, ab_u] == 1.0).all()
    assert (bio["y_bio"][has_gp, am_u] == 1.0).all()


def test_partial_label_bio_eval_and_mutex_expected_jaccard():
    labels = PeptideData(DATA_ROOT, "fine39").labels
    ab_u = labels.index("antibacterial__unknown")
    ag_p = labels.index("anti_gram+")
    ag_m = labels.index("anti_gram_")
    ac_u = labels.index("anticancer__unknown")
    ac_b = labels.index("antibreastcancer")
    ac_l = labels.index("antilungcancer")

    y = np.zeros((4, 39), dtype=bool)
    p = np.zeros((4, 39), dtype=bool)
    # Row 0: true antibacterial__unknown, pred anti_gram+ & anti_gram_ -> 1.0 partial-label (anti-gaming capped at 1 slot)
    y[0, ab_u] = True
    p[0, [ag_p, ag_m]] = True
    # Row 1: true anti_gram+, pred antibacterial__unknown -> 0.5 coarse credit (granularity-only zero)
    y[1, ag_p] = True
    p[1, ab_u] = True
    # Row 2: true anticancer__unknown, pred BOTH anticancer__unknown & antibreastcancer -> penalized (< 1.0)
    y[2, ac_u] = True
    p[2, [ac_u, ac_b]] = True
    # Row 3: true antibreastcancer, pred antilungcancer -> 0.0 (within anticancer axis, wrong subtype)
    y[3, ac_b] = True
    p[3, ac_l] = True

    res = partial_label_bio_eval(y, p, labels)
    assert res["granularity_only_zero_rows"] == 3  # rows 0, 1, 3 share biological axis despite 0 literal Jaccard
    assert res["cross_axis_zero_rows"] == 0
    assert res["illegal_mutex_rows"] == 1
    assert np.isclose(res["partial_label_jaccard_pct"], (1.0 + 0.5 + 0.5 + 0.0) / 4.0 * 100.0)

    # Mutex-EJ must forbid {antibacterial__unknown, anti_gram+} even when both probabilities are high.
    probs = np.full((1, 39), 0.01, dtype=np.float64)
    probs[0, ab_u] = 0.72
    probs[0, ag_p] = 0.70
    decision = {
        "rule": "prior_calibrated_expected_jaccard",
        "task": "fine39",
        "labels": labels,
        "alpha": 1.0,
        "beta0": 0.0,
        "beta1": 0.0,
        "log_positive_weights": [0.0] * 39,
        "top_m": 8,
        "granularity_mutex": True,
    }
    pred = apply_decision(probs, labels, "fine39", decision)
    assert not (pred[0, ab_u] and pred[0, ag_p])


def test_cluster_balanced_multipliers_preserve_positive_mass():
    data = PeptideData(DATA_ROOT, "fine39")
    tr = data.indices("train")
    mult = cluster_balanced_positive_multipliers(data, alpha=0.35)
    for col in range(len(data.labels)):
        pos = tr[data.y[tr, col] > 0]
        assert np.isclose(mult[pos, col].sum(), float(len(pos)), rtol=1e-4)


def test_stage4_dualpath_bio_head_and_biophysics_v2_invariance_and_isolation():
    labels = PeptideData(DATA_ROOT, "fine39").labels
    rng = np.random.default_rng(7)
    y = (rng.random((64, 39)) < 0.15).astype(np.float32)
    y[:, 0] = 1.0
    stats = label_statistics(y, np.arange(64), labels)

    torch.manual_seed(11)
    model = DualPathMFP(
        input_dim=32, num_labels=39, vocab_size=16, statistics=stats,
        peptide_input_dim=16, dim=64, heads=4, cnn_channels=16, pair_dim=16,
        b_seq_depth=2, b_prop_depth=1, b_latent_count=8, b_latent_depth=2,
        b_decoder_depth=2, full_branch_rdrop=True,
        biophysics_v2=True, use_bio_head=True, bio_statistics=stats,
    )
    model.eval()
    short_s = _make_sample(9, seed=1)
    long_s = _make_sample(46, seed=2)
    with torch.no_grad():
        out_solo = model(collate_multimodal([short_s]))
        out_mixed = model(collate_multimodal([short_s, long_s]))
    for key in ("path_a_logits", "path_b_logits", "path_a_bio_logits", "path_b_bio_logits", "logits"):
        diff = (out_solo[key][0] - out_mixed[key][0]).abs().max().item()
        assert diff < 1e-5, f"Padding invariance failed on {key}: {diff}"

    # Verify exact zero gradient on masked m_obs == 0 cells in bio loss.
    bio_logits = torch.zeros((2, 39), requires_grad=True)
    y_bio = torch.zeros((2, 39))
    y_bio[0, 0] = 1.0
    m_obs = torch.ones((2, 39))
    m_obs[0, 5] = 0.0
    loss_bio = fine39_bio_supervision_loss(bio_logits, y_bio, m_obs, stats, unknown_weight=0.35)
    loss_bio.backward()
    assert bio_logits.grad[0, 5].item() == 0.0
    assert bio_logits.grad[0, 6].item() != 0.0

    # Verify exact zero gradient crosstalk between Path A (+ path_a_gran) and Path B (+ bio_head).
    batch = collate_multimodal([_make_sample(12, seed=3), _make_sample(20, seed=4)])
    model.train()
    torch.manual_seed(99)
    l1, _ = model.compute_loss(batch, {"unlabeled_weight": 0.3, "b_unlabeled_weight": 0.45, "direct": 0.2,
                                       "evidence": 0.1, "b_latent": 0.15, "bio_weight": 0.5,
                                       "modality_dropout": 0.0, "rdrop": 0.1, "sym_rank": 0.0})
    l1.backward()
    ga1 = [p.grad.detach().clone() for p in model.branch_parameter_groups()[0] if p.grad is not None]
    model.zero_grad(set_to_none=True)

    torch.manual_seed(99)
    l2, _ = model.compute_loss(batch, {"unlabeled_weight": 0.3, "b_unlabeled_weight": 0.45, "direct": 0.2,
                                       "evidence": 0.1, "b_latent": 0.15, "bio_weight": 0.5,
                                       "modality_dropout": 0.0, "rdrop": 0.1, "sym_rank": 5.0})
    l2.backward()
    ga2 = [p.grad.detach().clone() for p in model.branch_parameter_groups()[0] if p.grad is not None]
    assert len(ga1) == len(ga2) and max((x - y_g).abs().max().item() for x, y_g in zip(ga1, ga2)) == 0.0

    # Smoke-check ExactSubsetEnergyHead forward/loss/decode shapes & mutex enforcement.
    ehead = ExactSubsetEnergyHead(labels, [0.0] * 39, top_m=8)
    eloss = ehead.expected_jaccard_loss(out_mixed["logits"], batch["y"])
    assert torch.isfinite(eloss)
    decoded = ehead.decode(out_mixed["logits"])
    assert decoded.shape == (2, 39)
    assert not ((decoded[:, :, None] & decoded[:, None, :]) & granularity_mutex_matrix(labels)[None, :, :]).any()


def test_stage35_clean_bottleneck_pure_orth_and_mask_token():
    labels = PeptideData(DATA_ROOT, "fine39").labels
    rng = np.random.default_rng(13)
    y = (rng.random((32, 39)) < 0.15).astype(np.float32)
    y[:, 0] = 1.0
    stats = label_statistics(y, np.arange(32), labels)

    model = DualPathMFP(
        input_dim=32, num_labels=39, vocab_size=16, statistics=stats,
        peptide_input_dim=16, orth_input_dim=24, dim=64, heads=4, cnn_channels=16, pair_dim=16,
        b_seq_depth=2, b_prop_depth=1, b_latent_count=8, b_latent_depth=2,
        b_decoder_depth=0, full_branch_rdrop=True,
        clean_span_mask=True, b_use_peptide_esm=False,
    )
    assert model.path_b.mask_token is not None
    assert model.path_b.plm_proj.in_features == 24

    s1 = {**_make_sample(9, seed=5), "esm_orth": torch.randn(9, 24)}
    s2 = {**_make_sample(30, seed=6), "esm_orth": torch.randn(30, 24)}
    model.eval()
    with torch.no_grad():
        o1 = model(collate_multimodal([s1]))
        o2 = model(collate_multimodal([s1, s2]))
    assert (o1["path_b_logits"][0] - o2["path_b_logits"][0]).abs().max().item() < 1e-5

    model.train()
    loss, _ = model.compute_loss(
        collate_multimodal([s1, s2]),
        {"unlabeled_weight": 0.3, "b_unlabeled_weight": 0.45, "direct": 0.2, "evidence": 0.1,
         "modality_dropout": 0.0, "rdrop": 0.1, "sym_rank": 0.04},
    )
    loss.backward()
    assert model.path_b.mask_token.grad is not None

