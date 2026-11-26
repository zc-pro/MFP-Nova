"""Strict verification tests for Stage 2 DualPath-MFP:
1. Mixed-length batch padding invariance (Path A, Path B, fused logits).
2. Exact zero gradient crosstalk between Path A and Path B.
3. Symmetric PU ranking loss fixed-point balance, zero tail gradient, and overflow safety.
4. Pure-sequence / missing-modality transfer resilience.
"""

import numpy as np
import torch

from mfp_nova.models.dualpath_mfp import DualPathMFP
from mfp_nova.multimodal import collate_multimodal, label_statistics
from mfp_nova.research_layers import symmetric_pu_ranking_loss


def _dummy_statistics(num_labels=8):
    rng = np.random.default_rng(42)
    y = (rng.random((64, num_labels)) < 0.2).astype(np.float32)
    y[:, 0] = 1.0
    return label_statistics(y, np.arange(64), [f"lbl_{i}" for i in range(num_labels)])


def _make_sample(length, input_dim=32, pep_dim=16, num_labels=8, seed=0, with_aux=True):
    g = torch.Generator().manual_seed(seed)
    aa = torch.randint(1, 21, (length,), generator=g)
    d = torch.randint(0, 2, (length,), generator=g)
    mask = torch.ones(length, dtype=torch.bool)
    esm = torch.randn(length, input_dim, generator=g)
    esm_pep = torch.randn(length, pep_dim, generator=g)
    y = torch.zeros(num_labels)
    y[seed % num_labels] = 1.0
    sample = {
        "aa": aa,
        "d": d,
        "mask": mask,
        "esm": esm,
        "esm_peptide": esm_pep,
        "chemical": torch.tensor([2, 3], dtype=torch.int64),
        "residue_chem": torch.zeros(length, dtype=torch.int64),
        "terminal_chemical": torch.tensor([True, False], dtype=torch.bool),
        "y": y,
        "row_id": torch.tensor(seed),
    }
    if with_aux:
        struct_mask = torch.ones(length, dtype=torch.bool)
        coords = torch.randn(length, 3, generator=g) * 3.0
        geom = torch.randn(length, 16, generator=g)
        geom[:, 9] = torch.rand(length, generator=g)
        edge_type = torch.zeros((length, length), dtype=torch.int64)
        if length > 1:
            idx = torch.arange(length - 1)
            edge_type[idx, idx + 1] = 1
            edge_type[idx + 1, idx] = 1
        sample.update(
            coords=coords,
            geometry=geom,
            struct_mask=struct_mask,
            dssp=torch.randint(0, 8, (length,), generator=g),
            edge_type=edge_type,
            confidence_known=torch.ones(length, dtype=torch.bool),
            phys=torch.randn(515, generator=g),
            phys_mask=torch.ones(515, dtype=torch.bool),
            struct_global=torch.randn(25, generator=g),
            struct_global_mask=torch.ones(25, dtype=torch.bool),
            quality=torch.ones(3, 6),
        )
    else:
        sample.update(
            coords=torch.zeros(length, 3),
            geometry=torch.zeros(length, 16),
            struct_mask=torch.zeros(length, dtype=torch.bool),
            dssp=torch.full((length,), 9, dtype=torch.int64),
            edge_type=torch.zeros((length, length), dtype=torch.int64),
            confidence_known=torch.zeros(length, dtype=torch.bool),
            phys=torch.zeros(515),
            phys_mask=torch.zeros(515, dtype=torch.bool),
            struct_global=torch.zeros(25),
            struct_global_mask=torch.zeros(25, dtype=torch.bool),
            quality=torch.zeros(3, 6),
        )
    return sample


def _small_dualpath(stats, input_dim=32, pep_dim=16, num_labels=8):
    return DualPathMFP(
        input_dim=input_dim, num_labels=num_labels, vocab_size=16, statistics=stats,
        peptide_input_dim=pep_dim, dim=64, heads=4, cnn_channels=16, pair_dim=16,
        b_seq_depth=2, b_prop_depth=1, b_latent_count=8, b_latent_depth=2,
    )


def test_padding_invariance_mixed_lengths():
    torch.manual_seed(123)
    stats = _dummy_statistics(8)
    model = _small_dualpath(stats).eval()

    short_s = _make_sample(9, seed=1, with_aux=True)
    long_s = _make_sample(46, seed=2, with_aux=True)

    solo_batch = collate_multimodal([short_s])
    mixed_batch = collate_multimodal([short_s, long_s])

    with torch.no_grad():
        out_solo = model(solo_batch)
        out_mixed = model(mixed_batch)

    for key in ("path_a_logits", "path_b_logits", "logits"):
        diff = (out_solo[key][0] - out_mixed[key][0]).abs().max().item()
        assert diff < 1e-5, f"Padding invariance violated on {key}: max_abs_diff={diff}"


def test_exact_zero_gradient_crosstalk_between_paths():
    stats = _dummy_statistics(8)
    batch = collate_multimodal([_make_sample(12, seed=3), _make_sample(20, seed=4)])

    # Run 1: sym_rank = 0.0
    torch.manual_seed(777)
    m1 = _small_dualpath(stats)
    m1.train()
    torch.manual_seed(999)
    loss1, _ = m1.compute_loss(batch, {"unlabeled_weight": 0.3, "direct": 0.2, "evidence": 0.1,
                                       "modality_dropout": 0.0, "rdrop": 0.0, "sym_rank": 0.0})
    loss1.backward()
    grads_a_1 = {n: p.grad.detach().clone() for n, p in m1.path_a.named_parameters() if p.grad is not None}
    grads_b_1 = {n: p.grad.detach().clone() for n, p in m1.path_b.named_parameters() if p.grad is not None}

    # Run 2: identical RNG and weights, but change Path B's sym_rank weight drastically to 5.0
    torch.manual_seed(777)
    m2 = _small_dualpath(stats)
    m2.train()
    torch.manual_seed(999)
    loss2, _ = m2.compute_loss(batch, {"unlabeled_weight": 0.3, "direct": 0.2, "evidence": 0.1,
                                       "modality_dropout": 0.0, "rdrop": 0.0, "sym_rank": 5.0})
    loss2.backward()
    grads_a_2 = {n: p.grad.detach().clone() for n, p in m2.path_a.named_parameters() if p.grad is not None}
    grads_b_2 = {n: p.grad.detach().clone() for n, p in m2.path_b.named_parameters() if p.grad is not None}

    # Path A gradients must be bitwise identical (0.0 crosstalk), while Path B gradients change.
    max_diff_a = max((grads_a_1[k] - grads_a_2[k]).abs().max().item() for k in grads_a_1)
    max_diff_b = max((grads_b_1[k] - grads_b_2[k]).abs().max().item() for k in grads_b_1)
    assert max_diff_a == 0.0, f"Gradient leaked into Path A: max_diff_a={max_diff_a}"
    assert max_diff_b > 1e-4, "Path B gradient did not respond to sym_rank"

    # Also verify external loss on fused output['logits'] produces zero grad on both branches (stop_grad).
    m2.zero_grad()
    out = m2(batch)
    assert not out["logits"].requires_grad


def test_symmetric_pu_ranking_zero_point_balance_and_tail_safety():
    stats = _dummy_statistics(10)
    # Test both single-positive (|P|=1) and multi-positive (|P|=3) at symmetric margin s_pos=+0.5, s_neg=-0.5.
    for n_pos in (1, 3):
        targets = torch.zeros((1, 10))
        targets[0, :n_pos] = 1.0
        logits = torch.full((1, 10), -4.0)
        logits[0, :n_pos] = 0.5            # recorded positives at +0.5
        logits[0, n_pos:n_pos + 3] = -0.5  # top-3 unannotated at -0.5; remaining tail at -4.0
        logits.requires_grad_(True)

        loss = symmetric_pu_ranking_loss(
            logits, targets, stats, top_k=3, pos_margin=0.5, neg_margin=-0.5, gamma=2.0, cooccur_beta=0.0
        )
        loss.backward()
        g = logits.grad[0]
        up_mass = (-g[:n_pos]).sum().item()
        down_mass = g[n_pos:n_pos + 3].sum().item()
        tail_grad = g[n_pos + 3:].abs().max().item()

        # Total upward and downward gradient mass must be 1:1 balanced (= 0.5 / C = 0.05).
        assert np.isclose(up_mass, 0.5 / 10.0, atol=1e-6)
        assert np.isclose(down_mass, 0.5 / 10.0, atol=1e-6)
        assert tail_grad == 0.0, f"Non-topK tail received non-zero grad: {tail_grad}"

    # Extreme logits (+/- 500) must never overflow to inf/NaN.
    ext_logits = torch.tensor([[500.0, -500.0, 500.0, -500.0] + [0.0] * 6], requires_grad=True)
    ext_targets = torch.tensor([[1.0, 1.0, 0.0, 0.0] + [0.0] * 6])
    ext_loss = symmetric_pu_ranking_loss(ext_logits, ext_targets, stats, top_k=3)
    ext_loss.backward()
    assert torch.isfinite(ext_loss)
    assert torch.isfinite(ext_logits.grad).all()


def test_pure_sequence_missing_modality_transfer():
    stats = _dummy_statistics(8)
    model = _small_dualpath(stats)
    # Include a 2-residue ultra-short peptide with all structure and external physchem missing.
    batch = collate_multimodal([
        _make_sample(2, seed=10, with_aux=False),
        _make_sample(15, seed=11, with_aux=False),
    ])
    batch.pop("terminal_chemical", None)
    model.train()
    loss, terms, out = model.compute_loss(
        batch, {"unlabeled_weight": 0.3, "direct": 0.2, "evidence": 0.1,
                "modality_dropout": 0.2, "rdrop": 0.1, "sym_rank": 0.1},
        return_output=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(out["logits"]).all()
