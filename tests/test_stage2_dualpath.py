"""Strict verification tests for Stage 2 & Stage 3 DualPath-MFP:
1. Mixed-length batch padding invariance (Path A, Path B with SetDecoder, heterogeneous orth PLM, fused logits).
2. Exact zero gradient crosstalk between Path A and Path B + full per-branch R-Drop scaling.
3. Symmetric PU ranking loss fixed-point balance, zero tail gradient, and overflow safety.
4. Pure-sequence / missing-modality transfer resilience + non-destructive terminal_caps parity.
5. ProstT5 (<AA2fold>) and ProtT5 residue coordinate slicing & token boundary assertions.
6. Prior-calibrated Top-8 Expected Jaccard decision rule.
"""

import numpy as np
import pytest
import torch

from mfp_nova.decisions import apply_decision
from mfp_nova.models.dualpath_mfp import DualPathMFP
from mfp_nova.multimodal import collate_multimodal, extract_terminal_caps, label_statistics, localize_terminal_chemistry
from mfp_nova.research_layers import symmetric_pu_ranking_loss


def _dummy_statistics(num_labels=8):
    rng = np.random.default_rng(42)
    y = (rng.random((64, num_labels)) < 0.2).astype(np.float32)
    y[:, 0] = 1.0
    return label_statistics(y, np.arange(64), [f"lbl_{i}" for i in range(num_labels)])


def _make_sample(length, input_dim=32, pep_dim=16, orth_dim=0, num_labels=8, seed=0, with_aux=True):
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
        "terminal_caps": torch.tensor([True, False], dtype=torch.bool),
        "y": y,
        "row_id": torch.tensor(seed),
    }
    if orth_dim > 0:
        sample["esm_orth"] = torch.randn(length, orth_dim, generator=g) * 0.2
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


def _small_dualpath(stats, input_dim=32, pep_dim=16, orth_dim=0, num_labels=8,
                    b_decoder_depth=2, full_branch_rdrop=True, use_aa34_path=False):
    return DualPathMFP(
        input_dim=input_dim, num_labels=num_labels, vocab_size=16, statistics=stats,
        peptide_input_dim=pep_dim, orth_input_dim=orth_dim, dim=64, heads=4, cnn_channels=16, pair_dim=16,
        b_seq_depth=2, b_prop_depth=1, b_latent_count=8, b_latent_depth=2,
        b_decoder_depth=b_decoder_depth, full_branch_rdrop=full_branch_rdrop,
        use_aa34_path=use_aa34_path,
    )


@pytest.mark.parametrize("orth_dim,b_dec", [(0, 0), (0, 2), (24, 2)])
def test_padding_invariance_mixed_lengths(orth_dim, b_dec):
    torch.manual_seed(123)
    stats = _dummy_statistics(8)
    model = _small_dualpath(stats, orth_dim=orth_dim, b_decoder_depth=b_dec).eval()

    short_s = _make_sample(9, orth_dim=orth_dim, seed=1, with_aux=True)
    long_s = _make_sample(46, orth_dim=orth_dim, seed=2, with_aux=True)

    solo_batch = collate_multimodal([short_s])
    mixed_batch = collate_multimodal([short_s, long_s])

    with torch.no_grad():
        out_solo = model(solo_batch)
        out_mixed = model(mixed_batch)

    for key in ("path_a_logits", "path_b_logits", "logits"):
        diff = (out_solo[key][0] - out_mixed[key][0]).abs().max().item()
        assert diff < 1e-5, f"Padding invariance violated on {key} (orth={orth_dim}, dec={b_dec}): max_abs_diff={diff}"


def test_exact_zero_gradient_crosstalk_and_full_branch_rdrop():
    stats = _dummy_statistics(8)
    batch = collate_multimodal([_make_sample(12, seed=3), _make_sample(20, seed=4)])

    torch.manual_seed(777)
    m1 = _small_dualpath(stats, b_decoder_depth=2, full_branch_rdrop=True)
    m1.train()
    torch.manual_seed(999)
    loss1, terms1 = m1.compute_loss(batch, {"unlabeled_weight": 0.3, "direct": 0.2, "evidence": 0.1, "b_latent": 0.15,
                                            "modality_dropout": 0.0, "rdrop": 0.1, "sym_rank": 0.0})
    loss1.backward()
    grads_a_1 = {n: p.grad.detach().clone() for n, p in m1.path_a.named_parameters() if p.grad is not None}
    grads_b_1 = {n: p.grad.detach().clone() for n, p in m1.path_b.named_parameters() if p.grad is not None}

    torch.manual_seed(777)
    m2 = _small_dualpath(stats, b_decoder_depth=2, full_branch_rdrop=True)
    m2.train()
    torch.manual_seed(999)
    loss2, terms2 = m2.compute_loss(batch, {"unlabeled_weight": 0.3, "direct": 0.2, "evidence": 0.1, "b_latent": 0.15,
                                            "modality_dropout": 0.0, "rdrop": 0.1, "sym_rank": 5.0})
    loss2.backward()
    grads_a_2 = {n: p.grad.detach().clone() for n, p in m2.path_a.named_parameters() if p.grad is not None}
    grads_b_2 = {n: p.grad.detach().clone() for n, p in m2.path_b.named_parameters() if p.grad is not None}

    max_diff_a = max((grads_a_1[k] - grads_a_2[k]).abs().max().item() for k in grads_a_1)
    max_diff_b = max((grads_b_1[k] - grads_b_2[k]).abs().max().item() for k in grads_b_1)
    assert max_diff_a == 0.0, f"Gradient leaked into Path A: max_diff_a={max_diff_a}"
    assert max_diff_b > 1e-4, "Path B gradient did not respond to sym_rank"
    assert "b_latent" in terms2

    # Under identical RNG (both views identical), full_branch_rdrop=True adds exact +0.1 * mean_kl over legacy half R-Drop.
    torch.manual_seed(777)
    m_half = _small_dualpath(stats, b_decoder_depth=2, full_branch_rdrop=False)
    m_half.train()
    torch.manual_seed(999)
    loss_half, _ = m_half.compute_loss(batch, {"unlabeled_weight": 0.3, "direct": 0.2, "evidence": 0.1, "b_latent": 0.15,
                                               "modality_dropout": 0.0, "rdrop": 0.1, "sym_rank": 0.0})
    assert np.isclose(float(loss1 - loss_half), 0.1 * float(terms1["rdrop"]), atol=1e-6)


def test_symmetric_pu_ranking_zero_point_balance_and_tail_safety():
    stats = _dummy_statistics(10)
    for n_pos in (1, 3):
        targets = torch.zeros((1, 10))
        targets[0, :n_pos] = 1.0
        logits = torch.full((1, 10), -4.0)
        logits[0, :n_pos] = 0.5
        logits[0, n_pos:n_pos + 3] = -0.5
        logits.requires_grad_(True)

        loss = symmetric_pu_ranking_loss(
            logits, targets, stats, top_k=3, pos_margin=0.5, neg_margin=-0.5, gamma=2.0, cooccur_beta=0.0
        )
        loss.backward()
        g = logits.grad[0]
        up_mass = (-g[:n_pos]).sum().item()
        down_mass = g[n_pos:n_pos + 3].sum().item()
        tail_grad = g[n_pos + 3:].abs().max().item()

        assert np.isclose(up_mass, 0.5 / 10.0, atol=1e-6)
        assert np.isclose(down_mass, 0.5 / 10.0, atol=1e-6)
        assert tail_grad == 0.0, f"Non-topK tail received non-zero grad: {tail_grad}"

    ext_logits = torch.tensor([[500.0, -500.0, 500.0, -500.0] + [0.0] * 6], requires_grad=True)
    ext_targets = torch.tensor([[1.0, 1.0, 0.0, 0.0] + [0.0] * 6])
    ext_loss = symmetric_pu_ranking_loss(ext_logits, ext_targets, stats, top_k=3)
    ext_loss.backward()
    assert torch.isfinite(ext_loss)
    assert torch.isfinite(ext_logits.grad).all()


def test_pure_sequence_missing_modality_and_nondestructive_terminal_caps():
    tokens = ["[ac]", "C-terminal modification:Amidation", "disulfide"]
    caps = extract_terminal_caps(tokens)
    assert caps == [True, True]
    assert tokens == ["[ac]", "C-terminal modification:Amidation", "disulfide"]
    rem, loc_caps = localize_terminal_chemistry(tokens)
    assert loc_caps == caps and "[ac]" not in rem

    stats = _dummy_statistics(8)
    model = _small_dualpath(stats, orth_dim=20, b_decoder_depth=2)
    batch = collate_multimodal([
        _make_sample(2, orth_dim=20, seed=10, with_aux=False),
        _make_sample(15, orth_dim=20, seed=11, with_aux=False),
    ])
    model.train()
    loss, _, out = model.compute_loss(
        batch, {"unlabeled_weight": 0.3, "direct": 0.2, "evidence": 0.1, "b_latent": 0.15,
                "modality_dropout": 0.2, "rdrop": 0.1, "sym_rank": 0.04},
        return_output=True,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(out["logits"]).all()
    assert model.path_b.mask_token.grad is not None


def test_t5_prostt5_encoder_slice_alignment():
    from mfp_nova.encoder import ALPHABET, FrozenEncoder

    class DummyTokenizer:
        def __init__(self, prost):
            self.vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
            for idx, aa in enumerate(ALPHABET, start=3):
                self.vocab[f"\u2581{aa}"] = idx
            if prost:
                self.vocab["<AA2fold>"] = 99
            self.prost = prost

        def get_vocab(self):
            return dict(self.vocab)

        def __call__(self, texts, padding=True, truncation=False, return_tensors="pt"):
            rows = []
            for txt in texts:
                parts = txt.split()
                ids = [99 if p == "<AA2fold>" else self.vocab[f"\u2581{p}"] for p in parts] + [1]
                rows.append(ids)
            max_l = max(len(r) for r in rows)
            inp = torch.zeros((len(rows), max_l), dtype=torch.long)
            att = torch.zeros((len(rows), max_l), dtype=torch.long)
            for i, r in enumerate(rows):
                inp[i, :len(r)] = torch.tensor(r)
                att[i, :len(r)] = 1
            return {"input_ids": inp, "attention_mask": att}

    class DummyT5(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            # Encode token ID directly into hidden state channel 0 so we can verify exact 1-to-1 residue alignment!
            h = input_ids.unsqueeze(-1).float().repeat(1, 1, 8)
            return type("Out", (), {"last_hidden_state": h})()

    for prost in (False, True):
        enc = object.__new__(FrozenEncoder)
        enc.device = torch.device("cpu")
        enc.max_length = 80
        enc.is_t5 = True
        enc.is_prost_t5 = prost
        enc.tokenizer = DummyTokenizer(prost)
        enc.t5_aa_ids = {enc.tokenizer.vocab[f"\u2581{aa}"] for aa in ALPHABET}
        enc.model = DummyT5()

        seqs = ["ACWY", "KLMNPQR"]
        states, mask = enc.encode(seqs)
        for j, seq in enumerate(seqs):
            sliced = states[j, mask[j], 0].long().tolist()
            expected = [enc.tokenizer.vocab[f"\u2581{aa}"] for aa in seq]
            assert sliced == expected, f"Prost={prost} misaligned slice: {sliced} != {expected}"


def test_prior_calibrated_expected_jaccard_decision():
    labels = [f"c{i}" for i in range(6)]
    probs = np.array([
        [0.92, 0.41, 0.05, 0.02, 0.01, 0.01],  # confident singleton -> should predict only c0
        [0.032, 0.030, 0.028, 0.025, 0.020, 0.015],  # flat low confidence -> must predict exact singleton {c0}, no inflation
    ])
    decision = {
        "rule": "prior_calibrated_expected_jaccard",
        "task": "fine39",
        "labels": labels,
        "alpha": 1.0,
        "beta0": 0.0,
        "beta1": 0.0,
        "log_positive_weights": [0.0] * 6,
        "top_m": 6,
    }
    pred = apply_decision(probs, labels, "fine39", decision)
    assert pred.shape == (2, 6)
    assert pred[0].tolist() == [True, False, False, False, False, False]
    assert pred[1].tolist() == [True, False, False, False, False, False]


def test_tripath_aa34_gradient_isolation_and_independent_clipping():
    stats = _dummy_statistics(8)
    batch = collate_multimodal([_make_sample(10, seed=12), _make_sample(18, seed=13)])
    model = _small_dualpath(stats, use_aa34_path=True)
    model.train()
    groups = model.branch_parameter_groups()
    assert len(groups) == 3

    loss, terms, out = model.compute_loss(
        batch, {"unlabeled_weight": 0.3, "b_unlabeled_weight": 0.45, "direct": 0.2,
                "evidence": 0.1, "b_latent": 0.15, "modality_dropout": 0.1, "rdrop": 0.1, "sym_rank": 0.04},
        return_output=True,
    )
    assert "path_c_logits" in out and not out["logits"].requires_grad
    assert "path_c_cls" in terms
    loss.backward()
    for grp in groups:
        norm = torch.nn.utils.clip_grad_norm_(grp, 5.0)
        assert torch.isfinite(norm) and norm > 0
