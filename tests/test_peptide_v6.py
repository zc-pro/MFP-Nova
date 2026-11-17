import math

import torch
from torch.nn import functional as F

from mfp_nova.models.peptide_v6 import PeptideV6, pool_windows, shift_context


def test_pool_uses_two_valid_windows_and_ignores_missing_widths():
    scores = torch.tensor([[[4.], [0.], [90.]], [[2.], [99.], [98.]],
                           [[99.], [99.], [99.]]], requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
    pooled = pool_windows(scores, valid).squeeze(-1)
    torch.testing.assert_close(pooled[:2], torch.tensor([math.log((math.exp(4) + 1) / 2), 2.]))
    assert torch.isfinite(pooled).all()
    pooled.sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert scores.grad[~valid].count_nonzero() == 0


def test_context_shift_keeps_the_valid_window_set_per_peptide():
    context = torch.arange(10).reshape(2, 5, 1)
    shifted = shift_context(context, torch.arange(5), torch.tensor([[4], [1]]))
    torch.testing.assert_close(shifted[0, :4, 0], torch.tensor([2, 3, 0, 1]))
    torch.testing.assert_close(shifted[1, :1, 0], torch.tensor([5]))


def test_v6_padding_invariance_and_interaction_diagnostics():
    torch.manual_seed(42)
    stats = {"labels": ["one", "two"], "observed_prior": [.3, .1], "positive_weights": [1., 1.]}
    model = PeptideV6(6, 4, 2, stats, dim=8, cnn_channels=4, interaction_rank=3, dropout=0.).eval()
    aa = torch.tensor([[1, 2, 0, 0, 0], [5, 4, 3, 2, 1]])
    batch = {"aa": aa, "mask": aa > 0, "esm": torch.randn(2, 5, 6), "esm_peptide": torch.randn(2, 5, 4),
             "y": torch.tensor([[1., 1.], [0., 1.]])}

    first = model(batch, diagnostics=True)
    torch.testing.assert_close(first["logits"], first["without_interaction"])
    torch.testing.assert_close(first["logits"], first["shifted_context"])
    loss, _ = model.compute_loss(batch, {})
    loss.backward()
    assert model.interaction.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.interaction.weight.normal_()
    model.zero_grad()

    final = model(batch, diagnostics=True)
    assert not torch.allclose(final["logits"], final["without_interaction"])
    torch.testing.assert_close(final["logits"][0], final["shifted_context"][0])
    assert not torch.allclose(final["logits"][1], final["shifted_context"][1])
    torch.testing.assert_close(final["evidence_weights"].sum(1), torch.ones(2, 2))
    assert final["evidence_weights"][0, 2:].count_nonzero() == 0

    padded = {**batch, "aa": F.pad(aa, (0, 5), value=99), "mask": F.pad(batch["mask"], (0, 5))}
    for name in ("esm", "esm_peptide"):
        padded[name] = F.pad(batch[name], (0, 0, 0, 5), value=float("nan"))
    for name, values in model(padded, diagnostics=True).items():
        torch.testing.assert_close(values, final[name])

    short = {key: value[:1, :2] if key != "y" else value[:1] for key, value in batch.items()}
    for name, values in model(short, diagnostics=True).items():
        torch.testing.assert_close(values, final[name][:1])
    loss, _ = model.compute_loss(batch, {})
    loss.backward()
    for parameter in (model.local_factor.weight, model.context_factor.weight):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
