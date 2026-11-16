import pytest
import torch
from torch.nn import functional as F

from mfp_nova.models.peptide_v3 import DISTANCE_BANDS, OrderedInteractions, PeptideV3
from mfp_nova.models.theraset import TheraSet


@pytest.mark.parametrize("mode", ("additive", "product", "centered"))
def test_interval_messages_match_explicit_pairs(mode):
    generator = torch.Generator().manual_seed(42)
    left, right = torch.randn(2, 3, 19, 2, generator=generator)
    mask = torch.arange(19)[None] < torch.tensor([2, 9, 19])[:, None]
    module = OrderedInteractions(4, 8, 2, mode)
    with torch.no_grad():
        module.band_weight.copy_(torch.randn((2, 4, 2), generator=generator))
    actual = module.messages(left, right, mask)
    expected = [torch.zeros_like(left), torch.zeros_like(right)]
    for row, valid in enumerate(mask):
        length = int(valid.sum())
        counts = torch.zeros(2, length, 1)
        for band, (low, high) in enumerate(DISTANCE_BANDS):
            first, second = torch.where((torch.arange(length)[None] - torch.arange(length)[:, None] >= low) &
                                       (torch.arange(length)[None] - torch.arange(length)[:, None] <= high))
            if not first.numel():
                continue
            a, b = left[row, first], right[row, second]
            if mode != "product":
                a, b = a - a.mean(0), b - b.mean(0)
            values = a + b if mode == "additive" else a * b
            for role, endpoints in enumerate((first, second)):
                for i in endpoints.unique():
                    expected[role][row, i] += values[endpoints == i].mean(0) * module.band_weight.detach()[role, band]
                    counts[role, i] += 1
        for i in range(2):
            expected[i][row, :length] /= counts[i].clamp_min(1)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)
    if mode == "centered":
        for value in module.messages(torch.full_like(left, 2.), right, mask):
            torch.testing.assert_close(value, torch.zeros_like(value))


def test_zero_initialized_v3_matches_sequence_cnn_and_learns():
    statistics = {"labels": ["example", "other"], "observed_prior": [.4, .2], "positive_weights": [1., 1.]}
    torch.manual_seed(42)
    original = TheraSet(0, 2, 2, statistics, cnn_channels=4, dropout=0.,
                       evidence_decoder=False, chemical_evidence=False).eval()
    torch.manual_seed(42)
    model = PeptideV3(2, statistics, cnn_channels=4, relation_rank=3, dropout=0.).eval()
    aa = torch.tensor([[1, 2, 3, 7, 8, 9], [2, 7, 4, 1, 0, 0]])
    batch = {"aa": aa, "mask": aa > 0, "y": torch.tensor([[1., 0.], [0., 1.]])}
    padded = {**batch, "aa": F.pad(batch["aa"], (0, 16), value=16), "mask": F.pad(batch["mask"], (0, 16))}
    torch.testing.assert_close(model(batch)["logits"], original(batch)["logits"])
    loss, _ = model.compute_loss(batch, {})
    loss.backward()
    assert model.interaction.write_left.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.interaction.write_left.weight.normal_()
        model.interaction.write_right.weight.normal_()
    model.zero_grad()
    torch.testing.assert_close(model(batch)["logits"], model(padded)["logits"])
    loss, _ = model.compute_loss(batch, {})
    loss.backward()
    assert model.interaction.left.weight.grad.abs().sum() > 0
    assert model.interaction.right.weight.grad.abs().sum() > 0
