import json
from pathlib import Path

import pytest
import torch

from mfp_nova.models.hh_setreread import HHSetReread
from mfp_nova.metric_protocol import final_threshold, selection
from test_research import sample_statistics, synthetic_batch


def small_model(relation="bilinear"):
    stats = sample_statistics(39)
    labels = Path(__file__).resolve().parents[1] / "data/therapeutic_2025/tracks/fine39/labels.json"
    stats["labels"] = json.loads(labels.read_text(encoding="utf-8"))["labels"]
    return HHSetReread(input_dim=12, num_labels=39, vocab_size=10, statistics=stats,
                       axis_statistics={"labels": ["axis"]*15, "positive_weights": [1.]*15},
                       dim=32, heads=4, dropout=0, cnn_channels=8, pair_dim=8, decoder_depth=1,
                       relation=relation, label_context=False, prototype_readout=False)


@pytest.mark.parametrize("relation", ["off", "shared", "bilinear", "mlp"])
def test_zero_relation_and_training_gradients(relation):
    batch = synthetic_batch(39)
    model = small_model(relation).eval()
    with torch.no_grad():
        unary = model.backbone(batch)["logits"]
        torch.testing.assert_close(model(batch)["logits"], unary, rtol=0, atol=0)
    model.train()
    model.freeze_backbone(True)
    loss, terms = model.compute_loss(batch, {"hh_ranking": "weak"})
    loss.backward()
    assert all(torch.isfinite(value) for value in terms.values())
    assert model.query.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    if relation != "off":
        assert model.shared_readout.weight.grad.abs().sum() > 0
    if relation in ("bilinear", "mlp"):
        readout = model.private_readout if relation == "bilinear" else model.private_readout[-1]
        assert readout.weight.grad.abs().sum() > 0


def test_nonzero_pair_potentials_are_padding_invariant_and_symmetric():
    batch = synthetic_batch(39)
    model = small_model().eval()
    with torch.no_grad():
        model.shared_readout.weight.normal_(std=.1)
        model.private_readout.weight.normal_(std=.1)
        output = model(batch)
        assert output["psi"].abs().max() <= 1
        padded = dict(batch, esm=batch["esm"].masked_fill(~batch["mask"][..., None], 100))
        torch.testing.assert_close(model(padded)["logits"], output["logits"], rtol=1e-5, atol=1e-6)
        model.pair_left, model.pair_right = model.pair_right, model.pair_left
        torch.testing.assert_close(model(batch)["psi"], output["psi"], rtol=1e-5, atol=1e-6)


def test_compressed_decoder_matches_full_axis_enumeration():
    torch.set_num_threads(1)
    torch.manual_seed(310)
    model = small_model()
    unary = torch.randn(5, 39) * .18
    generic_label = model.projection[:, 4].bool()
    unary[0, generic_label] = .1
    unary[1, generic_label] = -.05
    unary[2, generic_label] = 0
    psi = torch.rand(5, 101) * 2-1
    psi[2] = 0
    prediction = model.exact_decode(unary, psi)
    states = ((torch.arange(32768)[:, None] >> torch.arange(15)) & 1).bool()
    membership = model.projection.T.bool()
    positive = torch.where(membership, unary[:, None].clamp_min(0), 0.).sum(-1)
    maximum = unary[:, None].masked_fill(~membership, -torch.inf).amax(-1)
    group_value = torch.where(positive > 0, positive, maximum)
    reference = group_value @ states.float().T + model.relation_energy(model.effective(states), psi)
    energy = model.set_energy(prediction[:, None], unary, psi).squeeze(-1)
    torch.testing.assert_close(energy, reference.amax(-1), atol=2e-6, rtol=1e-6)
    assert torch.equal(prediction[2], unary[2] >= 0)
    cond = model.conditional_logits(unary, psi)
    base = unary >= 0
    label = torch.eye(39, dtype=torch.bool)[None]
    full_delta = model.set_energy(base[:, None] | label, unary, psi) - model.set_energy(base[:, None] & ~label, unary, psi)
    torch.testing.assert_close(cond, full_delta, atol=2e-6, rtol=1e-6)


def test_jaccard_selection_uses_fixed_threshold_and_tiebreaks():
    report = dict(sample_jaccard=.6, sample_precision=.7, sample_recall=.8, micro_recall=.9, micro_f1=.95)
    assert selection(None, None, report, "jaccard", "per_label")[:2] == ((.6, .7, .8, .9), .5)
    assert final_threshold(None, None, "jaccard", "per_label") == .5
