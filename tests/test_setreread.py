import numpy as np
import pytest
import torch

from mfp_nova.models.setreread import SetReread
from test_research import sample_statistics, synthetic_batch


def make_model(labels=15, **options):
    return SetReread(12, labels, 10, sample_statistics(labels), np.eye(15)[np.arange(labels) % 15],
                     dim=24, cnn_channels=8, dropout=0., **options)


@pytest.mark.parametrize("labels,interval", [(15, False), (39, False), (39, True)])
def test_trainable_reread_and_padding(labels, interval):
    batch = synthetic_batch(labels)
    batch["y"][0, 1] = 1
    model = make_model(labels, interval_evidence=interval)
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(2):
        opt.zero_grad()
        loss, terms = model.compute_loss(batch, {"edit_gain": .5, "recovery": .2})
        loss.backward()
        opt.step()
    assert all(torch.isfinite(value) for value in terms.values())
    assert terms["recovery"] > 0
    assert model.cnn.convs1[0].weight.grad.abs().sum() > 0
    assert model.add_head[-1].weight.grad.abs().sum() > 0
    if interval:
        assert model.interval_routing.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        logits = model(batch)["logits"]
        corrupted = dict(batch, esm=batch["esm"].masked_fill(~batch["mask"][..., None], 99),
                         aa=batch["aa"].masked_fill(~batch["mask"], 20))
        torch.testing.assert_close(model(corrupted)["logits"], logits)
    assert logits.shape == (3, labels)
    assert torch.isfinite(logits).all()


def test_budget_and_threshold_match_selected_set():
    model = make_model()
    scores = torch.full((4, 15), -1.)
    scores[1, :1] = scores[2, :2] = scores[3, :] = 1.
    weights = torch.rand(4, 15, 4).softmax(-1)
    local = torch.rand(4, 4, 24)
    signals = torch.rand(4, 15, 2)
    _, _, sets, gains, valid = model.score_edits(local, weights, signals, scores)
    assert valid.sum(1).tolist() == [5, 10, 15, 3]
    for candidates, use in zip(sets, valid):
        assert len(candidates[use].unique(dim=0)) == use.sum()
    # Give a real edit a positive gain and a padded candidate an even higher one.
    gains.fill_(-1.)
    gains[:, 0] = 0.
    gains[0, 1] = gains[2, -1] = .1
    gains[0, -1] = 100.
    expected = torch.stack((sets[0, 1], sets[1, 0], sets[2, -1], sets[3, 0]))
    assert torch.equal(model.decode(scores, sets, gains, valid).sigmoid() >= .5, expected)


def test_replacing_function_rereads_shared_and_remaining_evidence():
    model = make_model()
    local = torch.ones(1, 4, 24)
    weights = torch.zeros(1, 15, 4)
    weights[0, :3] = torch.tensor([[.55, .35, .10, 0], [.10, .20, .50, .20], [.20, .10, .55, .15]])
    signals = torch.zeros(1, 15, 2)
    retained = torch.zeros(1, 15, dtype=torch.bool)
    retained[:, [0, 2]] = True
    candidate = torch.tensor([[1]])
    before = model.read(local, weights, signals, retained, candidate)
    retained = retained.clone()
    retained[:, 2] = False
    after = model.read(local, weights, signals, retained, candidate)
    torch.testing.assert_close(before[..., -3:-1], torch.tensor([[[.95, .05]]]))
    torch.testing.assert_close(after[..., -3:-1], torch.tensor([[[.40, .60]]]))
