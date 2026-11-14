import numpy as np
import pytest
import torch

from mfp_nova.hh_v13 import select_edits
from mfp_nova.models.hh_setreread_v13 import HHSetRereadV13, addition_targets, apply_edits
from test_hh_setreread import small_model


def test_edit_targets_match_jaccard_and_keep_rule():
    truth = torch.tensor([[True, True, False], [True, False, False], [False, True, False]])
    base = torch.tensor([[True, False, False], [False, False, False], [True, True, True]])
    jaccard = lambda s: (s & truth).sum(-1) / (s | truth).sum(-1)
    for c in range(3):
        added = base.clone()
        added[:, c] = True
        candidate = ~base[:, c]
        torch.testing.assert_close(addition_targets(base, truth)[candidate, c],
                                   (jaccard(added)-jaccard(base))[candidate])
    unary = base.float()*2-1
    gains = torch.tensor([[0., 1., .2], [0., 0., 0.], [1., 1., 1.]])
    edited, labels = apply_edits(unary, gains, 0.)
    assert labels.tolist() == [1, -1, -1]
    assert (edited | base).equal(edited)
    assert apply_edits(unary, gains, None)[0].equal(base)
    arrays = {"base_logits": unary.numpy(), "edit_gain": gains.numpy(), "targets": truth.numpy()}
    best, _, curve = select_edits(arrays)
    assert curve[-1]["edit_threshold"] is None and curve[-1]["edits"] == 0
    assert best["sample_jaccard"] == pytest.approx(float(jaccard(edited).mean()))


@pytest.mark.parametrize("readout", ["scores", "unconditional", "conditional"])
def test_real_edit_head_gradients_and_no_background_equivalence(readout):
    model = HHSetRereadV13(small_model("off").statistics, readout=readout)
    batch = {"mask": torch.tensor([[True, True, False], [True, True, True]]),
             "esm": torch.randn(2, 3, 1280), "esm_peptide": torch.randn(2, 3, 640),
             "base_logits": -torch.ones(2, 39), "y": torch.zeros(2, 39)}
    batch["base_logits"][1, 1] = 2
    batch["y"][:, 3] = 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert model.fine_queries.grad.abs().sum() > 0
    if readout != "scores":
        assert model.projection.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        out = model(batch)["edit_gain"]
        no_background = model(batch, without_background=True)["edit_gain"]
        torch.testing.assert_close(out[0], no_background[0])
        padded = dict(batch, esm=batch["esm"].masked_fill(~batch["mask"][..., None], 100.))
        torch.testing.assert_close(out, model(padded)["edit_gain"])
        assert np.isfinite(out.numpy()).all()
