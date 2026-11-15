import torch

from mfp_nova.models.hh_setreread_v15_ranker import HHSetRereadV15Ranker, select_hard_pair


def test_hard_pair_only_uses_missed_recorded_and_called_unrecorded_labels():
    scores = torch.tensor([[-2., 3., -.5, 1.], [.5, -1., -2., -3.], [-1., -2., -3., -4.]])
    targets = torch.tensor([[1., 0., 1., 0.], [1., 0., 0., 0.], [1., 0., 0., 0.]])
    add, drop, valid = select_hard_pair(scores, targets)
    assert valid.tolist() == [True, False, False]
    assert (int(add[0]), int(drop[0])) == (2, 1)


def test_ranker_padding_and_pair_gradient():
    labels = ["anti_gram+", "antibacterial__unknown", "opioid", "antimicrobial__unknown"]
    statistics = {"labels": labels, "observed_prior": [.25] * 4, "positive_weights": [1.] * 4}
    model = HHSetRereadV15Ranker(8, 6, 4, 22, statistics, dim=16, dropout=0).eval()
    batch = {
        "esm": torch.randn(2, 5, 8),
        "esm_peptide": torch.randn(2, 5, 6),
        "mask": torch.tensor([[True] * 5, [True] * 3 + [False] * 2]),
        "y": torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
        "base_logits": torch.tensor([[-1., 1., -1., -1.], [-1., -1., 1., -1.]]),
    }
    unpadded = {"esm": batch["esm"][1:, :3], "esm_peptide": batch["esm_peptide"][1:, :3],
                "mask": batch["mask"][1:, :3]}
    loss, terms, output = model.compute_loss(batch, {"hard_pair": .1}, return_output=True)
    torch.testing.assert_close(output["logits"][1:], model(unpadded)["logits"], atol=1e-6, rtol=1e-5)
    assert model.ambiguous_pair[0, 3] and model.ambiguous_pair[3, 0]
    grad, = torch.autograd.grad(terms["hard_pair"], output["logits"], retain_graph=True)
    assert grad[0, 0] < 0 and grad[0, 1] > 0
    assert grad[1, 0] < 0 and grad[1, 2] > 0
    loss.backward()
    assert model.peptide[1].weight.grad.isfinite().all()
    batch["base_logits"].fill_(-1)
    assert model.compute_loss(batch, {"hard_pair": .1})[1]["hard_pair"] == 0
