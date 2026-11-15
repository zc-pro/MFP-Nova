import numpy as np
import pytest
import torch

from mfp_nova.hh_v14 import base_candidate_scores, choose_threshold, supervision
from mfp_nova.models.hh_setreread_v14 import HHSetRereadV14, apply_second_axis
from test_hh_setreread import small_model


def example(model):
    labels = model.statistics["labels"]
    gram = labels.index("anti_gram+")
    sibling = labels.index("anti_gram_")
    generic = labels.index("antimicrobial__unknown")
    other = labels.index("opioid")
    fourth = labels.index("antihiv")
    unary = torch.full((4, 39), -3.)
    unary[0, gram] = unary[1, generic] = unary[3, gram] = unary[3, other] = 2.
    y = torch.zeros_like(unary)
    y[0, [gram, other, fourth]] = 1.
    y[1, [generic, other]] = 1.
    y[2, other] = y[3, gram] = 1.
    return unary, y, gram, sibling, generic, other, fourth


def test_candidates_preserve_the_axis_and_require_the_exact_fine_label():
    model = HHSetRereadV14(small_model("off").statistics)
    unary, y, gram, sibling, generic, other, fourth = example(model)
    valid, exact, _, anchor, multi = model.supervision(unary, y)
    assert valid.any(-1).tolist() == [True, True, False, False]
    assert not valid[0, sibling] and not valid[0, generic] and not valid[1, gram]
    assert exact[0, other] and exact[0, fourth] and exact[1, other]
    assert anchor.tolist() == [True, True, False, True] and multi.tolist() == [True, True, False, False]
    scores = torch.zeros_like(unary)
    scores[:, other] = 1.
    out, chosen = apply_second_axis(unary, valid, torch.tensor([.5, .7, .9, 1.]), scores, .5)
    assert chosen.tolist() == [-1, other, -1, -1]
    assert torch.equal(model.effective(model.raw_axes(out)).sum(-1), torch.tensor([1, 2, 0, 2]))


def test_separate_branch_gradients_and_padding():
    torch.manual_seed(8)
    model = HHSetRereadV14(small_model("off").statistics)
    unary, y, *_ = example(model)
    batch = {"base_logits": unary[:2], "y": y[:2],
             "mask": torch.tensor([[True, True, False], [True, True, True]]),
             "esm": torch.randn(2, 3, 1280), "esm_peptide": torch.randn(2, 3, 640)}
    model.branch_loss(batch, "row").backward()
    assert model.row_head.reader.embedding.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.label_head.parameters())
    model.zero_grad(set_to_none=True)
    model.branch_loss(batch, "label").backward()
    assert model.label_head.context.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.row_head.parameters())
    with torch.no_grad():
        out = model(batch)
        damaged = dict(batch, esm=batch["esm"].masked_fill(~batch["mask"][..., None], 99.))
        for name in ("row_logit", "label_logit"):
            torch.testing.assert_close(out[name], model(damaged)[name])


def test_tied_threshold_is_deployable_and_empty_cohort_holds():
    model = HHSetRereadV14(small_model("off").statistics)
    unary, y, *_, other, _ = example(model)
    u, y = unary.numpy(), y.numpy()
    targets = supervision(model, u, y)
    targets["base_multifunction"] = np.array([False, False, False, True])
    row = np.full(len(u), .42, dtype=np.float32)
    labels = np.zeros_like(u)
    labels[:, other] = 1.
    report = choose_threshold(u, y, targets, row, labels)
    point = report["selected"]
    assert point["edits"] == point["strict_second_axis_recoveries"] == 2
    final, picked = apply_second_axis(unary, torch.from_numpy(targets["candidate_mask"]),
                                     torch.from_numpy(row), torch.from_numpy(labels), point["edit_threshold"])
    assert picked.tolist() == [other, other, -1, -1]
    predicted = final.numpy()
    assert point["micro_f1"] == pytest.approx(2*(predicted & y.astype(bool)).sum()/(y.sum()+predicted.sum()))
    empty = {k: v[2:3] for k, v in targets.items()}
    control = base_candidate_scores(u[2:3], empty["candidate_mask"])
    result = choose_threshold(u[2:3], y[2:3], empty, control, labels[2:3])
    assert result["selected"]["edit_threshold"] is None and result["best_active"] is None
