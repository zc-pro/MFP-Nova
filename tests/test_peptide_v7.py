from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from mfp_nova.models.peptide_v7 import (
    AXIS_NAMES, PeptideV7, axis_columns, axis_observations, child_log_odds, fit_axis_counts,
    nonempty_subsets,
)


LABELS = [
    "growth_regulatory", "anti_gram_", "antiplasmodial", "antihcv", "antiinflammatory",
    "antiangiogenesis", "opioid", "glucose_metabolism", "quorum_sensing",
    *(f"{axis}__unknown" for axis in AXIS_NAMES), "growth_regeneration__unknown", "anti_gram+",
]


def test_refined_subset_marginals_allow_both_siblings():
    state = torch.tensor([[.2, .1, .7]]).log()
    energies = torch.tensor([[2., 3.]]).log()
    # The two singleton sets and the joint set have weights 2, 3, and 6.
    logits = child_log_odds(state, energies, nonempty_subsets(2))
    torch.testing.assert_close(logits.sigmoid(), torch.tensor([[.7 * 8 / 11, .7 * 9 / 11]]))
    torch.testing.assert_close(child_log_odds(state, None, None).sigmoid(), torch.tensor([[.7]]))


def test_extreme_energies_have_finite_losses_and_gradients():
    state = torch.tensor([[80., -80., 60.], [-80., -60., 80.]], requires_grad=True)
    energies = torch.tensor([[-90., -85., -75.], [90., 85., 75.]], requires_grad=True)
    logits = child_log_odds(state.log_softmax(-1), energies, nonempty_subsets(3))
    loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([[1., 0., 0.], [0., 1., 1.]]))
    loss.backward()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(state.grad).all() and torch.isfinite(energies.grad).all()
    assert energies.grad.abs().sum() > 0


def test_v7_keeps_column_order_padding_and_both_head_gradients():
    targets = torch.zeros(3, len(LABELS))
    targets[0, LABELS.index("antibacterial__unknown")] = 1
    targets[1, [LABELS.index("anti_gram+"), LABELS.index("anti_gram_")]] = 1
    targets[2, [LABELS.index("growth_regeneration__unknown"), LABELS.index("growth_regulatory")]] = 1
    data = SimpleNamespace(y=targets.numpy(), labels=LABELS, indices=lambda split: [0, 1, 2])
    counts = fit_axis_counts(data)
    assert counts[0] == [1, 1, 1]
    states, conflict = axis_observations(targets, axis_columns(LABELS))
    assert states[:, 0].tolist() == [1, 2, 0] and not conflict.any()
    # An incompatible training label is rejected for an axis the head cannot express.
    targets[0, LABELS.index("anti_gram+")] = 1
    with pytest.raises(ValueError, match="antibacterial"):
        fit_axis_counts(data)
    targets[0, LABELS.index("anti_gram+")] = 0

    stats = {"labels": LABELS, "observed_prior": [.2] * len(LABELS),
             "positive_weights": [1.] * len(LABELS)}
    args = dict(input_dim=6, peptide_input_dim=4, num_labels=len(LABELS), statistics=stats,
                dim=12, res_dim=8, heads=2, seeds=2, cnn_channels=4)
    aa = torch.tensor([[1, 2, 0, 0], [5, 4, 3, 2], [3, 4, 5, 0]])
    batch = {"aa": aa, "mask": aa > 0, "esm": torch.randn(3, 4, 6),
             "esm_peptide": torch.randn(3, 4, 4), "y": targets}
    torch.manual_seed(42)
    flat = PeptideV7(**args, head="flat").eval()
    torch.manual_seed(42)
    structured = PeptideV7(**args, axis_counts=counts).eval()
    torch.testing.assert_close(flat.encode(batch), structured.encode(batch))

    padded = {**batch, "aa": F.pad(aa, (0, 5), value=99), "mask": F.pad(batch["mask"], (0, 5))}
    for key in ("esm", "esm_peptide"):
        padded[key] = F.pad(batch[key], (0, 0, 0, 5), value=float("nan"))
    result = structured(batch)
    torch.testing.assert_close(structured(padded)["logits"], result["logits"])
    for i, (unknown, children) in enumerate(structured.axes):
        probabilities = result["logits"].sigmoid()
        state = result["state_logits"][:, i].softmax(-1)
        torch.testing.assert_close(probabilities[:, unknown], state[:, 1])
        if len(children) == 1:
            torch.testing.assert_close(probabilities[:, children[0]], state[:, 2])
        else:
            assert (probabilities[:, children].sum(-1) >= state[:, 2]).all()

    for model in (flat, structured):
        loss, terms = model.train().compute_loss(batch, {"rdrop": .1})
        assert torch.isfinite(loss) and terms["rdrop"] > 0
        loss.backward()
        assert model.label.weight.grad.abs().sum() > 0
    assert structured.state.weight.grad.abs().sum() > 0
    assert structured.child.weight.grad.abs().sum() > 0
