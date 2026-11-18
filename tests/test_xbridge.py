import numpy as np
import torch
from torch.nn import functional as F

from mfp_nova.models.xbridge import XBridge, axis_coverage, positive_bridge
from mfp_nova.multimodal import function_projection
from mfp_nova.xbridge_data import PositivePairBatches
from test_research import sample_statistics, synthetic_batch


LABELS = ["anti_gram+", "anti_gram_", "antimicrobial", "antioxidant", "antihypertensive"]
PROJECTION = torch.tensor(function_projection(LABELS, "fine39"), dtype=torch.bool)


def test_axis_coverage_preserves_observed_fine_labels_and_empty_rows():
    logits = torch.tensor([[-3., -2., -1000., -4., 1000.], [2., 3., 4., 5., 6.]], requires_grad=True)
    targets = torch.tensor([[1, 1, 1, 1, 0], [0, 0, 0, 0, 0]])
    loss = axis_coverage(logits, targets, PROJECTION, .2)
    gram = .2 * (torch.logsumexp(logits[0, :2] / .2, dim=0) - np.log(2))
    torch.testing.assert_close(loss, (F.softplus(-gram) + F.softplus(-logits[0, 3])) / 2)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad[0, [0, 1, 3]] < 0).all()
    assert (logits.grad[0, [2, 4]] == 0).all()
    assert (logits.grad[1] == 0).all()


def test_positive_bridge_obeys_label_and_cluster_mask():
    evidence = torch.tensor([[[1., 0.]], [[0., 1.]], [[-1., 0.]], [[-1., 0.]]], requires_grad=True)
    targets = torch.tensor([[1], [1], [1], [0]])
    clusters = torch.tensor([0, 1, 1, 2])
    loss, pairs, active = positive_bridge(evidence, targets, clusters, .5)
    torch.testing.assert_close(loss, torch.tensor((.25 + 2.25) / 2))
    assert pairs == 2 and active == 1
    loss.backward()
    assert evidence.grad[0].abs().sum() > 0 and evidence.grad[-1].count_nonzero() == 0
    empty, pairs, _ = positive_bridge(evidence, targets, torch.zeros_like(clusters), .5)
    assert empty == 0 and pairs == 0 and empty.requires_grad
    constant, _, _ = positive_bridge(torch.ones_like(evidence), targets, clusters, .5)
    assert constant == 0


def test_sampler_keeps_all_rows_and_supplies_cross_cluster_positives():
    targets = np.array([[1], [0], [0], [1], [0], [1], [0]])
    clusters = np.array([1, 1, 1, 2, 2, 3, 3])
    batches = list(PositivePairBatches(targets, clusters, 4, torch.Generator().manual_seed(42)))
    assert set(sum(batches, [])) == set(range(len(targets)))
    for batch in batches:
        assert len(batch) <= 4 and len(batch) == len(set(batch))
        assert len(set(clusters[index] for index in batch if targets[index, 0])) >= 2


def test_reference_initialization_mask_and_gradient_paths():
    batch = synthetic_batch(len(LABELS))
    batch["esm_peptide"] = torch.randn(3, batch["aa"].shape[1], 8)
    args = dict(num_labels=len(LABELS), statistics=sample_statistics(len(LABELS)), input_dim=12,
                peptide_input_dim=8, vocab_size=10)
    model = XBridge(**args, function_projection=PROJECTION, dim=16, dropout=0.,
                    reference_args={**args, "dim": 32, "heads": 4, "cnn_channels": 8,
                                    "pair_dim": 8, "decoder_depth": 1}).train()
    output = model(batch)
    assert torch.equal(output["logits"], output["reference_logits"])
    assert not model.reference.training
    output["logits"].sum().backward()
    assert model.queries.grad.count_nonzero() == 0
    assert model.residual[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.reference.parameters())
    model.zero_grad(set_to_none=True)
    fold_scores = torch.full_like(batch["y"], -1.)
    loss, _, output = model.compute_loss(dict(batch, base_logits=fold_scores), {}, return_output=True)
    assert torch.equal(output["logits"], fold_scores)
    loss.backward()
    assert model.queries.grad.abs().sum() > 0
    model.eval()
    assert torch.equal(model(dict(batch, base_logits=fold_scores))["logits"],
                       model(batch)["reference_logits"])
    standalone = XBridge(**args, function_projection=PROJECTION, dim=16).eval()
    padded = dict(batch, mask=F.pad(batch["mask"], (0, 4)),
                  residue_chem=F.pad(batch["residue_chem"], (0, 4), value=99))
    for key in ("esm", "esm_peptide"):
        padded[key] = F.pad(batch[key], (0, 0, 0, 4), value=float("nan"))
    torch.testing.assert_close(standalone(padded)["logits"], standalone(batch)["logits"])
