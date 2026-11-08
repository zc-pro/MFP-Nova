import torch
import pytest

from mfp_nova.models.privilegedbridge import PrivilegedBridge, export_deployment
from mfp_nova.research import make_optimizer, make_model, checkpoint_state, load_model
from test_research import synthetic_batch, sample_statistics


def small(stage="A", **kwargs):
    return PrivilegedBridge(12, 15, 10, sample_statistics(15), dim=24, dropout=0., stage=stage, **kwargs)


def stages(**kwargs):
    a = small()
    b = small("B", **kwargs)
    b.initialize_from(a)
    c = small("C", **kwargs)
    c.initialize_from(b)
    return a, b, c


def missing(batch):
    return dict(batch, phys_mask=torch.zeros_like(batch["phys_mask"]),
                struct_mask=torch.zeros_like(batch["struct_mask"]),
                struct_global_mask=torch.zeros_like(batch["struct_global_mask"]))


def test_initialization_freezing_gradients_and_zero_evidence():
    batch = synthetic_batch()
    a, b, c = stages()
    b.train()
    assert not b.sequence.training
    assert not any(p.requires_grad for p in b.sequence.parameters())
    loss, terms = b.compute_loss(batch, {})
    loss.backward()
    assert all(p.grad is None for p in b.sequence.parameters())
    assert b.teacher.residual[0][0].weight.grad.abs().sum() > 0
    zero = b(missing(batch))
    torch.testing.assert_close(zero["residuals"], torch.zeros_like(zero["residuals"]))
    c.eval();a.eval()
    torch.testing.assert_close(c(batch)["logits"], a(batch)["logits"])
    c.train()
    c.compute_loss(batch, {"rdrop": .1})[0].backward()
    assert all(p.grad is None for p in c.reference.parameters())
    assert all(p.grad is None for p in c.teacher.parameters())
    assert c.bridges[0][-1].weight.grad.abs().sum() > 0
    assert c.sequence.features.projection.weight.grad.abs().sum() > 0
    c.eval()
    torch.testing.assert_close(c(batch)["logits"], c(missing(batch))["logits"])
    minimal = {k: batch[k] for k in ("esm", "aa", "mask")}
    torch.testing.assert_close(c(batch)["logits"], c(minimal)["logits"])


def test_missing_kd_and_stage_optimizers():
    _, b, c = stages()
    batch = synthetic_batch()
    loss, terms = c.compute_loss(missing(batch), {})
    assert terms["kd"] == 0
    loss.backward()
    opt = make_optimizer(c, {"learning_rate": 2e-5, "bridge_learning_rate": 2e-4, "weight_decay": .01})
    assert [g["lr"] for g in opt.param_groups] == [2e-5, 2e-4]
    for stage in (b, c):
        assert all(torch.isfinite(p.grad).all() for p in stage.parameters() if p.grad is not None)
    with pytest.raises(ValueError):
        c.initialize_from(small())


def test_actual_prediction_kd_allows_base_compensation():
    _, _, c = stages()
    batch = synthetic_batch()
    target = c.teacher_target(batch)
    # New objective depends on the actual student prediction, not a fixed bridge residual.
    before = c.compute_loss(batch, {})[1]["kd"].detach()
    with torch.no_grad():
        c.sequence.bias.add_(1)
    after = c.compute_loss(batch, {})[1]["kd"].detach()
    assert after > before


def test_export_keeps_deployment_predictions(tmp_path):
    _, _, c = stages()
    args = dict(input_dim=12, num_labels=15, vocab_size=10, statistics=sample_statistics(15), dim=24, dropout=0., stage="C")
    checkpoint = {"model_name": "privilegedbridge", "model_args": args, "model_state": checkpoint_state(c)}
    path = tmp_path/"deployment.pt"
    export_deployment(checkpoint, path)
    deployed, saved = load_model(path)
    c.eval();batch = synthetic_batch()
    torch.testing.assert_close(c(batch)["logits"], deployed(batch)["logits"])
    assert not hasattr(deployed, "teacher") and not hasattr(deployed, "reference")
