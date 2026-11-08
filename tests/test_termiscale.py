import torch
import pytest

from mfp_nova.models.termiscale import TermiScale, endpoint_log_measures
from test_research import synthetic_batch, sample_statistics


def model(labels=15, **kwargs):
    return TermiScale(12, labels, 10, sample_statistics(labels), dim=32, dropout=0., **kwargs)


@pytest.mark.parametrize("labels", [15, 39, 48, 49])
def test_forward_backward_padding(labels):
    batch = synthetic_batch(labels)
    net = model(labels).eval()
    out = net(batch)["logits"]
    assert out.shape == (3, labels)
    changed = dict(batch, esm=batch["esm"].masked_fill(~batch["mask"][..., None], 999))
    torch.testing.assert_close(out, net(changed)["logits"])
    loss, _ = net.compute_loss(batch, {})
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert net.blocks[0].router.weight.grad.abs().sum() > 0
    assert net.condition_weights.grad.abs().sum() > 0
    single = {k: v[:1].clone() for k, v in batch.items()}
    for key in ("esm", "aa", "mask"):
        single[key] = single[key][:, :3]
    torch.testing.assert_close(out[:1], net(single)["logits"], atol=1e-6, rtol=1e-5)


def test_measures_constant_field_single_residue_and_exact_budget():
    batch = synthetic_batch(15, length=1)
    net = model().eval()
    assert torch.isfinite(net(batch)["logits"]).all()
    batch = synthetic_batch()
    with torch.no_grad():
        net.label.weight.zero_()
    evidence = net(batch, return_evidence=True)
    torch.testing.assert_close(evidence["logits"], net.bias[None].expand(3, -1))
    rho = endpoint_log_measures(batch["mask"]).exp()
    torch.testing.assert_close(rho.sum(1), torch.ones(3, 3))
    full = TermiScale(1280, 39, 10, sample_statistics(39))
    assert sum(p.numel() for p in full.parameters()) == 1703808


def test_dual_and_rdrop_short_optimization():
    batch = synthetic_batch()
    batch["esm_peptide"] = torch.randn(3, 7, 6)
    net = model(peptide_input_dim=6).train()
    opt = torch.optim.AdamW(net.parameters(), lr=.002)
    for _ in range(3):
        opt.zero_grad()
        loss, terms = net.compute_loss(batch, {"rdrop": .1})
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
    assert net.features.projection.weight.grad[:, 12:].abs().sum() > 0


def test_control_semantics_uniform_pool_label_only_and_equal_scales():
    batch = synthetic_batch()
    uniform = model(pooling="uniform").eval()
    out = uniform(batch, return_evidence=True)
    actual = out["log_measure"].exp().masked_fill(~batch["mask"][..., None], 0)
    expected = (batch["mask"].float()/batch["mask"].sum(1, keepdim=True))[..., None].expand_as(actual)
    torch.testing.assert_close(actual, expected)
    fixed = model(pooling="fixed").eval()
    with torch.no_grad():
        fixed.condition_weights.normal_()
    evidence = fixed(batch, return_evidence=True)
    torch.testing.assert_close(evidence["mixture"][0], evidence["mixture"][1])
    evidence["logits"].square().sum().backward()
    assert fixed.condition_weights.grad is None
    assert fixed.measure_bias.grad.abs().sum() > 0
    equal = model(routing="uniform").eval()(batch, return_evidence=True)
    torch.testing.assert_close(equal["scale_weights"], torch.full_like(equal["scale_weights"], 1/3))
