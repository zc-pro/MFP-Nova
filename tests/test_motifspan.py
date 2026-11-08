import torch
import pytest

from mfp_nova.models.motifspan import MotifSpan, relation_statistics, soft_assignment
from test_research import synthetic_batch, sample_statistics


def small(**kwargs):
    return MotifSpan(12, 15, 10, sample_statistics(15), dim=32, cnn_channels=8,
                     relation_dim=4, dropout=0., **kwargs)


def test_prefix_matches_explicit_values_and_gradients():
    torch.manual_seed(8)
    a = torch.randn(2, 5, 3, dtype=torch.double).softmax(-1).requires_grad_()
    left = torch.randn(2, 5, 4, dtype=torch.double, requires_grad=True)
    right = torch.randn_like(left, requires_grad=True)
    t, mass = relation_statistics(a, left, right)
    explicit = torch.zeros_like(t)
    expected_mass = torch.zeros_like(mass)
    for s, rho in enumerate((.5, .9, 1.)):
        for i in range(5):
            for j in range(i+1, 5):
                edge = a[:, i, :, None]*a[:, j, None, :]*rho**(j-i-1)
                expected_mass[:, s] += edge
                explicit[:, s] += edge[..., None]*(left[:, i]*right[:, j])[:, None, None]
    torch.testing.assert_close(t, explicit)
    torch.testing.assert_close(mass, expected_mass)
    grad1 = torch.autograd.grad(t.square().sum(), (a, left, right), retain_graph=True)
    grad2 = torch.autograd.grad(explicit.square().sum(), (a, left, right))
    for x, y in zip(grad1, grad2):
        torch.testing.assert_close(x, y)


def test_assignment_and_conservation():
    mask = torch.ones(2, 13, dtype=torch.bool)
    mask[0, 3:] = False
    a, _, active, count = soft_assignment(torch.ones(2, 13), mask)
    torch.testing.assert_close(a.sum(-1), mask.float())
    assert count.tolist() == [1, 3]
    left, right = torch.randn(2, 13, 4), torch.randn(2, 13, 4)
    total = relation_statistics(a, left, right)[0].sum((2, 3))
    changed, _, _, _ = soft_assignment(1+.4*torch.randn(2, 13).tanh(), mask)
    torch.testing.assert_close(total, relation_statistics(changed, left, right)[0].sum((2, 3)), atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("length", [1, 7, 13])
def test_complete_mask_gradients_and_degenerate_relations(length):
    batch = synthetic_batch(length=length)
    net = small().eval()
    evidence = net(batch, return_evidence=True)
    assert torch.isfinite(evidence["logits"]).all()
    changed = dict(batch, esm=batch["esm"].masked_fill(~batch["mask"][..., None], 999))
    torch.testing.assert_close(evidence["logits"], net(changed)["logits"])
    if length == 1:
        assert evidence["relation"].count_nonzero() == 0
    torch.testing.assert_close(evidence["spatial_summary"][0], torch.zeros_like(evidence["spatial_summary"][0]), atol=1e-7, rtol=0)
    net.compute_loss(batch, {})[0].backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    if length > 6:
        assert net.velocity_out.weight.grad.abs().sum() > 0
        assert net.left.weight.grad.abs().sum() > 0
        assert net.detail_relation.weight.grad.abs().sum() > 0
    # Short peptide is independent of the padded length of other samples.
    single = {k: v[:1].clone() for k, v in batch.items()}
    n = min(3, length)
    for key in ("esm", "aa", "mask"):
        single[key] = single[key][:, :n]
    torch.testing.assert_close(evidence["logits"][:1], net(single)["logits"], atol=2e-6, rtol=1e-5)


def test_uniform_spatial_selection_and_short_training():
    batch = synthetic_batch(length=13)
    net = small()
    with torch.no_grad():
        net.spatial_query.zero_()
    evidence = net(batch, return_evidence=True)
    torch.testing.assert_close(evidence["spatial_summary"], torch.zeros_like(evidence["spatial_summary"]), atol=1e-6, rtol=0)
    opt = torch.optim.AdamW(net.parameters(), lr=.003)
    first = net.compute_loss(batch, {})[0].item()
    for _ in range(4):
        opt.zero_grad()
        loss, _ = net.compute_loss(batch, {"rdrop": .1})
        loss.backward()
        opt.step()
    assert loss.item() < first
