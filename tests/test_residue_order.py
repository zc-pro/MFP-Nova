from itertools import permutations

import torch
from torch.nn import functional as F

from mfp_nova.models.residue_order import ResidueOrderEvidence
from test_research import synthetic_batch
from test_theraset import small_model


def test_order_model_normalizes_over_each_composition():
    aa = torch.tensor(sorted(set(permutations((1, 1, 9, 10)))))
    order = ResidueOrderEvidence([18, 9], output_dim=4)
    background, conditional = order.observed_log_probabilities(aa, aa > 0)
    torch.testing.assert_close(background.sum(1).exp().sum(), torch.tensor(1.))
    torch.testing.assert_close(conditional.sum(1).exp().sum(0), torch.ones(2))
    assert (conditional[..., 0] - background).abs().max() > 0


def test_order_context_is_causal_and_scores_ignore_padding():
    order = ResidueOrderEvidence([18, 9], output_dim=4)
    aa = torch.tensor([[1, 9, 1, 10], [3, 3, 3, 0]])
    mask = aa > 0
    changed = torch.tensor([[1, 9, 10, 1], [3, 3, 3, 0]])
    torch.testing.assert_close(order.prefix_context(aa, mask)[:, :3],
                               order.prefix_context(changed, mask)[:, :3])
    with torch.no_grad():
        order.write.weight.normal_()
    evidence, _ = order(aa, mask)
    padded, _ = order(F.pad(aa, (0, 8), value=20), F.pad(mask, (0, 8)))
    torch.testing.assert_close(evidence, padded)
    torch.testing.assert_close(evidence[1], torch.zeros_like(evidence[1]))


def test_order_supervision_is_positive_only_and_background_gradients_are_isolated():
    order = ResidueOrderEvidence([18, 9], output_dim=4)
    aa = torch.tensor([[1, 9, 1, 10], [3, 3, 3, 0]])
    targets = torch.tensor([[0., 1.], [0., 0.]])
    _, totals = order(aa, aa > 0, targets)
    numerator, denominator = totals["order_conditional"]
    torch.testing.assert_close(denominator, order.positive_weight[1])
    numerator.backward()
    assert order.background.weight.grad is None
    assert order.label_factor.grad[0].count_nonzero() == 0
    assert order.label_factor.grad[1].abs().sum() > 0


def test_theraset_v4_matches_baseline_at_initialization_and_trains_with_rdrop():
    batch = synthetic_batch()
    options = {"label_context": False, "prototype_readout": False}
    torch.manual_seed(42)
    baseline = small_model(**options).eval()
    torch.manual_seed(42)
    v4 = small_model(**options, order_evidence=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(v4(batch)["logits"], baseline(batch)["logits"], atol=0., rtol=0.)
    v4.train()
    visits = []
    with_order = v4.order_evidence.register_forward_hook(lambda *args: visits.append(True))
    total, terms = v4.compute_loss(batch, {"rdrop": .1, "order_background": .01, "order_conditional": .01})
    total.backward()
    with_order.remove()
    assert visits == [True]
    assert terms["order_background"] > 0 and terms["order_conditional"] > 0
    for name in ("background", "context_factor", "residue_factor", "write"):
        gradient = getattr(v4.order_evidence, name).weight.grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
