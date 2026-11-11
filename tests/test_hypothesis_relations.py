import numpy as np
import torch
from torch.nn import functional as F

from mfp_nova.models.hypothesis_relations import (
    DecisionResidual,
    EvidenceMessages,
    PairJointHead,
    PrototypeReadout,
    align_edge_probes,
    build_prototypes,
    four_state_support,
    joint_loss,
    matched_pairs,
    message_probe_values,
    paired_rank_loss,
    select_edges,
    select_joint_pairs,
)


def test_prototypes_count_independent_groups_and_match_fold_banks():
    y = np.repeat([[0, 0], [1, 0], [0, 1], [1, 1]], 5, axis=0)
    observed = np.ones_like(y, dtype=bool)
    groups = np.arange(len(y))
    evidence = torch.randn(len(y), 2, 8, generator=torch.Generator().manual_seed(3))
    prototype, active, companions = build_prototypes(evidence, y, observed, groups, ["a", "b"])
    np.testing.assert_array_equal(companions, [1, 0])
    assert active.all()
    torch.testing.assert_close(prototype.norm(dim=-1), torch.ones(2, 2, 2))
    duplicated = np.r_[np.arange(len(y)), [0] * 50]
    repeated, _, _ = build_prototypes(evidence[duplicated], y[duplicated], observed[duplicated],
                                       groups[duplicated], ["a", "b"])
    torch.testing.assert_close(repeated, prototype)
    head = PrototypeReadout(2, 8)
    head.prototypes.copy_(prototype)
    head.active.copy_(active)
    logits = torch.randn(len(y), 2)
    torch.testing.assert_close(head(evidence, logits), logits, atol=0, rtol=0)
    with torch.no_grad():
        head.alpha.fill_(1)
    row_banks = prototype.expand(len(y), -1, -1, -1, -1)
    row_active = active.expand(len(y), -1).clone()
    row_active[:5, 0] = False
    updated = head(evidence, logits, row_banks, row_active)
    torch.testing.assert_close(updated[:5, 0], logits[:5, 0], atol=0, rtol=0)
    torch.testing.assert_close(updated[5:], head(evidence, logits)[5:])
    insufficient = groups.copy()
    insufficient[0] = insufficient[1]
    assert not build_prototypes(evidence, y, observed, insufficient, ["a", "b"])[1].any()


def test_directed_message_mask_probe_gain_and_alignment():
    y = torch.tensor([[1., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
    observed = torch.tensor([[True, True, True], [False, True, True], [True, True, False]])
    base = torch.zeros_like(y)
    evidence = torch.randn(3, 3, 8)
    edges = np.array([[0, 1], [0, 2]])
    head = EvidenceMessages(3, 8, edges)
    torch.testing.assert_close(head(evidence, base), base, atol=0, rtol=0)
    with torch.no_grad():
        head.message[-1].bias.fill_(1)
    first_only = head(evidence, base, edge_mask=torch.tensor([True, False]))
    torch.testing.assert_close(first_only[:, [0, 2]], base[:, [0, 2]], atol=0, rtol=0)
    assert (first_only[:, 1] > base[:, 1]).all()
    torch.testing.assert_close(head(evidence, base, gate_mode="closed"), base, atol=0, rtol=0)
    delta = head.messages(evidence, base)
    loss, gain, valid = message_probe_values(delta, base, y, observed, head.edges)
    assert gain[0, 0] > 0.01 and gain[0, 1] < 0
    assert valid[:, 0].tolist() == [True, False, True]
    filtered, gate_target, gate_mask = align_edge_probes(
        edges, [(np.arange(len(y)), edges[:1], gain[:, :1].detach().numpy(), valid[:, :1].numpy())], len(y))
    np.testing.assert_array_equal(filtered, edges[:1])
    np.testing.assert_array_equal(gate_target[:, 0], [True, False, True])
    np.testing.assert_array_equal(gate_mask[:, 0], [True, False, True])
    loss[valid].mean().backward()
    assert head.message[-1].bias.grad is not None


def test_rank_pair_tie_breaking_constraints_and_gradient():
    y = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
                  [0, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    observed = np.ones_like(y, dtype=bool)
    observed[1, 2] = False
    observed[5, 0] = False
    pairs = matched_pairs(y, observed, [0, 1, 0, 3, 4, 5, 6], ["target", "b", "c"],
                          [20, 20, 20, 22, 22, 20, 20], ["same"] * 7,
                          ["p", "one-bg", "same-family", "bb", "aa", "unknown", "mismatch"],
                          backgrounds=[[1, 2], [], []])
    np.testing.assert_array_equal(pairs, [[0, 4, 0]])
    positive, negative = torch.tensor([0.], requires_grad=True), torch.tensor([1.], requires_grad=True)
    paired_rank_loss(positive, negative).backward()
    assert positive.grad.item() < 0 < negative.grad.item()
    ranking = torch.randn(5, 3)
    decision, correction = DecisionResidual(8)(torch.randn(5, 3, 8), torch.randn(5, 8), ranking)
    torch.testing.assert_close(decision, ranking, atol=0, rtol=0)
    assert correction.count_nonzero() == 0


def test_joint_head_matches_independent_masked_bce_and_marginalizes_partial_labels():
    base = torch.tensor([[0.2, -0.5, 1.], [2., 0.1, -1.], [1., 3., 0.],
                         [1000., -1000., -1000.]], requires_grad=True)
    evidence = torch.randn(4, 3, 8)
    observed = torch.tensor([[True, True, True], [True, False, False],
                             [False, False, True], [True, True, False]])
    y = torch.tensor([[1., 1., 0.], [1., 0., 0.], [0., 0., 0.], [0., 1., 0.]])
    head = PairJointHead(8, [[0, 1]])
    marginal, state = head(evidence, base)
    expected = F.binary_cross_entropy_with_logits(base[observed], y[observed])
    actual = joint_loss(state, marginal, y, observed, head.pairs)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(marginal, base)
    with torch.no_grad():
        head.interaction[-1].bias.fill_(0.5)
    marginal, state = head(evidence, base)
    assert not torch.equal(marginal[:, :2], base[:, :2])
    prob = state.softmax(-1)
    torch.testing.assert_close(marginal[:, 0].sigmoid(), prob[:, 0, 1] + prob[:, 0, 3])
    torch.testing.assert_close(marginal[:, 1].sigmoid(), prob[:, 0, 2] + prob[:, 0, 3])
    actual = joint_loss(state, marginal, y, observed, head.pairs)
    actual.backward()
    assert torch.isfinite(base.grad).all()
    assert base.grad[2, :2].count_nonzero() == 0


def test_four_state_support_controls_edge_and_pair_selection():
    y = np.repeat([[0, 0], [1, 0], [0, 1], [1, 1]], 5, axis=0)
    observed = np.ones_like(y, dtype=bool)
    families = np.arange(len(y))
    labels = ["b", "a"]
    scores = np.zeros_like(y, dtype=float)
    support = four_state_support(y, observed, families)
    np.testing.assert_array_equal(support[0, 1], [5, 5, 5, 5])
    np.testing.assert_array_equal(select_edges(support, labels), [[1, 0], [0, 1]])
    pairs, details = select_joint_pairs(y, observed, families, labels, scores, return_details=True)
    np.testing.assert_array_equal(pairs, [[1, 0]])
    assert details[0]["group_support"] == [5, 5, 5, 5]
    scores[0, 0] = np.nan
    assert not len(select_joint_pairs(y, observed, families, labels, scores))
