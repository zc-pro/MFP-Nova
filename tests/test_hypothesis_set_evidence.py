from itertools import product

import numpy as np
import torch

from mfp_nova.models.hypothesis_backbone import SequenceBackbone, pad_tokens, tokenize
from mfp_nova.models.hypothesis_evidence import (
    evidence_auxiliary, evidence_diagnostics, evidence_selection, match_random_mask,
)
from mfp_nova.models.hypothesis_set import (
    CardinalitySetHead, cardinality_nll, cardinality_output, count_support,
    independent_cardinality,
)


def enumerate_sets(count_logits, energies):
    labels = energies.shape[-1]
    sets = torch.tensor(list(product((0.0, 1.0), repeat=labels)))
    sizes = sets.sum(-1).long()
    p_count = count_logits.softmax(-1)
    score = (energies[:, sizes] * sets).sum(-1)
    probability = torch.empty_like(score)
    for k in range(labels + 1):
        match = sizes == k
        probability[:, match] = score[:, match].softmax(-1) * p_count[:, k:k + 1]
    return sets, sizes, probability


def test_h3_matches_enumeration_and_partial_likelihood_gradients():
    rng = torch.Generator().manual_seed(28)
    count = torch.randn(5, 5, generator=rng, requires_grad=True)
    energies = torch.randn(5, 5, 4, generator=rng, requires_grad=True)
    targets = torch.tensor([[0, 0, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1],
                            [1, 0, 0, 0], [0, 0, 0, 0]])
    observed = torch.tensor([[1, 1, 1, 1]] * 3 + [[1, 0, 0, 1], [0, 0, 0, 0]]).bool()
    sets, sizes, probabilities = enumerate_sets(count, energies)
    compatible = ((sets[None] == targets[:, None]) | ~observed[:, None]).all(-1)
    expected = -(probabilities * compatible).sum(-1)[:4].log().mean()
    actual = cardinality_nll(count, energies, targets, observed, partial=torch.ones(5, dtype=torch.bool))
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(torch.autograd.grad(actual, (count, energies)),
                                                  torch.autograd.grad(expected, (count, energies))):
        torch.testing.assert_close(actual_gradient, expected_gradient)

    rank = energies[:, 0]
    result = cardinality_output(rank, count, energies)
    expected_joint = torch.stack([probabilities[:, sizes == k] @ sets[sizes == k] for k in range(5)], dim=-1)
    torch.testing.assert_close(result["joint_probabilities"], expected_joint)
    assert result["rank_scores"] is rank
    intersection = sets @ sets.T
    f1 = 2 * intersection / (sizes[:, None] + sizes[None]).clamp_min(1)
    f1[0, 0] = 1
    maximum = (probabilities @ f1).amax(-1)
    winning = (result["predicted"][:, None] == sets[None]).all(-1)
    torch.testing.assert_close(((probabilities @ f1) * winning).sum(-1), maximum)


def test_h3_label_widths_and_empty_set_rule():
    rng = torch.Generator().manual_seed(42)
    for labels in (15, 39, 48):
        logits = torch.randn(2, labels, generator=rng)
        head = CardinalitySetHead(8, labels)
        _, energies = head(torch.randn(2, labels, 8), torch.randn(2, 8), logits)
        torch.testing.assert_close(energies, logits[:, None].expand(-1, labels + 1, -1))
        sizes = torch.arange(labels + 1)
        count, joint = independent_cardinality(logits)
        torch.testing.assert_close(joint.sum(1), count * sizes, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(joint.sum(-1), logits.sigmoid(), rtol=2e-5, atol=2e-5)
        concentration = torch.full((2, labels + 1), -50.0)
        concentration[0, 0] = concentration[1, labels] = 50.0
        prediction = cardinality_output(logits, concentration, energies)["predicted"]
        assert prediction.sum(-1).tolist() == [0, labels]

    support = count_support(np.array([[0, 0], [1, 0], [1, 0], [1, 1]]),
                            np.array([[1, 1], [1, 1], [1, 1], [0, 1]], bool),
                            ["a", "a", "a", "b"])
    assert support.tolist() == [1, 1, 0]


def test_h6_preserves_coordinates_random_counts_and_gradient_scope():
    attention = torch.ones(3, 3, 8, requires_grad=True)
    valid = torch.tensor([[1, 1, 0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 1, 1, 0, 0, 0, 0]]).bool()
    targets = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 1, 0]])
    observed = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 1]]).bool()
    rng = torch.Generator().manual_seed(8)
    rows, labels, keep = evidence_selection((attention, attention), valid, targets, observed, generator=rng)
    assert rows.tolist() == [0] and labels.tolist() == [1]
    assert torch.where(keep[0])[0].tolist() == [0, 1, 3]
    random, _ = match_random_mask(valid[rows], keep, generator=rng)
    segment = torch.arange(8) * 4 // 7
    assert not (random & ~valid[rows]).any()
    for index in range(4):
        assert (random & (segment == index)).sum() == (keep & (segment == index)).sum()

    full_logits = torch.zeros((3, 3), requires_grad=True)
    kept_logits = torch.zeros((1, 3), requires_grad=True)
    evidence_auxiliary(full_logits, kept_logits, rows, labels).backward()
    assert attention.grad is None and full_logits.grad is None
    assert kept_logits.grad[0, 1] < 0
    assert torch.count_nonzero(kept_logits.grad) == 1
    rows, labels, keep = evidence_selection(attention, valid, targets, torch.zeros_like(observed))
    assert keep.shape == (0, 8) and evidence_auxiliary(full_logits, kept_logits[:0], rows, labels) == 0


def test_h6_real_backbone_removes_token_content_and_runs_all_diagnostic_views():
    model = SequenceBackbone(15, dim=12, heads=3, depth=1).eval()
    tokens, mask = pad_tokens([tokenize("CCARWGHKS"), tokenize("LEQ"), tokenize("A")])
    targets = torch.zeros((3, 15), dtype=torch.bool)
    targets[0, [1, 2]] = targets[1, 0] = targets[2, 0] = True
    full = model(tokens, mask, return_attention=True)
    rows, _, keep = evidence_selection(full.attention, mask, targets, targets)
    kept_tokens = tokens[rows]
    retained = model(kept_tokens, keep).logits
    changed = kept_tokens.masked_fill(~keep, 1)
    torch.testing.assert_close(retained, model(changed, keep).logits)
    padded_tokens = torch.nn.functional.pad(kept_tokens, (0, 4))
    padded_mask = torch.nn.functional.pad(keep, (0, 4))
    torch.testing.assert_close(retained, model(padded_tokens, padded_mask).logits)
    diagnostics = evidence_diagnostics(model, tokens, mask, targets, targets)
    assert diagnostics["row_indices"].tolist() == [0, 0, 1]
    assert diagnostics["label_indices"].tolist() == [1, 2, 0]
    assert diagnostics["kept_residues"].tolist() == [5, 5, 2]
    assert diagnostics["random_keep_effect"].shape == (3, 8, 2)
    assert diagnostics["random_delete_effect"].isfinite().all()
    assert diagnostics["random_degenerate"][-1]
