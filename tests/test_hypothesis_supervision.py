import numpy as np
import torch

from mfp_nova.models.hypothesis_supervision import (
    distillation_targets,
    fit_observation_fold,
    observation_domain,
    observation_weights,
)


def test_h7_uses_complementarity_and_rejects_identical_teachers():
    group = np.arange(1500)
    targets = (group[:, None] % 5) < 3
    direction = 2 * targets - 1
    first = direction * np.where(group[:, None] % 25 == 1, -17.0, 8.0)
    second = direction * 0.1
    teachers = np.stack((first, second), axis=1)
    observed = np.ones_like(targets)
    observed[4] = False
    teachers[5, 1, 0] = np.nan
    inputs = (targets, observed, group, group % 2, np.full(len(group), 15))

    soft, accepted, eligible = distillation_targets(teachers, *inputs)
    equal, _, _ = distillation_targets(teachers, *inputs, mode="equal")
    best, _, _ = distillation_targets(teachers, *inputs, mode="best")
    assert not eligible[4:6].any()
    assert accepted.sum() > 0 and not (accepted & ~eligible).any()

    def bce(probabilities):
        return -np.log(np.where(targets[accepted], probabilities[accepted], 1 - probabilities[accepted])).mean()

    assert bce(soft) < min(bce(equal), bce(best))
    _, random, _ = distillation_targets(teachers, *inputs, mode="random")
    assert random.sum() == accepted.sum()
    _, rejected, _ = distillation_targets(np.stack((first, first), axis=1), *inputs)
    assert not rejected.any()


def test_h7_does_not_use_a_group_own_targets_to_fit_its_soft_labels():
    group = np.repeat(np.arange(120), 2)
    targets = (group[:, None] % 5) < 3
    direction = 2 * targets - 1
    teachers = np.stack((direction * 3.0, direction * 1.0), axis=1)
    observed = np.ones_like(targets)
    args = (observed, group, group % 2, np.full(len(group), 30))
    original, accepted, _ = distillation_targets(teachers, targets, *args, mode="weighted")
    changed = targets.copy()
    changed[group == 42] = ~changed[group == 42]
    recalibrated, _, _ = distillation_targets(teachers, changed, *args, mode="weighted")
    assert accepted[group == 42].all()
    np.testing.assert_array_equal(original[group == 42], recalibrated[group == 42])


def test_h8_requires_audited_unmeasured_groups_and_limits_weighting_to_known_sources():
    group = np.arange(20)
    source = np.full(20, "lab-a")
    observed = np.column_stack((group < 10, group < 10))
    audited = np.column_stack((np.ones(20, dtype=bool), observed[:, 1]))
    vocabulary, domain = observation_domain(source, observed, audited, group)
    assert domain[vocabulary["lab-a"]].tolist() == [True, False]
    assert not domain[0].any()

    summary = np.random.default_rng(42).normal(size=(20, 8)).astype(np.float32)
    held_source = np.array(["lab-a", "lab-b", "lab-a"])
    propensity, applicable = fit_observation_fold(summary, summary[:3], source, held_source,
                                                   observed, audited, group, epochs=1)
    assert applicable.tolist() == [[True, False], [False, False], [True, False]]
    assert ((propensity > 0) & (propensity < 1)).all()
    held_observed = np.array([[True, True], [True, True], [False, True]])
    weights = observation_weights(propensity, applicable, held_observed)
    assert 1 < weights[0, 0] <= 10
    assert (weights[~(applicable & held_observed)] == 1).all()

    unavailable, applicable = fit_observation_fold(summary, summary[:3], source, held_source,
                                                    observed, observed, group, epochs=1)
    assert not applicable.any()
    np.testing.assert_array_equal(observation_weights(unavailable, applicable, held_observed), 1)
