from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mfp_nova import cli, research
from mfp_nova.multimodal import GRAM_LABELS, gram_positive_multipliers
from mfp_nova.research_layers import observed_label_loss


def test_balances_existing_gram_positives_and_preserves_total_mass():
    targets = np.array([[1, 0, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0], [1, 1, 1]])
    data = SimpleNamespace(labels=[*GRAM_LABELS, "other"], y=targets,
                           clusters=np.array([10, 20, 20, 30, 20]),
                           indices=lambda split: np.array([3, 0, 2, 1]))
    cluster, _ = gram_positive_multipliers(data, "cluster")
    context, summary = gram_positive_multipliers(data, "cluster_context")
    np.testing.assert_allclose(cluster[:4, :2], [[1.25, 1], [.875, .875], [.875, .875], [1, 1.25]])
    for weights in (cluster, context):
        np.testing.assert_array_equal(weights[targets == 0], 1.)
        np.testing.assert_array_equal(weights[4], 1.)
        np.testing.assert_array_equal(weights[:, 2], 1.)
        np.testing.assert_allclose((weights * targets)[:4].sum(0), targets[:4].sum(0))
    assert summary[GRAM_LABELS[0]]["single_weight_fraction"] == pytest.approx((1.25 / 3 + .5) / 2)
    assert context[0, 0] > cluster[0, 0] and context[3, 1] > cluster[3, 1]


def test_only_the_positive_loss_gradient_is_weighted():
    statistics = {"labels": [*GRAM_LABELS, "other"], "positive_weights": [1., 1., 1.]}
    target = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    logits = torch.zeros_like(target, requires_grad=True)
    loss = observed_label_loss(logits, target, statistics)
    expected, = torch.autograd.grad(loss, logits)
    weights = torch.tensor([[2., 3., 1.], [4., .5, 1.]])
    adjusted = observed_label_loss(logits, target, statistics, positive_multiplier=weights)
    actual, = torch.autograd.grad(adjusted, logits)
    torch.testing.assert_close(actual, expected * torch.where(target.bool(), weights, 1.))


def test_gram_weighting_cli_uses_v5_baseline_protocol(monkeypatch):
    configs = []
    monkeypatch.setattr(research, "train", lambda config, name, **kwargs: configs.append(config))
    cli.main(["train", "--model", "theraset", "--gram-positive-weighting", "cluster_context"])
    config = configs[0]
    assert config["data"]["track"] == "fine39"
    assert config["train"]["v5_mode"] == "baseline"
    assert config["train"]["metric_suite"] == "jaccard"
    assert config["train"]["gram_positive_weighting"] == "cluster_context"
    assert config["model"]["prototype_readout"] is config["model"]["label_context"] is False
