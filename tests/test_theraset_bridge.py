import torch

from mfp_nova.models.theraset_bridge import TheraSetBridge
from mfp_nova.research import make_model
from test_research import sample_statistics, synthetic_batch


def test_cluster_bridge_direction_and_supervision_control():
    from mfp_nova.models.theraset_bridge import cross_cluster_bridge

    evidence = torch.tensor([
        [[.1, 1.]], [[1., 0.]], [[1., 0.]], [[-1., 0.]], [[1., 0.]],
    ], requires_grad=True)
    scores = torch.tensor([[-1.], [2.], [2.], [2.], [2.]])
    output = {"evidence_state": evidence, "logits": scores, "evidence_logits": evidence[..., 0] - 1}
    other = dict(output, logits=torch.tensor([[-1.], [2.], [-2.], [2.], [2.]]))
    targets = torch.tensor([[1.], [1.], [1.], [1.], [0.]])
    clusters = torch.tensor([0, 1, 2, 3, 4])
    args = (output, other, targets, clusters, torch.tensor([[1., 0.]]), torch.tensor([-3.]), .5)
    loss, terms = cross_cluster_bridge(*args, "align")
    assert terms["bridge_directed"] == 2 and terms["bridge_pairs"] == 1
    loss.backward(retain_graph=True)
    assert evidence.grad[0, 0, 0] < 0
    assert evidence.grad[1:].count_nonzero() == 0
    align_norm = evidence.grad.norm().clone()
    evidence.grad.zero_()
    control, control_terms = cross_cluster_bridge(*args, "positive")
    assert control_terms["bridge_pairs"] == terms["bridge_pairs"]
    control.backward()
    assert evidence.grad[0, 0, 0] < 0 and evidence.grad[1:].count_nonzero() == 0
    torch.testing.assert_close(evidence.grad.norm(), align_norm)
    empty, empty_terms = cross_cluster_bridge(output, other, targets, torch.zeros_like(clusters), args[4], args[5],
                                              .5, "align")
    assert empty == 0 and empty_terms["bridge_pairs"] == 0


def test_cluster_bridge_averages_cluster_pairs_before_labels():
    from mfp_nova.models.theraset_bridge import cluster_balanced_mean

    values = torch.zeros((2, 4, 4))
    values[0, 0, 1], values[0, 0, 2], values[0, 0, 3] = .2, .4, .7
    values[1, 1, 3] = .3
    active = values > 0
    loss, pairs, labels = cluster_balanced_mean(values, active, active, torch.tensor([0, 1, 1, 2]))
    assert pairs == 3 and labels == 2
    torch.testing.assert_close(loss, torch.tensor(.4))
def test_bridge_model_has_own_identity_and_matches_baseline_with_bridge_off():
    args = dict(input_dim=12, num_labels=39, vocab_size=10, statistics=sample_statistics(39),
                dim=32, heads=4, cnn_channels=8, pair_dim=8, decoder_depth=1,
                prototype_readout=False, label_context=False)
    baseline = make_model("theraset", args)
    bridge = make_model("theraset_bridge", args)
    assert type(baseline) is not type(bridge) and isinstance(bridge, TheraSetBridge)
    bridge.load_state_dict(baseline.state_dict())
    batch = synthetic_batch(labels=39)
    settings = {"rdrop": .1, "bridge": 0., "theraset_bridge": "off"}
    torch.manual_seed(42)
    plain, _, plain_output = baseline.compute_loss(batch, settings, return_output=True)
    torch.manual_seed(42)
    fused, terms, fused_output = bridge.compute_loss(batch, settings, return_output=True)
    assert "bridge" not in terms
    torch.testing.assert_close(plain, fused, rtol=0, atol=0)
    torch.testing.assert_close(plain_output["logits"], fused_output["logits"], rtol=0, atol=0)
    bridge.eval()
    baseline.eval()
    torch.testing.assert_close(bridge(batch)["logits"], baseline(batch)["logits"], rtol=0, atol=0)
