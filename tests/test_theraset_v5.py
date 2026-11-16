import numpy as np
import pytest
import torch

from mfp_nova.models.theraset_v5 import function_adjacency
from test_research import sample_statistics, synthetic_batch
from test_theraset import small_model


LABELS = ["antihiv", "antihcv", "antihsv", "anti_gram+", "anti_gram_",
          "antiplasmodial", "antitrypanosomic", "antiviral__unknown"]


def make_model(mode):
    statistics = sample_statistics(len(LABELS))
    statistics["labels"] = LABELS
    return small_model(len(LABELS), v5_mode=mode, statistics=statistics)


def test_shuffled_graph_preserves_degrees_and_isolation():
    graph = function_adjacency(LABELS)
    shuffled = function_adjacency(LABELS, shuffled=True)
    assert graph[0, 1] and not graph[0, 3] and not graph[-1].any()
    assert not np.array_equal(graph, shuffled)
    np.testing.assert_array_equal(shuffled, shuffled.T)
    np.testing.assert_array_equal(shuffled.sum(1), graph.sum(1))
    assert not shuffled.diagonal().any()


@pytest.mark.parametrize("mode", ("shared", "split", "semantic", "shuffled"))
def test_v5_view_masks_fusion_and_gradients(mode):
    batch = synthetic_batch(len(LABELS))
    model = make_model(mode)
    loss, terms, output = model.compute_loss(batch, {"rdrop": .1, "modality_dropout": 0.}, return_output=True)
    loss.backward()
    assert torch.isfinite(loss) and terms["views"] > 0
    for source in (model.v5_readout.local.attention.in_proj_weight,
                   model.v5_readout.context.attention.in_proj_weight,
                   model.v5_readout.pair.queries.weight):
        assert torch.isfinite(source.grad).all() and source.grad.abs().sum() > 0
    torch.testing.assert_close(output["logits"].sigmoid(), output["view_logits"].sigmoid().mean(1))
    missing = synthetic_batch(len(LABELS), length=1)
    missing["struct_mask"].zero_()
    output = model.eval()(missing)
    assert torch.isfinite(output["logits"]).all()
    if mode != "shared":
        assert not output["view_available"][:, 2].any()
        torch.testing.assert_close(output["logits"].sigmoid(), output["view_logits"][:, :2].sigmoid().mean(1))


def test_split_views_isolate_esm_and_semantic_update_starts_at_zero():
    batch = synthetic_batch(len(LABELS))
    model = make_model("semantic").eval()
    with torch.no_grad():
        first = model(batch)
        direct = model(batch, use_relation=False)
        torch.testing.assert_close(first["logits"], direct["logits"])
        for layer in model.v5_readout.relation_write:
            layer.weight.normal_(std=.1)
        full = model(batch)
        assert not torch.allclose(full["view_logits"][:, :, 0], direct["view_logits"][:, :, 0])
        torch.testing.assert_close(full["view_logits"][:, :, -1], direct["view_logits"][:, :, -1])
        changed = model(dict(batch, esm=batch["esm"].flip(1)))
        torch.testing.assert_close(full["view_logits"][:, (0, 2)], changed["view_logits"][:, (0, 2)])
        assert not torch.allclose(full["view_logits"][:, 1], changed["view_logits"][:, 1])
