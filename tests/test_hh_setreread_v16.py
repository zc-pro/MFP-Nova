from itertools import product
from types import SimpleNamespace

import torch

from mfp_nova.models.hh_setreread_v16 import HHSetRereadV16, observed_set_nll
from test_research import sample_statistics, synthetic_batch


def test_joint_distribution_marginals_and_decoding_match_enumeration():
    torch.manual_seed(16)
    classes, components = 5, 3
    sets = torch.tensor(list(product((0., 1.), repeat=classes))[1:])
    count = sets.sum(-1).long()
    energy = torch.randn(1, components, classes)
    count_logits = torch.randn_like(energy)
    mixture_logits = torch.randn(1, components)
    log_weight = sets @ energy[0].T
    normalized = torch.empty_like(log_weight)
    for size in range(1, classes + 1):
        normalized[count == size] = log_weight[count == size].softmax(dim=0)
    prob = (normalized * count_logits[0].softmax(-1)[:, count - 1].T
            * mixture_logits.softmax(-1)).sum(-1)
    torch.testing.assert_close(prob.sum(), torch.tensor(1.))

    nll = observed_set_nll(energy.expand(len(sets), -1, -1),
                           count_logits.expand(len(sets), -1, -1),
                           mixture_logits.expand(len(sets), -1), sets)
    torch.testing.assert_close(nll.neg().exp(), prob)
    marginal, predicted, top_count = HHSetRereadV16.decode(
        SimpleNamespace(max_output_size=classes),
        {"energies": energy, "count_logits": count_logits, "mixture_logits": mixture_logits},
    )
    torch.testing.assert_close(marginal[0], prob @ sets)
    torch.testing.assert_close(marginal.sum(), prob @ count.float())
    orders = torch.cat((marginal[:, None], energy), dim=1).argsort(dim=-1, descending=True)[0]
    legal = torch.stack([torch.zeros(classes).scatter_(0, order[:size], 1.)
                         for size in range(1, classes + 1) for order in orders])
    overlaps = legal @ sets.T
    utility = (overlaps / (legal.sum(-1, keepdim=True) + count - overlaps)) @ prob
    torch.testing.assert_close(predicted[0].float(), legal[utility.argmax()])
    assert top_count.sum() == torch.bincount(count, weights=prob, minlength=classes + 1).argmax()


def test_fragment_masks_and_joint_gradient():
    batch = synthetic_batch(39)
    batch["edge_type"][2, 1, 4] = batch["edge_type"][2, 4, 1] = 2
    batch["y"][1, :3] = 1.
    batch["base_logits"] = batch["y"] * 4 - 2
    model = HHSetRereadV16(statistics=sample_statistics(39), num_labels=39, input_dim=12,
                            vocab_size=10, dim=32, fragment_dim=16, heads=4, cnn_channels=8,
                            pair_dim=8, dropout=0, decoder_depth=1, prototype_readout=False,
                            label_context=False).eval()
    with torch.no_grad():
        encoded = model(batch)
        corrupted = dict(batch, esm=batch["esm"].masked_fill(~batch["mask"][..., None], 100.),
                         coords=batch["coords"].masked_fill(~batch["struct_mask"][..., None], 100.))
        torch.testing.assert_close(model(corrupted)["energies"], encoded["energies"])
        assert torch.isfinite(encoded["energies"]).all()
    settings = {"unlabeled_weight": .3, "weak": .2, "fragment": .1, "retain": .05}
    model.train()
    model.freeze_backbone(True)
    loss, terms = model.compute_loss(batch, settings)
    loss.backward()
    assert set(terms) == {"joint_nll", "weak", "fragment", "retain"}
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert model.fragment_evidence.contact.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.freeze_backbone(False)
    model.compute_loss(batch, settings)[0].backward()
    assert model.backbone.sequence.plm.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)
