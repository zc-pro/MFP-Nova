import copy

import pytest
import torch
import yaml

from mfp_nova.models.theraset import TheraSet, residue_encoding
from mfp_nova.research import make_model
from mfp_nova.runtime_check import PROBES
from test_research import sample_statistics, synthetic_batch


def small_model(labels=15, **options):
    settings = dict(input_dim=12, num_labels=labels, vocab_size=10, statistics=sample_statistics(labels),
                    dim=32, heads=4, dropout=0, cnn_channels=8, pair_dim=8, decoder_depth=1)
    settings.update(options)
    return TheraSet(**settings)


def test_encoding_is_fixed_and_has_expected_residue_order():
    encoding = residue_encoding()
    assert encoding.shape == (21, 34) and encoding[0].count_nonzero() == 0
    torch.testing.assert_close(encoding[1:, :20], torch.eye(20))
    assert encoding[3, 20] == -1  # D
    assert encoding[9, 20] == 1   # K
    assert encoding[6, 27] == 1   # G flag
    assert encoding[13, 28] == 1  # P flag
    assert encoding[2, 29] == 1   # C flag


@pytest.mark.parametrize("labels", [15, 39])
def test_dual_context_gradients_and_evidence_semantics(labels):
    batch = synthetic_batch(labels)
    batch["esm_peptide"] = torch.randn(3, 7, 6)
    model = small_model(labels, peptide_input_dim=6, dropout=.1)
    total, terms = model.compute_loss(batch, {"modality_dropout": 0, "rdrop": .1})
    total.backward()
    assert torch.isfinite(total) and terms["rdrop"] > 0
    parameters = dict(model.named_parameters())
    for key in PROBES["theraset"]:
        assert torch.isfinite(parameters[key].grad).all(), key
        assert parameters[key].grad.abs().sum() > 0, key
    assert model.sequence.plm.weight.grad[:, :12].abs().sum()>0
    assert model.sequence.plm.weight.grad[:, 12:].abs().sum()>0
    assert model.prototype_coefficients.grad[~model.active].count_nonzero() == 0
    model.eval()
    with torch.no_grad():
        out = model(batch, return_evidence=True)
    torch.testing.assert_close(out["logits"], out["direct"]+out["correction"])
    torch.testing.assert_close(out["prototype_weights"].sum(-1), torch.ones(3, labels))
    assert out["prototype_weights"][:, ~model.active].count_nonzero() == 0
    torch.testing.assert_close(out["scale_weights"].sum(-1), torch.ones(3, labels, 4))
    assert out["source_weights"][1, ..., 3].count_nonzero() == 0
    assert out["pair_attention"].diagonal(dim1=-2, dim2=-1).count_nonzero() == 0
    torch.testing.assert_close(out["pair_attention"].sum((-1, -2)), torch.ones(3, labels))


def test_missing_values_padding_length_and_rigid_motion_invariance():
    batch = synthetic_batch()
    model = small_model().eval()
    batch["phys_mask"][:, 0:50] = False
    batch["struct_global_mask"][1] = False
    with torch.no_grad():
        original = model(batch)["logits"]
        changed = {k: v.clone() for k, v in batch.items()}
        for name, mask in (("esm", batch["mask"]), ("geometry", batch["struct_mask"]),
                           ("coords", batch["struct_mask"]), ("phys", batch["phys_mask"]),
                           ("struct_global", batch["struct_global_mask"])):
            missing = ~mask[..., None] if changed[name].ndim>mask.ndim else ~mask
            changed[name].masked_fill_(missing, 100)
        torch.testing.assert_close(original, model(changed)["logits"])
        changed = dict(batch, coords=batch["coords"] @ torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])+17)
        torch.testing.assert_close(original, model(changed)["logits"], atol=2e-6, rtol=1e-5)
        # Same peptide alone at length 3, versus padded alongside longer peptides.
        single = {key: value[:1].clone() for key, value in batch.items()}
        for key in ("esm", "aa", "d", "residue_chem", "mask", "coords", "geometry", "struct_mask", "dssp", "confidence_known"):
            single[key] = single[key][:, :3]
        single["edge_type"] = single["edge_type"][:, :3, :3]
        torch.testing.assert_close(original[:1], model(single)["logits"], atol=3e-6, rtol=1e-5)


def test_single_residue_and_completely_absent_optional_modalities():
    batch = synthetic_batch(length=1)
    batch.update(struct_mask=torch.zeros_like(batch["mask"]), phys_mask=torch.zeros_like(batch["phys_mask"]))
    model = small_model(contextual_evidence=False).eval()
    out = model(batch, return_evidence=True)
    assert torch.isfinite(out["logits"]).all()
    assert out["pair_attention"].count_nonzero() == 0
    assert out["source_weights"][..., 1:].count_nonzero() == 0
    out["logits"].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_structure_ablation_removes_every_structure_route():
    batch = synthetic_batch()
    model = small_model(structure_evidence=False).eval()
    changed = dict(batch, geometry=100*torch.randn_like(batch["geometry"]),
                   coords=100*torch.randn_like(batch["coords"]),
                   struct_global=100*torch.randn_like(batch["struct_global"]),
                   dssp=torch.full_like(batch["dssp"], 9), edge_type=torch.full_like(batch["edge_type"], 2))
    with torch.no_grad():
        torch.testing.assert_close(model(batch)["logits"], model(changed)["logits"])


def test_quality_zero_does_not_become_full_strength_after_normalization():
    batch = synthetic_batch()
    batch["geometry"][..., 9] = 0
    model = small_model().eval()
    no_structure = dict(batch, struct_mask=torch.zeros_like(batch["mask"]))
    with torch.no_grad():
        torch.testing.assert_close(model(batch)["logits"], model(no_structure)["logits"])


def test_duplicate_prototypes_do_not_create_count_bonus():
    batch = synthetic_batch()
    model = small_model().eval()
    with torch.no_grad():
        model.prototype_coefficients.copy_(model.prototype_coefficients[:, :1].expand_as(model.prototype_coefficients))
        multi = model(batch)["logits"]
        model.active[:, 1:] = False
        torch.testing.assert_close(multi, model(batch)["logits"], atol=2e-6, rtol=1e-5)


def test_cnn_control_is_independent_of_auxiliary_inputs_and_trains():
    batch = synthetic_batch()
    model = small_model(evidence_decoder=False, chemical_evidence=False)
    initial = model.compute_loss(batch, {"modality_dropout": 0})[0].detach()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(8):
        optimizer.zero_grad()
        loss, _ = model.compute_loss(batch, {"modality_dropout": 0})
        loss.backward()
        optimizer.step()
    assert loss < initial
    model.eval()
    with torch.no_grad():
        # The adapted CNN consumes only aa/mask, even when other inputs are absent.
        torch.testing.assert_close(model(batch)["logits"], model({"aa": batch["aa"], "mask": batch["mask"]})["logits"])


def test_full_model_short_optimization_and_checkpoint():
    batch = synthetic_batch()
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    initial = model.compute_loss(batch, {"modality_dropout": 0})[0].detach()
    for _ in range(8):
        optimizer.zero_grad()
        loss, _ = model.compute_loss(batch, {"modality_dropout": 0})
        loss.backward()
        optimizer.step()
    assert loss < initial
    restored = small_model()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    with torch.no_grad():
        torch.testing.assert_close(model.eval()(batch)["logits"], restored.eval()(batch)["logits"])


def test_cpu_bfloat16_forward_backward_is_finite():
    batch = synthetic_batch()
    model = small_model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, _ = model.compute_loss(batch, {"modality_dropout": 0, "rdrop": .1})
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_default_config_and_ablations_construct():
    from pathlib import Path
    config = yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/theraset.yaml").read_text())
    args = dict(input_dim=12, num_labels=15, vocab_size=10, statistics=sample_statistics(15), **config["model"])
    model = make_model("theraset", args)
    batch = synthetic_batch()
    with torch.no_grad():
        assert torch.isfinite(model.eval()(batch)["logits"]).all()
    for switch, value in config["model"].items():
        if isinstance(value, bool):
            candidate = small_model(**{switch: False}).eval()
            assert torch.isfinite(candidate(batch)["logits"]).all(), switch
