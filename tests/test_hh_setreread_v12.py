import pytest
import torch

from mfp_nova.models.hh_setreread_v12 import HHSetRereadV12
from test_hh_setreread import small_model
from test_research import synthetic_batch


def model_and_batch(readout="conditional"):
    batch = synthetic_batch(39, input_dim=1280)
    batch["esm_peptide"] = torch.randn(*batch["mask"].shape, 640)
    model = HHSetRereadV12(statistics=small_model("off").statistics, num_labels=39,
                          input_dim=1280, peptide_input_dim=640, vocab_size=10,
                          dim=32, heads=4, cnn_channels=8, pair_dim=8, decoder_depth=1,
                          readout=readout, background_dropout=0., prototype_readout=False,
                          label_context=False)
    return model, batch


@pytest.mark.parametrize("mode", ["conditional", "unconditional", "raw_unconditional", "affine"])
def test_readout_starts_at_base_and_updates_only_trainable_head(mode):
    model, batch = model_and_batch(mode)
    model.train()
    assert not model.backbone.training
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, _, output = model.compute_loss(batch, {}, return_output=True)
        if step == 0:
            torch.testing.assert_close(output["logits"] >= 0, output["unary"] >= 0)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.backbone.parameters())
    if mode == "affine":
        assert model.offset.grad.abs().sum() > 0
    else:
        for projection in (model.residue_projection, model.esm_projection, model.peptide_projection):
            assert projection.weight.grad.abs().sum() > 0
        assert model.fine_queries.grad.abs().sum() > 0


def test_conditional_background_excludes_self_and_respects_padding():
    model, batch = model_and_batch()
    model.eval()
    with torch.no_grad():
        model.label_weights.normal_()
        model.evidence_readout[-1].weight.normal_()
        # Give the fixed base a single predicted antibacterial label.
        model.backbone.correction_gate.weight.zero_()
        model.backbone.correction_gate.bias.fill_(-100.)
        model.backbone.direct[-1].weight.zero_()
        model.backbone.direct[-1].bias.fill_(-1.)
        anchor = model.statistics["labels"].index("anti_gram+")
        model.backbone.direct[-1].bias[anchor] = 1.
        output = model(batch, return_unconditional=True)
        unconditional, conditional = output["unconditional_logits"], output["logits"]
        affected = model.background_axes[:, model.axis_index[anchor]]
        assert affected.any() and (~affected).any()
        torch.testing.assert_close(conditional[:, ~affected], unconditional[:, ~affected])
        assert not torch.allclose(conditional[:, affected], unconditional[:, affected])
        padded = dict(batch, esm=batch["esm"].masked_fill(~batch["mask"][..., None], 100),
                      esm_peptide=batch["esm_peptide"].masked_fill(~batch["mask"][..., None], -100))
        torch.testing.assert_close(model(padded)["logits"], conditional)

        model.backbone.direct[-1].bias[anchor] = -1.
        output = model(batch, return_unconditional=True)
        torch.testing.assert_close(output["logits"], output["unconditional_logits"])
