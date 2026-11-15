import torch
from torch.nn import functional as F

from mfp_nova.models.hh_setreread_v20 import HHSetRereadV20
from mfp_nova.multimodal import localize_terminal_chemistry


LABELS = ["anti_gram+", "anti_gram_", "antibacterial__unknown", "antifungal__unknown"]
FAMILIES = ["antibacterial", "antifungal", "antiparasitic", "antiviral", "antimicrobial",
            "immunoactive", "anticancer", "antihypertensive", "thrombolytic", "neuropeptide",
            "antioxidant", "metabolic_regulatory", "growth_regeneration", "cell_communication", "drug_delivery"]


def model(mode="full"):
    def stats(names):
        return {"labels": names, "observed_prior": [.12] * len(names), "positive_weights": [1.] * len(names)}
    return HHSetRereadV20(8, len(LABELS), 6, stats(LABELS), stats(FAMILIES),
                          dim=12, cnn_channels=4, dropout=0., mode=mode)


def test_helm_only_terminal_tags_and_incompatible_records():
    assert localize_terminal_chemistry(["[ac]", "[am]", "connection:1"]) == (["connection:1"], [True, True])
    assert localize_terminal_chemistry(["[ac]", "N-terminal modification:Formylation",
                                       "C-terminal modification:Amidation"]) == (
        ["N-terminal modification:Formylation"], [False, True])


def test_major_only_positives_do_not_supervise_fine_detail():
    net = model()
    fine = torch.tensor([[0., 0., 1., 0.], [1., 1., 0., 0.], [0., 0., 0., 1.]])
    major = torch.zeros(3, 15)
    major[1, 2] = 1.
    major[2, 0] = 1.
    family, concrete, mask = net.supervision(fine, major, torch.tensor([True, True, True]))
    assert family[:, :3].tolist() == [[True, False, False], [True, False, True], [True, True, False]]
    assert concrete[:, 0].tolist() == [False, True, False]
    assert mask[:, 0].tolist() == [True, True, False]
    assert not mask[:, 1:].any()


def test_matched_modes_padding_and_backward():
    batch = {"aa": torch.tensor([[1, 2, 3], [4, 3, 0]]),
             "mask": torch.tensor([[True, True, True], [True, True, False]]),
             "d": torch.zeros(2, 3, dtype=torch.long), "chemical": torch.tensor([[1], [0]]),
             "residue_chem": torch.zeros(2, 3, dtype=torch.long),
             "terminal_chemical": torch.tensor([[True, False], [False, True]]),
             "esm": torch.randn(2, 3, 8),
             "y": torch.tensor([[0., 0., 1., 0.], [1., 1., 0., 0.]]),
             "aux_mask": torch.tensor([True, True]), "aux_y": torch.zeros(2, 15)}
    settings = {"family_weight": .3, "detail_weight": .1}
    padded = dict(batch, esm=F.pad(batch["esm"], (0, 0, 0, 4), value=25))
    for key in ("aa", "mask", "d", "residue_chem"):
        padded[key] = F.pad(batch[key], (0, 4))
    for mode in ("direct", "theraset_direct", "auxiliary", "full"):
        net = model(mode).eval()
        if mode == "full":
            with torch.no_grad():
                net.fine_weights.normal_()
        loss, terms, out = net.compute_loss(batch, settings, return_output=True)
        torch.testing.assert_close(out["logits"], net(padded)["logits"])
        assert out["logits"].shape == (2, 4)
        assert ("annotation_detail" in terms) == (mode in ("auxiliary", "full"))
        loss.backward()
        assert torch.isfinite(loss) and net.terminal.weight.grad.abs().sum() > 0
        if mode == "theraset_direct":
            assert net.direct[0].in_features == net.cnn.output_channels
            assert net.peptide[1].weight.grad.abs().sum() > 0
        if mode == "full":
            assert net.fine_queries.grad.abs().sum() > 0
