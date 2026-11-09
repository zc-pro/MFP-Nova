import copy
import json
from pathlib import Path

import pytest
import torch
import yaml

from mfp_nova import research
from mfp_nova.models.privilegedbridge import configure_stage
from test_research import multimodal_fixture


def config(tmp_path, name, root, features):
    c = yaml.safe_load(Path(f"configs/{name}.yaml").read_text())
    c["data"].update(root=str(root), track="major15", features=str(features))
    c["encoder"].update(name="fixture", revision="main")
    c["model"].update(dim=24 if name == "privilegedbridge" else 32, dropout=0.)
    if name == "motifspan":
        c["model"].update(cnn_channels=8, relation_dim=4)
    c["train"].update(output=str(tmp_path), epochs=1, batch_size=2, accumulation_steps=2,
                     warmup_epochs=0, ema_decay=.9, ema_start_epoch=1,
                     rdrop_start_epoch=1, rdrop_ramp_epochs=1, save_selection_candidates=False)
    return c


@pytest.mark.parametrize("name,model_options", [
    ("termiscale", {}), ("motifspan", {}), ("motifspan", {"global_only": True}),
], ids=["termiscale", "motifspan", "motifspan-global"])
def test_training_resume_and_evaluate(tmp_path, name, model_options):
    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    c = config(tmp_path, name, root, features)
    c["model"].update(model_options)
    result = research.train(c, "first")
    assert result["epochs_completed"] == 1
    report = research.evaluate(tmp_path/"first/best.pt", root, features, "val", "cpu", tmp_path/"evaluation.json")
    assert report["rows"] == 2
    from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
    import numpy as np
    source = MultimodalData(root, "major15", sequence_only=True)
    dataset = MultimodalPeptides(source, features, "val", sequence_only=True)
    batch = collate_multimodal([dataset[0], dataset[1]])
    np.savez(tmp_path/"inputs.npz", **{k: v.numpy() for k, v in batch.items() if k != "y"})
    predictions = research.predict_inputs(tmp_path/"first/best.pt", tmp_path/"inputs.npz")
    assert len(predictions) == 2 and predictions[0]["evidence_scope"] == "sequence_only"
    if name == "termiscale":
        different = copy.deepcopy(c)
        different["model"]["pooling"] = "uniform"
        with pytest.raises(ValueError, match="pooling differs"):
            research.train(different, "wrong_control", resume=tmp_path/"first/last.pt")
    if model_options.get("global_only"):
        different = copy.deepcopy(c)
        different["model"].pop("global_only")
        with pytest.raises(ValueError, match="global-only mode differs"):
            research.train(different, "wrong_control", resume=tmp_path/"first/last.pt")
    c["train"]["epochs"] = 2
    continued = research.train(c, "continued", resume=tmp_path/"first/last.pt")
    assert continued["epochs_completed"] == 2


def test_pb_full_phases_resume_and_compact_export(tmp_path):
    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    a = config(tmp_path, "privilegedbridge", root, features)
    research.train(a, "a")
    b = config(tmp_path, "privilegedbridge", root, features)
    configure_stage(b, "B")
    b["train"].update(epochs=1, warmup_epochs=0, initialize_from=str(tmp_path/"a/best.pt"))
    b["loss"]["rdrop"] = 0
    research.train(b, "b")
    selected_b, _ = research.load_model(tmp_path/"b/best.pt")
    selected_a, _ = research.load_model(tmp_path/"a/best.pt")
    for key, value in selected_a.sequence.state_dict().items():
        torch.testing.assert_close(value, selected_b.sequence.state_dict()[key], atol=0, rtol=0)
    c = config(tmp_path, "privilegedbridge", root, features)
    configure_stage(c, "C")
    c["train"].update(epochs=1, warmup_epochs=0, ema_start_epoch=1, rdrop_start_epoch=1,
                      initialize_from=str(tmp_path/"b/best.pt"))
    result = research.train(c, "c")
    assert result["epochs_completed"] == 1
    full, _ = research.load_model(tmp_path/"c/best.pt")
    deployed, _ = research.load_model(tmp_path/"c/deployment.pt")
    assert not hasattr(deployed, "teacher")
    from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
    source = MultimodalData(root, "major15")
    data = MultimodalPeptides(source, features, "val")
    batch = collate_multimodal([data[0], data[1]])
    torch.testing.assert_close(full(batch)["logits"], deployed(batch)["logits"])
    c["train"]["epochs"] = 2
    continued = research.train(c, "c_resume", resume=tmp_path/"c/last.pt")
    assert continued["epochs_completed"] == 2


def test_pb_unobserved_batches_do_not_update_teacher(tmp_path):
    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    a = config(tmp_path, "privilegedbridge", root, features)
    research.train(a, "a")
    # Keep inputs present for fitting A; no observations in the B training rows.
    import numpy as np
    for filename, keys in [("physchem_raw.npz", ("X",)),
                           ("structure_primary.npz", ("scalar_X", "residue_mask", "structure_available"))]:
        p = root/"arrays"/filename
        with np.load(p) as z:
            arrays = {k: z[k] for k in z.files}
        for key in keys:
            arrays[key][:4] = False if arrays[key].dtype == bool else np.nan
        np.savez(p, **arrays)
    b = config(tmp_path, "privilegedbridge", root, features)
    configure_stage(b, "B")
    b["train"].update(epochs=1, warmup_epochs=0, initialize_from=str(tmp_path/"a/best.pt"))
    result = research.train(b, "b_missing")
    assert result["global_step"] == 0


def test_sequence_statistics_reject_reordered_labels(tmp_path):
    from mfp_nova.multimodal import MultimodalData
    root, _ = multimodal_fixture(tmp_path)
    source = MultimodalData(root, "major15", sequence_only=True)
    assert not hasattr(source, "phys") and set(source.struct) == {"structure_available"}
    groups = source.groups(source.data.indices("train"), source.data.y[source.data.indices("train")])
    assert groups["structure_present"].sum() == groups["structure_missing"].sum() == 2
    statistics = copy.deepcopy(source.statistics)
    statistics["labels"]["labels"][0:2] = statistics["labels"]["labels"][1::-1]
    with pytest.raises(ValueError, match="label order"):
        MultimodalData(root, "major15", statistics, sequence_only=True)
