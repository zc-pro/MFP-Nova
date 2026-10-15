import json

import pytest
import torch

from mfp_nova.multimodal import MultimodalData, MultimodalPeptides
from mfp_nova.research import autocast_context, train
from test_research import multimodal_fixture


def test_selected_real_row_matches_esm_position(tmp_path):
    root, features = multimodal_fixture(tmp_path)
    source = MultimodalData(root, "major15")
    source.data.d_mask[1] = False
    source.data.d_mask[3, 1] = True
    dataset = MultimodalPeptides(source, features, "train", limit=3)
    assert dataset.indices.tolist()==[0, 3, 1]
    for j, index in enumerate(dataset.indices):
        expected, _ = dataset.cache[dataset.cache_indices[j]]
        actual = dataset[j]
        assert actual["row_id"].item()==index
        torch.testing.assert_close(expected, actual["esm"])


def test_bounded_accumulated_training_retains_train_priors(tmp_path):
    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    config = {"model_name": "trirelay", "model": {"dim": 32, "heads": 4, "dropout": 0},
              "data": {"root": str(root), "track": "major15", "features": str(features)},
              "encoder": {"name": "fixture", "revision": "main", "max_length": 80},
              "loss": {"mask_ratio": 0, "modality_dropout": 0},
              "train": {"device": "cpu", "seed": 42, "epochs": 10, "batch_size": 1, "learning_rate": .001,
                        "weight_decay": 0, "patience": 0, "warmup_epochs": 0, "num_workers": 0,
                        "output": str(tmp_path), "precision": "fp32", "accumulation_steps": 2, "limit_samples": 3, "max_steps": 2}}
    result = train(config, "bounded")
    checkpoint = torch.load(tmp_path/"bounded/last.pt", weights_only=True)
    assert result["global_step"]==2 and result["epochs_completed"]==1
    assert checkpoint["statistics"]["train_rows"]==4
    assert max(checkpoint["statistics"]["labels"]["support"])==2
    assert all(state["step"].item()==2 for state in checkpoint["optimizer_state"]["state"].values())
    assert result["diagnostic_subset"]
    history = json.loads((tmp_path/"bounded/history.jsonl").read_text().splitlines()[0])
    assert history["skipped_amp_updates"]==0
    config["train"].update(max_steps=3, epochs=2)
    continued = train(config, "continued", resume=tmp_path/"bounded/last.pt")
    assert continued["global_step"]==3
    with pytest.raises(ValueError, match="CUDA"):
        autocast_context("cpu", "fp16")
