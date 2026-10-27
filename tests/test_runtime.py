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


def test_busy_gpu_returns_before_cuda_context(tmp_path, monkeypatch):
    from mfp_nova import runtime_check
    environment = {"gpu": {"other_compute_processes": [["uuid", "99999", "python", "4000"]]},
                   "cuda_available": True, "caches": {"train": {}, "val": {}}}
    monkeypatch.setattr(runtime_check, "inspect", lambda *args: environment)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: selected.append(device))
    with pytest.raises(RuntimeError, match="other compute jobs"):
        runtime_check.run(["trirelay"], ["major15"], "features", "cuda:3", tmp_path/"gpu")
    assert selected==[] and not (tmp_path/"gpu").exists()


def test_micro_f1_selection_keeps_fixed_threshold_through_evaluation(tmp_path):
    from mfp_nova.research import evaluate
    from mfp_nova.metric_protocol import selection

    # AP and Macro F1 must not break a Micro F1 tie.
    assert selection(None, None, {"micro_f1": .7, "macro_ap": .9}, "micro_f1", "global")[0] == (.7,)
    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    config = {"model_name": "helixset",
              "model": {"dim": 32, "heads": 4, "dropout": .1, "latent_count": 4,
                        "sequence_depth": 1, "property_depth": 1, "geometry_depth": 1,
                        "pair_depth": 1, "latent_depth": 1, "decoder_depth": 1},
              "data": {"root": str(root), "track": "major15", "features": str(features)},
              "encoder": {"name": "fixture", "revision": "main", "max_length": 80},
              "loss": {"modality_dropout": 0},
              "train": {"device": "cpu", "seed": 42, "epochs": 2, "batch_size": 2,
                        "learning_rate": .001, "weight_decay": 0, "patience": 0,
                        "warmup_epochs": 0, "num_workers": 0, "output": str(tmp_path),
                        "metric_suite": "micro_f1", "ema_decay": .9, "ema_start_epoch": 1,
                        "save_selection_candidates": True}}
    result = train(config, "micro")
    history = [json.loads(line) for line in (tmp_path/"micro/history.jsonl").read_text().splitlines()]
    expected = max(history, key=lambda row: row["validation"]["micro_f1"])
    assert result["best_epoch"] == expected["epoch"]
    assert result["threshold"] == result["online_reference"]["threshold"] == .5
    assert result["threshold_fit_split"] is None
    assert "alternate_selection" not in result
    measured = evaluate(tmp_path/"micro/best.pt", root, features, "val", "cpu", tmp_path/"eval.json")
    assert measured["metrics"] == result["validation"]
