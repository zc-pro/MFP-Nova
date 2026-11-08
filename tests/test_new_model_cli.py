import pytest

from mfp_nova import cli, research, experiments


@pytest.mark.parametrize("name", ["termiscale", "motifspan", "privilegedbridge"])
def test_main_config_and_rich48_task(monkeypatch, name):
    calls = []
    monkeypatch.setattr(research, "train", lambda config, run, **kwargs: calls.append(config))
    cli.main(["train", "--model", name, "--task", "rich48", "--epochs", "2", "--rdrop-weight", "0",
              "--rdrop-start-epoch", "1", "--rdrop-ramp-epochs", "1"])
    config = calls[0]
    assert config["data"]["root"] == "data/therapeutic_2025/rich48"
    assert config["train"]["metric_suite"] == "micro_f1"
    assert config["train"]["epochs"] == 2
    assert config["loss"]["rdrop"] == 0
    assert config["train"]["rdrop_start_epoch"] == config["train"]["rdrop_ramp_epochs"] == 1


@pytest.mark.parametrize("phase", ["B", "C"])
def test_pb_phase_defaults_and_execution_overrides(monkeypatch, phase):
    calls = []
    monkeypatch.setattr(research, "train", lambda config, run, **kwargs: calls.append(config))
    cli.main(["train", "--model", "privilegedbridge", "--stage", phase,
              "--initialize-from", "preceding/best.pt", "--epochs", "2"])
    config = calls[0]
    assert config["model"]["stage"] == phase
    assert config["train"]["epochs"] == 2 and config["train"]["patience"] == 8
    assert config["train"]["initialize_from"] == "preceding/best.pt"
    if phase == "B":
        assert config["loss"]["rdrop"] == 0 and "ema_decay" not in config["train"]
    else:
        assert config["train"]["learning_rate"] == 2e-5
        assert config["train"]["bridge_learning_rate"] == 2e-4


def test_default_sweep_not_expanded():
    assert len(experiments.MODELS) == 5
    assert set(experiments.SUPPORTED_MODELS)-set(experiments.MODELS) == {"termiscale", "motifspan", "privilegedbridge"}


@pytest.mark.parametrize("name", ["termiscale", "motifspan", "privilegedbridge"])
def test_runtime_input_check_accepts_minimal_sequence_batch(name):
    import torch
    from mfp_nova.runtime_check import check_inputs
    from test_research import synthetic_batch, sample_statistics
    batch = synthetic_batch()
    samples = [{k: batch[k][i, :int(batch["mask"][i].sum())] for k in ("aa", "esm", "mask")} for i in range(3)]
    arguments = dict(input_dim=12, num_labels=15, vocab_size=2, statistics=sample_statistics(15),
                     dim=24 if name == "privilegedbridge" else 32, dropout=0.)
    if name == "motifspan":
        arguments.update(cnn_channels=8, relation_dim=4)
    result = check_inputs(research.make_model(name, arguments), samples, "cpu", "fp32")
    assert result["logits_finite"] and result["missing_structure_rows"] is None
