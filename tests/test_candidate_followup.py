import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_candidate_followup as followup


def test_four_conditions_change_only_the_named_factor(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for job in followup.JOBS:
        run, command = followup.command_for(job, "fine39", 42, "cuda:0", 2, 64, 1)
        assert command[command.index("--task")+1] == "fine39"
        assert command[command.index("--epochs")+1] == "100"
        assert command[command.index("--batch-size")+1] == "64"
        assert "--resume" not in command
        model, switches = followup.job_settings(job)
        assert command[command.index("--model")+1] == model
        assert len(switches) == (0 if job == "motifspan" else 2)
        assert "--split" not in command


def test_completed_runs_skip_and_incomplete_runs_resume(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run, _ = followup.command_for("uniform_pool", "fine39", 42, "cpu", 0, 64, 1)
    run.mkdir(parents=True)
    (run/"last.pt").write_bytes(b"parent only checks existence; trainer restores this file")
    _, resumed = followup.command_for("uniform_pool", "fine39", 42, "cpu", 0, 64, 1)
    assert "--resume-in-place" in resumed
    assert resumed[resumed.index("--resume")+1] == str(run/"last.pt")
    assert resumed[resumed.index("--termiscale-pooling")+1] == "uniform"
    (run/"results.json").write_text(json.dumps({"best_epoch": 30}))
    calls = []
    monkeypatch.setattr(followup.subprocess, "run", lambda *a, **k: calls.append(a))
    followup.main(["--jobs", "uniform_pool", "--device", "cpu"])
    assert not calls


def test_sequence_order_and_failure_stop(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = []
    def success(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["OMP_NUM_THREADS"] == "6"
        assert kwargs["check"]
    monkeypatch.setattr(followup.subprocess, "run", success)
    followup.main(["--device", "cpu", "--num-workers", "2"])
    assert len(calls) == 4
    assert calls[0][calls[0].index("--model")+1] == "motifspan"
    calls.clear()
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(followup.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        followup.main(["--device", "cpu"])
    assert len(calls) == 1
