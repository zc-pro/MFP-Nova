"""Train, evaluate and deploy the B0/H1–H8 sequence study."""

import copy
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .hypothesis_data import SequenceStudyData
from .hypothesis_evaluation import calibrate, evaluate_rows, predict_sequences
from .hypothesis_training import SequenceTraining
from .models.hypotheses import SequenceHypothesis
from .trainer import seed_everything, write_json


def train_from_cli(args, config):
    from .therapep import configure_task

    unsupported = (args.resume, args.resume_in_place, args.stop_file, args.stage, args.feature_root,
                   args.peptide_feature_root, args.esm, args.metric_suite, args.threshold_strategy,
                   args.loss, args.global_only, args.ablate)
    if any(value is not None and value is not False for value in unsupported):
        raise ValueError("Use the sequence study settings in configs/b0.yaml for B0/H1–H8")
    configure_task(config, args.task or args.track or config["data"]["track"])
    config["model_name"] = args.model
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.annotations:
        config["data"]["annotations"] = args.annotations
    for key in ("device", "precision", "seed", "epochs", "head_epochs", "finetune_epochs", "batch_size",
                "accumulation_steps", "num_workers", "limit_samples", "max_steps", "unlabeled_weight",
                "fold_dir", "evidence_selection"):
        value = getattr(args, key)
        if value is not None:
            config["train"][key] = value
    name = args.name or f"{args.model}_{config['data']['track']}_seed{config['train']['seed']}"
    train(config, name, args.initialize_from)


def load_data(config, data_root=None, annotations=None, *, read_sources=False):
    source = config["data"]
    return SequenceStudyData(data_root or source["root"], source["track"],
                             annotations or source.get("annotations"), read_sources=read_sources)


def take_splits(data, settings):
    train_rows, val_rows = (data.indices(split) for split in ("train", "val"))
    if np.intersect1d(data.clusters[train_rows], data.clusters[val_rows]).size:
        raise ValueError("The selected family grouping overlaps Train and Validation")
    cap = settings.get("limit_samples")
    if cap:
        rng = np.random.default_rng(settings["seed"])
        train_rows, val_rows = (np.sort(rng.choice(rows, min(len(rows), cap), replace=False))
                                for rows in (train_rows, val_rows))
    if not len(train_rows) or not len(val_rows):
        raise ValueError("Train and Validation must contain usable original sequences")
    return train_rows, val_rows


def calibration_json(calibration):
    def encode(value):
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, float) and math.isinf(value):
            return "always_off" if value > 0 else "always_on"
        return value

    return encode(calibration)


def train(config, name, initialize_from=None):
    model_name = config["model_name"]
    settings = config["train"]
    if initialize_from and model_name not in ("h1", "h2", "h3", "h4", "h5", "h6"):
        raise ValueError("Only H1–H6 initialize from a trained B0 checkpoint")
    for key in ("batch_size", "accumulation_steps", "epochs", "head_epochs", "finetune_epochs", "folds",
                "limit_samples", "max_steps"):
        if key in settings and settings[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= settings["unlabeled_weight"] <= 1:
        raise ValueError("unlabeled_weight must be between 0 and 1")
    data = load_data(config, read_sources=model_name in ("h4", "h8"))
    train_rows, validation_rows = take_splits(data, settings)
    run = Path(settings["output"]) / name
    run.mkdir(parents=True, exist_ok=False)
    seed_everything(settings["seed"])
    model = SequenceHypothesis(model_name, data.labels, config["model"], {"mechanism_active": False})
    session = SequenceTraining(config, data, train_rows, run=run)
    started = time.perf_counter()
    if session.device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(session.device)
    negative = data.observed[train_rows] & ~data.targets[train_rows].astype(bool)
    print(f"{model_name.upper()} {data.track} | Train={len(train_rows)} Validation={len(validation_rows)} "
          f"Confirmed negatives={int(negative.sum())} "
          f"Unlabeled weight={settings['unlabeled_weight']}", flush=True)
    session.setup(model, initialize_from)
    model.eval()
    calibration = calibrate(session, model, validation_rows)
    report, predictions = evaluate_rows(session, model, validation_rows, calibration)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    diagnostic = bool(settings.get("limit_samples") or settings.get("max_steps") or session.initialized_from_diagnostic)
    checkpoint = {"model_name": model_name, "model": model.state_dict(), "labels": data.labels,
                  "config": config, "assets": model.assets, "calibration": calibration,
                  "diagnostic_subset": diagnostic, "initialize_from": initialize_from}
    torch.save(checkpoint, run / "final.pt")
    np.savez_compressed(run / "validation_predictions.npz", **predictions)
    write_json(run / "results.json", {
        "model": model_name, "task": data.track, "labels": data.labels, "config": config,
        "git_commit": git.stdout.strip() or None, "initialize_from": initialize_from,
        "device": str(session.device), "torch": str(torch.__version__), "diagnostic_subset": diagnostic,
        "family_grouping": "exact_backbone" if data.track == "rich48" and data.clusters.dtype != object else "homology_groups",
        "train_rows": len(train_rows), "validation_used_for_calibration": True,
        "confirmed_negative_train_bits": int(negative.sum()), "assets": model.assets,
        "phases": session.phases, "calibration": calibration_json(calibration), "validation": report,
        "parameters": sum(param.numel() for param in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(session.device) if session.device.type == "cuda" else None,
    })
    print(f"Saved {run / 'final.pt'}", flush=True)
    return run


def load_model(state, device):
    model = SequenceHypothesis(state["model_name"], state["labels"], state["config"]["model"], state["assets"])
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def evaluate(state, device, split, output, *, data_root=None, annotations=None, batch_size=None,
             predictions=None, diagnostics=False):
    if split == "test" and state["diagnostic_subset"]:
        raise ValueError("Diagnostic training checkpoints are for Validation checks")
    config = copy.deepcopy(state["config"])
    config["train"].update(device=device, precision="fp32", num_workers=0)
    if batch_size:
        config["train"]["batch_size"] = batch_size
    data = load_data(config, data_root, annotations, read_sources=state["model_name"] == "h4")
    if data.labels != state["labels"]:
        raise ValueError("Checkpoint label order differs from the selected task")
    train_rows, validation_rows = take_splits(data, config["train"])
    rows = validation_rows if split == "val" else data.indices(split)
    if split == "test":
        used_groups = data.clusters[np.concatenate([train_rows, data.indices("val")])]
        if np.intersect1d(used_groups, data.clusters[rows]).size:
            raise ValueError("The selected family grouping overlaps Test and Train/Validation")
    session = SequenceTraining(config, data, train_rows)
    model = load_model(state, device)
    report, aligned = evaluate_rows(session, model, rows, state["calibration"], diagnostics=diagnostics)
    report.update(model=model.name, task=data.track, split=split, diagnostic_subset=state["diagnostic_subset"],
                  validation_used_for_calibration=split == "val", calibration=calibration_json(state["calibration"]))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, report)
    if predictions:
        path = Path(predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **aligned)


def predict(state, sequences, device):
    model = load_model(state, device)
    return predict_sequences(model, state["calibration"], sequences, device)
