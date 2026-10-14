import json
import random
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CachedPeptides, PeptideData, collate_features
from .losses import asymmetric_loss
from .metrics import evaluate_metrics, fit_global_threshold
from .model import NovaSequenceModel


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    targets, probabilities = [], []
    for residues, mask, y in loader:
        probabilities.append(model(residues.to(device), mask.to(device)).sigmoid().cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fit(model, train_loader, val_loader, config, run, checkpoint_info):
    started = time.perf_counter()
    settings = config["train"]
    device = torch.device(settings["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    loss_settings = dict(config["loss"])
    loss_name = loss_settings.pop("name")
    if loss_name not in ("asl", "bce"):
        raise ValueError(f"Unknown loss: {loss_name}")
    best_key, stale = (-float("inf"), -float("inf")), 0
    for epoch in range(1, settings["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        total, count = 0.0, 0
        for residues, mask, y in train_loader:
            residues, mask, y = residues.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(residues, mask)
            loss = asymmetric_loss(logits, y, **loss_settings) if loss_name == "asl" else torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
        y, probabilities = predict_loader(model, val_loader, device)
        metrics = evaluate_metrics(y, probabilities)
        if metrics["macro_ap"] is None:
            raise ValueError("Validation has no positive labels; Macro-AP model selection is undefined")
        record = {"epoch": epoch, "train_loss": total / count, "validation": metrics,
                  "seconds": time.perf_counter() - epoch_started}
        with (run / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
        key = (metrics["macro_ap"], metrics["micro_ap"])
        if key > best_key:
            best_key, stale = key, 0
            torch.save({**checkpoint_info, "model_state": model.state_dict(), "config": config,
                        "best_epoch": epoch, "validation_at_0_5": metrics}, run / "best.pt")
        else:
            stale += 1
        print(f"epoch={epoch} loss={total/count:.6f} val_macro_ap={metrics['macro_ap']:.6f}", flush=True)
        if settings["patience"] and stale >= settings["patience"]:
            break
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    y, probabilities = predict_loader(model, val_loader, device)
    threshold = fit_global_threshold(y, probabilities)
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, run / "best.pt")
    result = {"best_epoch": checkpoint["best_epoch"], "threshold": threshold,
              "epochs_completed": epoch, "elapsed_seconds": time.perf_counter() - started,
              "threshold_fit_split": "val", "checkpoint": str(run / "best.pt"),
              "validation": evaluate_metrics(y, probabilities, threshold)}
    write_json(run / "results.json", result)
    return result


def train(config, name):
    settings = config["train"]
    seed_everything(settings["seed"])
    data = PeptideData(config["data"]["root"], config["data"]["track"])
    train_data = CachedPeptides(data, config["data"]["features"], "train", config["data"]["augment"])
    val_data = CachedPeptides(data, config["data"]["features"], "val")
    for dataset in (train_data, val_data):
        if dataset.meta["name"] != config["encoder"]["name"] or dataset.meta["max_length"] != config["encoder"]["max_length"]:
            raise ValueError("Configured encoder differs from feature cache")
        if config["encoder"]["revision"] not in (dataset.meta["requested_revision"], dataset.meta["resolved_revision"]):
            raise ValueError("Configured encoder revision differs from feature cache")
    if any(train_data.meta[k] != val_data.meta[k] for k in (
            "resolved_revision", "input_dim", "dataset", "extraction_dtype", "storage_dtype")):
        raise ValueError("Train and Validation feature definitions differ")
    loaders = [DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                          num_workers=settings["num_workers"], collate_fn=collate_features,
                          generator=torch.Generator().manual_seed(settings["seed"]))
               for dataset, shuffle in [(train_data, True), (val_data, False)]]
    model_args = {"input_dim": train_data.meta["input_dim"], "num_labels": len(data.labels), **config["model"]}
    model = NovaSequenceModel(**model_args)
    run = Path(settings["output"]) / name
    run.mkdir(parents=True, exist_ok=False)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    metadata = {"experiment": name, "git_commit": git.stdout.strip() or None, "git_dirty": bool(status.stdout.strip()),
                "config": config, "labels": data.labels, "encoder": train_data.meta,
                "cohort": "resolved_backbone", "train_rows": len(train_data), "val_rows": len(val_data),
                "seed": settings["seed"], "python": sys.version, "torch": str(torch.__version__),
                "packages": {package: version(package) for package in ("numpy", "transformers", "scikit-learn", "PyYAML")},
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "cuda": torch.version.cuda, "device": settings["device"],
                "gpu": torch.cuda.get_device_name(torch.device(settings["device"])) if str(settings["device"]).startswith("cuda") else None}
    write_json(run / "metadata.json", metadata)
    return fit(model, *loaders, config, run, {"model_args": model_args, "labels": data.labels,
                                           "encoder": train_data.meta, "track": data.track})
