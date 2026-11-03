"""Run the author's iMFP-LG with row-preserving TheraPep CSVs and trained task heads."""
import argparse
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from mfp_nova.metrics import evaluate_metrics
from mfp_nova.therapep import ROOTS, TRACKS, read_data


def prepare(task, root, output):
    data = read_data(root, task)
    output.mkdir(parents=True, exist_ok=False)
    info = {"task": task, "data_root": str(root), "labels": data["labels"], "splits": {}}
    for split in ("train", "val", "test"):
        indices = np.flatnonzero(data["splits"] == split)
        with (output / f"{split}_rows.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row_id", "ID"])
            writer.writerows((int(data["rows"][i]), data["ids"][i]) for i in indices)
        info["splits"][split] = {"rows": len(indices),
                                "max_length": max(len(data["sequences"][i]) for i in indices)}
    (output / "task.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


def train(repo, output, seed, epochs, batch_size):
    info = json.loads((output / "task.json").read_text(encoding="utf-8"))
    root = Path(info["data_root"])
    data = read_data(root, info["task"])
    # The author's Task module imports utils/models from the repository root.
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "Task"))
    from TaskForMultiPeptide_MFTP import TrainConfig, TestModel, train as native_train

    result = output / "training" / f"seed{seed}_epochs{epochs}"
    result.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = TrainConfig(seed)
    cfg.data_name = info["task"]
    cfg.dataset_dir = str(root)
    cfg.train_file_path = str(root / "train.csv")
    cfg.test_file_path = str(root / "test.csv")
    cfg.num_labels = len(info["labels"])
    cfg.epochs = epochs
    cfg.batch_size = batch_size
    cfg.learning_rate = 5e-5
    cfg.train_task_heads = True
    cfg.save_predictions = True
    cfg.model_save_path = str(result / "model.bin")
    cfg.res_path = str(result / "test_predictions")
    print(f"Task={cfg.data_name}, labels={cfg.num_labels}, seed={seed}, epochs={epochs}, "
          f"batch={batch_size}, train_task_heads={cfg.train_task_heads}", flush=True)
    model, adjacency, loader = native_train(cfg)
    validation_loader = loader.load_train_test_data(
        test_file_path=str(root / "val.csv"), only_test=True
    )
    TestModel(validation_loader, adjacency, model, cfg.device, loader.PAD_IDX,
              str(result / "val_predictions"), seed, save_predictions=True)
    for split in ("val", "test"):
        path = result / f"{split}_predictions.npz"
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        indices = np.flatnonzero(data["splits"] == split)
        if not np.array_equal(arrays["targets"], data["y"][indices]):
            raise ValueError(f"{split} prediction labels differ from original CSV row order")
        np.savez_compressed(path, **arrays, row_ids=data["rows"][indices], ids=data["ids"][indices],
                            label_names=np.array(info["labels"]), thresholds=np.full(cfg.num_labels, 0.5))
    with np.load(result / "test_predictions.npz") as archive:
        y, p = archive["targets"], archive["probabilities"]
    predicted = p > 0.5
    positive = y[:, :48].any(1)
    summary = {
        "model": "iMFP-LG", "method_variant": "task_heads_trained",
        "task": info["task"], "seed": seed,
        "epochs": epochs, "selected_epoch": epochs, "selection": "fixed_final_epoch",
        "batch_size": batch_size, "decision": "probability > 0.5",
        "optimizer_setting": "bert lr=5e-5; GAT and all task heads lr=5e-4; original FGM",
        "label_names": info["labels"], "test_rows": len(y),
        "positive_test_rows": int(positive.sum()),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "transformers": version("transformers"),
                        "numpy": np.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "full_test": evaluate_metrics(y, p, predicted=predicted),
        "positive_function48": evaluate_metrics(y[positive, :48], p[positive, :48],
                                                 predicted=predicted[positive, :48]),
        "background_any_function_fpr": float(predicted[~positive, :48].any(1).mean()) if (~positive).any() else None,
    }
    (result / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Completed. Results: {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--task", choices=TRACKS, default="therapep48")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.home() / "paper-repro/iMFP-LG")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.action == "prepare":
        prepare(args.task, (args.data_root or PROJECT / ROOTS[args.task]).resolve(), args.output.resolve())
    else:
        train(args.repo.resolve(), args.output.resolve(), args.seed, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
