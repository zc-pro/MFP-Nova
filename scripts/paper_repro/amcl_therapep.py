"""Prepare Train-only AMCL augmentation and train on a TheraPep task."""
import argparse
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
from torch.utils.data import DataLoader, TensorDataset

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from mfp_nova.metrics import evaluate_metrics
from mfp_nova.therapep import ROOTS, TRACKS, read_data


class ConstantFlagDBL(torch.nn.Module):
    """Keep functional DBL and use finite BCE for the constant therapeutic flag."""
    def __init__(self, functional_loss):
        super().__init__()
        self.functional_loss = functional_loss

    def forward(self, logits, labels):
        functional = self.functional_loss(logits[:, :48].clone(), labels[:, :48])
        flag = torch.nn.functional.binary_cross_entropy_with_logits(logits[:, 48], labels[:, 48])
        return (48 * functional + flag) / 49


def prepare(task, root, output, seed):
    from DA import augment_peptide_dataset_with_reversals_and_back_translation, save_augmented_dataset

    data = read_data(root, task)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    indices = np.flatnonzero(data["splits"] == "train")
    seqs = data["sequences"][indices].tolist()
    targets = data["y"][indices]
    augmented, labels = augment_peptide_dataset_with_reversals_and_back_translation(
        seqs, targets, N=3, function_count=48
    )
    save_augmented_dataset(str(output / "augmented_train.txt"), augmented, labels)
    info = {"task": task, "data_root": str(root), "labels": data["labels"],
            "augmentation_seed": seed, "augmentation_N": 3, "augmented_train_rows": len(augmented),
            "single_function_train_rows": int((targets[:, :48].sum(1) == 1).sum()),
            "splits": {}}
    for split in ("train", "val", "test"):
        selected = np.flatnonzero(data["splits"] == split)
        info["splits"][split] = {"rows": len(selected),
                                "pad_length": max(50, max(len(data["sequences"][i]) for i in selected))}
    (output / "task.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


@torch.no_grad()
def export_predictions(model, loader, device, threshold):
    from train import custom_sigmoid_multi_threshold

    model.eval()
    logits, labels, decisions = [], [], []
    t = torch.tensor(threshold, device=device)
    for x, _, y in loader:
        score = model(x.to(device), return_features=False)
        decisions.append((custom_sigmoid_multi_threshold(score, t) > t).cpu().numpy())
        logits.append(score.cpu())
        labels.append(y.numpy())
    scores = torch.cat(logits)
    return np.concatenate(labels), scores.sigmoid().numpy(), scores.numpy(), np.concatenate(decisions)


def train(output, seed, epochs, batch_size):
    info = json.loads((output / "task.json").read_text(encoding="utf-8"))
    os.environ["AMCL_TRAIN_PATH"] = str(output / "augmented_train.txt")
    from DataLoad import getSequenceData, PadEncode
    from losses import FocalDiceLoss, ResampleLoss, gain_class_freq_and_train_num
    from models.TextCNN import TextCNN
    from train import CosineScheduler, DataTrain, optimize_thresholds_with_freq

    data = read_data(Path(info["data_root"]), info["task"])
    result = output / "training" / f"seed{seed}_epochs{epochs}"
    result.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs, labels, _, _ = getSequenceData(str(output / "augmented_train.txt"))
    x, y, lengths = PadEncode(seqs, labels, info["splits"]["train"]["pad_length"])
    train_loader = DataLoader(TensorDataset(x, lengths, y), batch_size=batch_size, shuffle=True)
    loaders = {}
    for split in ("val", "test"):
        selected = np.flatnonzero(data["splits"] == split)
        x, y, lengths = PadEncode(data["sequences"][selected], data["y"][selected],
                                  info["splits"][split]["pad_length"])
        if len(y) != len(selected):
            raise ValueError(f"AMCL encoding unexpectedly dropped {split} rows")
        loaders[split] = DataLoader(TensorDataset(x, lengths, y), batch_size=batch_size, shuffle=False)
    freq, count = gain_class_freq_and_train_num()
    model = TextCNN(21, 128, 128, [3, 4, 5], len(info["labels"]), 0.6).to(device)
    if info["task"] == "therapep49_positive":
        dbl = ConstantFlagDBL(ResampleLoss(class_freq=freq[:48], train_num=count,
                                          reweight_func="rebalance", device=device))
    else:
        dbl = ResampleLoss(class_freq=freq, train_num=count, reweight_func="rebalance", device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
    trainer = DataTrain(model, optimizer, [FocalDiceLoss(), dbl],
                        scheduler=CosineScheduler(epochs, base_lr=0.0007),
                        loss_weights=[0.99, 0.01], device=device, contrastive_label_count=48)
    print(f"Task={info['task']}, outputs={len(info['labels'])}, augmented Train={count}, "
          f"seed={seed}, epochs={epochs}, device={device}", flush=True)
    trainer.train_step(train_loader, epochs=epochs, model_name="AMCL")
    torch.save(model.state_dict(), result / "model.pth")
    threshold = np.asarray(optimize_thresholds_with_freq(
        model, loaders["val"], freq, count, device=device
    ))
    (result / "thresholds.json").write_text(json.dumps(threshold.tolist(), indent=2), encoding="utf-8")
    for split in ("val", "test"):
        selected = np.flatnonzero(data["splits"] == split)
        y, p, logits, predicted = export_predictions(model, loaders[split], device, threshold)
        if not np.array_equal(y, data["y"][selected]):
            raise ValueError(f"Prediction labels differ from original {split} order")
        np.savez_compressed(result / f"{split}_predictions.npz", targets=y, probabilities=p,
                            logits=logits, predicted=predicted, thresholds=threshold,
                            row_ids=data["rows"][selected], ids=data["ids"][selected],
                            label_names=np.array(info["labels"]))
    positive = y[:, :48].any(1)
    summary = {
        "model": "AMCL", "task": info["task"], "seed": seed, "epochs": epochs,
        "selected_epoch": epochs, "selection": "fixed_final_epoch", "batch_size": batch_size,
        "augmentation_seed": info["augmentation_seed"], "augmented_train_rows": count,
        "label_names": info["labels"], "contrastive_function_columns": 48,
        "constant_flag_loss": "functional DBL plus flag BCE, normalized over 49 outputs" if info["task"] == "therapep49_positive" else None,
        "threshold_fit": "Validation only; single-class columns fixed at 0.5",
        "decision": "sigmoid(10*(logit-threshold)) > threshold",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "numpy": np.__version__,
                        "transformers": version("transformers"),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "full_test": evaluate_metrics(y, p, predicted=predicted),
        "positive_function48": evaluate_metrics(y[positive, :48], p[positive, :48],
                                                 predicted=predicted[positive, :48]),
        "test_rows": len(y), "positive_test_rows": int(positive.sum()),
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
    parser.add_argument("--repo", type=Path, default=Path.home() / "paper-repro/AMCL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    repo, output = args.repo.resolve(), args.output.resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    if args.action == "prepare":
        prepare(args.task, (args.data_root or PROJECT / ROOTS[args.task]).resolve(), output, args.seed)
    else:
        train(output, args.seed, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
