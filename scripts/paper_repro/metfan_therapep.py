"""Prepare TheraPep CSVs and run the author's complete METFAN method."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from importlib.metadata import version

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

PROJECT = Path(__file__).resolve().parents[2]
KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from mfp_nova.metrics import evaluate_metrics
from mfp_nova.paper_benchmark import CLUSTERED, ROOTS, TRACKS, positive_scope, read_data, traditional5_report


def prepare(task, root, output):
    data = read_data(root, task)
    output.mkdir(parents=True, exist_ok=False)
    info = {"task": task, "data_root": str(root), "labels": data["labels"], "splits": {}}
    for split in ("train", "val", "test"):
        selected = np.flatnonzero(data["splits"] == split)
        kept = [i for i in selected if len(data["sequences"][i]) <= 50]
        folder = output / "data" / split
        folder.mkdir(parents=True)
        with (folder / f"{split}.txt").open("w", encoding="utf-8") as stream:
            for i in kept:
                bits = "".join(map(str, data["y"][i].tolist()))
                stream.write(f">{bits}\n{data['sequences'][i]}\n")
        with (folder / "rows.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row_id", "ID"])
            writer.writerows((int(data["rows"][i]), data["ids"][i]) for i in kept)
        info["splits"][split] = {"source_rows": len(selected), "retained_rows": len(kept)}
    (output / "task.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


def extract(repo, output, esm2, prott5, batch_size):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["METFAN_FEATURES"] = str(output / "features")
    for feature, checkpoint in (("esm2_mean", esm2), ("prot_t5_mean", prott5)):
        if not checkpoint:
            raise ValueError(f"Set the local checkpoint path for {feature}")
        subprocess.run(
            [sys.executable, str(KIT / "metfan_features.py"), feature, str(Path(checkpoint).resolve()),
             "--data-dir", str(output / "data"), "--batch-size", str(batch_size)],
            cwd=repo, env=env, check=True,
        )
    path = output / "task.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    info["features"] = {"esm2": str(Path(esm2).resolve()), "prott5": str(Path(prott5).resolve()),
                        "pooling": "mean residues excluding special/padding tokens",
                        "extraction_dtype": "float16", "saved_dtype": "float32"}
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def train(repo, output, seed, epochs=None):
    info = json.loads((output / "task.json").read_text(encoding="utf-8"))
    clustered = info["task"] in CLUSTERED
    check = ["git", "apply", "--ignore-space-change", "--reverse", "--check"]
    latest = str(KIT / "metfan_traditional5.patch")
    if clustered:
        subprocess.run([*check, latest], cwd=repo, check=True)
    elif subprocess.run([*check, latest], cwd=repo, capture_output=True).returncode:
        subprocess.run([*check, str(KIT / "metfan_therapep.patch")], cwd=repo, check=True)
    training = output / "training" / f"seed{seed}"
    training.mkdir(parents=True, exist_ok=False)
    entry = "from main import main; from config import get_args; args = get_args(); "
    env = os.environ.copy()
    if clustered:
        env["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        entry += f"args.metric_suite = 'traditional5'; args.paper_epochs = {epochs or 999}; "
    entry += "main(args)"
    subprocess.run(
        [sys.executable, "-c", entry,
         "--data_dir", str(output / "data"), "--features_dir", str(output / "features"),
         "--result_dir", str(training), "--num_classes", str(len(info["labels"])),
         "--seed", str(seed), "--features", "textcnn", "esm2_mean", "prot_t5_mean",
         "--optimize_features"],
        cwd=repo, env=env, check=True,
    )
    result = training / "0" / "textcnn_esm2_mean_prot_t5_mean_optimized"
    native = json.loads((result / "results.json").read_text(encoding="utf-8"))
    data = read_data(Path(info["data_root"]), info["task"]) if clustered else None
    for split in ("val", "test"):
        path = result / f"{split}_predictions.npz"
        with np.load(path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        with (output / "data" / split / "rows.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(arrays["targets"]):
            raise ValueError(f"Prediction rows differ from prepared {split} rows")
        if clustered and not np.array_equal(arrays["targets"], data["y"][data["splits"] == split]):
            raise ValueError(f"Prediction labels differ from prepared {split} order")
        if clustered:
            arrays["predicted"] = arrays["probabilities"] >= float(arrays["thresholds"][0])
        np.savez_compressed(
            path, **arrays, row_ids=np.array([int(row["row_id"]) for row in rows]),
            ids=np.array([row["ID"] for row in rows]), label_names=np.array(info["labels"]),
        )
    with np.load(result / "test_predictions.npz") as archive:
        y, p, threshold = archive["targets"], archive["probabilities"], archive["thresholds"]
    history = json.loads((result / "training_history.json").read_text(encoding="utf-8"))
    key = "selection_key" if clustered else "macro_f1"
    best_epoch = max(enumerate(history["val_metrics"], start=1), key=lambda item: item[1][key])[0]
    summary = {
        "model": "METFAN", "task": info["task"], "seed": seed, "best_epoch": best_epoch,
        "label_names": info["labels"], "features": info.get("features"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "transformers": version("transformers"),
                        "numpy": np.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "native_test": native["test_metrics"],
        "test_rows": len(y),
        "ap_definition": "average precision; macro averages classes with positive support",
    }
    if clustered:
        summary.update(traditional5_report(y, p, float(threshold[0]), best_epoch, info["task"], info["labels"],
                                           data["y"][data["splits"] == "train"].sum(0)))
    else:
        summary.update(full_test=evaluate_metrics(y, p, threshold=threshold),
                       **positive_scope(info["task"], y, p, p >= threshold))
    (result / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Completed. Results: {result}", flush=True)


def exact_f1_thresholds(targets, probabilities):
    """Fit F1 at distinct Validation scores; choose the highest threshold on ties."""
    thresholds = np.full(targets.shape[1], 0.5, dtype=float)
    for c in range(targets.shape[1]):
        if np.unique(targets[:, c]).size < 2:
            continue
        precision, recall, candidates = precision_recall_curve(targets[:, c], probabilities[:, c])
        denominator = precision[:-1] + recall[:-1]
        f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                       out=np.zeros_like(denominator), where=denominator != 0)
        thresholds[c] = candidates[np.flatnonzero(f1 == f1.max())[-1]]
    return thresholds


def diagnose(output, seed):
    """Analyze a completed checkpoint without training or replacing native results."""
    result = output / "training" / f"seed{seed}" / "0" / "textcnn_esm2_mean_prot_t5_mean_optimized"
    with np.load(result / "val_predictions.npz") as archive:
        val_y, val_p = archive["targets"], archive["probabilities"]
        labels = archive["label_names"].tolist()
        original = archive["thresholds"]
    # Fit all new decisions before reading Test scores or targets.
    exact = exact_f1_thresholds(val_y, val_p)
    with np.load(result / "test_predictions.npz") as archive:
        test_y, test_p = archive["targets"], archive["probabilities"]
        if not np.array_equal(archive["thresholds"], original):
            raise ValueError("Val and Test must use the same original checkpoint thresholds")
        if archive["label_names"].tolist() != labels:
            raise ValueError("Val and Test label orders differ")
    report = {
        "model": "METFAN", "seed": seed,
        "task": json.loads((output / "task.json").read_text(encoding="utf-8"))["task"],
        "checkpoint": "existing best_model.pth; no retraining or checkpoint reselection",
        "protocol": "Supplementary decision comparison; exact F1 fit on Validation only; "
                    "highest threshold on ties; single-class columns fixed at 0.5. "
                    "Keep original native results; do not choose a strategy by Test performance.",
        "label_names": labels, "strategies": {},
    }
    keys = ("micro_f1", "macro_f1", "sample_jaccard", "exact_match", "hamming_loss")
    for name, thresholds in (("native_val_grid", original),
                             ("fixed_0.5", np.full(len(labels), 0.5)),
                             ("val_exact_f1", exact)):
        entry = {"thresholds": thresholds.tolist()}
        for split, y, p in (("val", val_y, val_p), ("test", test_y, test_p)):
            predicted = p >= thresholds
            entry[split] = {
                "metrics": evaluate_metrics(y, p, predicted=predicted),
                "positive_support": y.sum(0).astype(int).tolist(),
                "predicted_positive_count": predicted.sum(0).astype(int).tolist(),
                "all_positive_labels": [labels[c] for c in np.flatnonzero(predicted.all(0))],
                "mean_predicted_labels": float(predicted.sum(1).mean()),
            }
            if split == "test" and report["task"] not in CLUSTERED:
                positive = y[:, :48].any(1)
                entry["positive_function48"] = evaluate_metrics(
                    y[positive, :48], p[positive, :48], predicted=predicted[positive, :48])
        report["strategies"][name] = entry
        print(name, json.dumps({k: round(entry["test"]["metrics"][k], 6) for k in keys}), flush=True)
    native_counts = report["strategies"]["native_val_grid"]["test"]["predicted_positive_count"]
    print("Native all-positive or zero-threshold columns (Val/Test support, threshold, predicted Test rows):")
    for c, label in enumerate(labels):
        if native_counts[c] == len(test_y) or original[c] == 0:
            print(f"{label}: {int(val_y[:, c].sum())}/{int(test_y[:, c].sum())}, "
                  f"native={original[c]:.8g}, exact={exact[c]:.8g}, Test predicted={native_counts[c]}")
    path = result / "threshold_analysis.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "features", "train", "diagnose"))
    parser.add_argument("--task", choices=TRACKS, default="therapep48")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.home() / "paper-repro/METFAN")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, help="Optional maximum epochs for major15/fine39")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--esm2", default=os.environ.get("METFAN_ESM2"))
    parser.add_argument("--prott5", default=os.environ.get("METFAN_PROTT5"))
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    output, repo = args.output.resolve(), args.repo.resolve()
    if args.action == "prepare":
        root = (args.data_root or PROJECT / ROOTS[args.task]).resolve()
        prepare(args.task, root, output)
    elif args.action == "features":
        extract(repo, output, args.esm2, args.prott5, args.feature_batch_size)
    elif args.action == "diagnose":
        diagnose(output, args.seed)
    else:
        train(repo, output, args.seed, args.epochs)


if __name__ == "__main__":
    main()
