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

PROJECT = Path(__file__).resolve().parents[2]
KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from mfp_nova.metrics import evaluate_metrics
from mfp_nova.therapep import ROOTS, TRACKS, read_data


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


def train(repo, output, seed):
    # This incremental patch requires the existing Val-threshold patch beneath it.
    subprocess.run(["git", "apply", "--ignore-space-change", "--reverse", "--check",
                    str(KIT / "metfan_therapep.patch")], cwd=repo, check=True)
    info = json.loads((output / "task.json").read_text(encoding="utf-8"))
    training = output / "training" / f"seed{seed}"
    training.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [sys.executable, "-c", "from main import main; from config import get_args; main(get_args())",
         "--data_dir", str(output / "data"), "--features_dir", str(output / "features"),
         "--result_dir", str(training), "--num_classes", str(len(info["labels"])),
         "--seed", str(seed), "--features", "textcnn", "esm2_mean", "prot_t5_mean",
         "--optimize_features"],
        cwd=repo, check=True,
    )
    result = training / "0" / "textcnn_esm2_mean_prot_t5_mean_optimized"
    native = json.loads((result / "results.json").read_text(encoding="utf-8"))
    for split in ("val", "test"):
        path = result / f"{split}_predictions.npz"
        with np.load(path) as archive:
            arrays = {name: archive[name] for name in archive.files}
        with (output / "data" / split / "rows.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != len(arrays["targets"]):
            raise ValueError(f"Prediction rows differ from prepared {split} rows")
        np.savez_compressed(
            path, **arrays, row_ids=np.array([int(row["row_id"]) for row in rows]),
            ids=np.array([row["ID"] for row in rows]), label_names=np.array(info["labels"]),
        )
    with np.load(result / "test_predictions.npz") as archive:
        y, p, threshold = archive["targets"], archive["probabilities"], archive["thresholds"]
    positive = y[:, :48].any(1)
    history = json.loads((result / "training_history.json").read_text(encoding="utf-8"))
    best_epoch = int(np.argmax([metrics["macro_f1"] for metrics in history["val_metrics"]])) + 1
    summary = {
        "model": "METFAN", "task": info["task"], "seed": seed, "best_epoch": best_epoch,
        "label_names": info["labels"], "features": info.get("features"),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "transformers": version("transformers"),
                        "numpy": np.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        "native_test": native["test_metrics"],
        "full_test": evaluate_metrics(y, p, threshold=threshold),
        "positive_function48": evaluate_metrics(y[positive, :48], p[positive, :48], threshold=threshold[:48]),
        "test_rows": len(y), "positive_test_rows": int(positive.sum()),
        "background_any_function_fpr": float((p[~positive, :48] >= threshold[:48]).any(1).mean()) if (~positive).any() else None,
        "ap_definition": "average precision; macro averages classes with positive support",
    }
    (result / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Completed. Results: {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "features", "train"))
    parser.add_argument("--task", choices=TRACKS, default="therapep48")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.home() / "paper-repro/METFAN")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--esm2", default=os.environ.get("METFAN_ESM2"))
    parser.add_argument("--prott5", default=os.environ.get("METFAN_PROTT5"))
    args = parser.parse_args()
    output, repo = args.output.resolve(), args.repo.resolve()
    if args.action == "prepare":
        root = (args.data_root or PROJECT / ROOTS[args.task]).resolve()
        prepare(args.task, root, output)
    elif args.action == "features":
        extract(repo, output, args.esm2, args.prott5, args.feature_batch_size)
    else:
        train(repo, output, args.seed)


if __name__ == "__main__":
    main()
