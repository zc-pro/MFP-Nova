"""Retrain the selected TheraSet architecture on the author's positive48 split."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.chdir(PROJECT)
    output = args.output or Path("runs") / f"theraset_therapep48_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
                       PYTHONPATH=str(PROJECT / "src"), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    cli = [sys.executable, "-u", "-m", "mfp_nova.cli"]

    def run(arguments, log_name):
        print(f"START {log_name} {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
        with (output / log_name).open("x", encoding="utf-8") as log:
            subprocess.run(cli + arguments, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(f"DONE {log_name}", flush=True)

    print(f"RUN={output}", flush=True)
    if not (PROJECT / "data/therapep48/labels.json").exists():
        run(["download-therapep"], "download.log")
        run(["prepare-therapep-variants"], "data.log")
    for view, encoder in (("primary", "esm2_650m"), ("peptide", "esm2_peptide_150m")):
        for split in ("train", "val", "test"):
            cache = PROJECT / "features/therapep_ai" / encoder / split
            if not (cache / "metadata.json").exists():
                run(["features", "--config", "configs/theraset.yaml", "--track", "therapep48",
                     "--encoder-view", view, "--splits", split, "--device", args.device,
                     "--batch-size", "8"], f"features_{view}_{split}.log")

    model = output / "model"
    run(["train", "--model", "theraset", "--task", "therapep48", "--seed", str(args.seed),
         "--ablate", "label_context", "prototype_readout", "--metric-suite", "jaccard",
         "--device", args.device, "--precision", "fp32", "--batch-size", str(args.batch_size),
         "--accumulation-steps", "1", "--num-workers", "0", "--epochs", "100",
         "--name", str(model)], "train.log")
    run(["evaluate", "--checkpoint", str(model / "best.pt"), "--split", "test",
         "--device", args.device, "--batch-size", str(args.batch_size),
         "--output", str(model / "test.json"),
         "--predictions", str(model / "test_predictions.npz")], "test.log")

    training = json.loads((model / "results.json").read_text(encoding="utf-8"))
    test = json.loads((model / "test.json").read_text(encoding="utf-8"))
    summary = {"task": "therapep48", "seed": args.seed, "best_epoch": training["best_epoch"],
               "epochs_completed": training["epochs_completed"], "test_rows": test["rows"],
               "threshold": test["threshold"], "metric_protocol": test["metric_protocol"],
               "validation": training["validation"], "test": test["metrics"]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = output / "acceptance.zip"
    with ZipFile(archive, "x", ZIP_DEFLATED) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".npz", ".yaml", ".log"):
                zipped.write(path, path.relative_to(output))
    print(json.dumps({"run": str(output), "best_epoch": summary["best_epoch"],
                      "test_ACC_pct": 100 * test["metrics"]["sample_jaccard"],
                      "test_MicroF1_pct": 100 * test["metrics"]["micro_f1"],
                      "archive": str(archive)}, ensure_ascii=False, indent=2), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
