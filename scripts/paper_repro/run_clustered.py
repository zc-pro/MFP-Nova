"""Run one reproduced method on major15 and fine39 with traditional5 evaluation."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


KIT = Path(__file__).resolve().parent
PROJECT = KIT.parents[1]
MODELS = {"METFAN": "metfan", "iMFP-LG": "imfp", "AMCL": "amcl", "SA-MTP-single": "sa_mtp"}
TASKS = ("major15", "fine39")


def run(script, *args):
    subprocess.run([sys.executable, "-u", str(KIT / script), *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--paper-root", type=Path, default=os.environ.get("PAPER_ROOT"))
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data/therapeutic_2025")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, help="Override the author's maximum training schedule")
    parser.add_argument("--ss2-source", type=Path, help="SA-MTP-single SS2 files from the major15 query export")
    args = parser.parse_args()
    if args.paper_root is None:
        parser.error("Pass --paper-root or set PAPER_ROOT")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    root = args.paper_root.resolve()
    data = args.data_root.resolve()
    output = args.output.resolve() / args.model
    script = f"{MODELS[args.model]}_therapep.py"
    train_args = ["--seed", args.seed]
    if args.epochs is not None:
        train_args.extend(["--epochs", args.epochs])

    if args.model == "SA-MTP-single":
        repo = root / "SA-MTP"
        shared = output / "major15"
        query_source = output / "major15_inputs"
        for task in TASKS:
            inputs = output / f"{task}_inputs"
            query_args = [] if task == "major15" else ["--query-source", query_source]
            run("structure_inputs.py", "--task", task, "--data-root", data, "--output", inputs, *query_args)
            run(script, "prepare", "--repo", repo, "--input", inputs, "--output", output / task)
        if args.ss2_source is None:
            run(script, "ss2", "--repo", repo, "--output", shared)
        else:
            shutil.copytree(args.ss2_source, shared / "ss2")
        run(script, "features", "--repo", repo, "--output", shared, "--esm2",
            root / "pretrained/esm2_t33_650M_UR50D")
        run(script, "graphs", "--repo", repo, "--output", shared)
        for task in TASKS:
            feature_args = [] if task == "major15" else ["--feature-source", shared]
            run(script, "train", "--repo", repo, "--output", output / task, *train_args, *feature_args)
        return

    repo = root / args.model
    for task in TASKS:
        run_dir = output / task
        prepare_args = ["--seed", args.seed] if args.model == "AMCL" else []
        run(script, "prepare", "--repo", repo, "--task", task, "--data-root", data,
            "--output", run_dir, *prepare_args)
        if args.model == "METFAN":
            run(script, "features", "--repo", repo, "--output", run_dir,
                "--esm2", root / "pretrained/esm2_t33_650M_UR50D",
                "--prott5", root / "pretrained/prot_t5_xl_half_uniref50-enc")
        run(script, "train", "--repo", repo, "--output", run_dir, *train_args)


if __name__ == "__main__":
    main()
