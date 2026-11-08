"""Four fixed Train/Val experiments: MotifSpan main, then three TermiScale controls.

Run at the project root. Completed runs are reused, and interrupted runs resume
in place using the existing trainer. No Test evaluation or generated configs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


JOBS = ("motifspan", "uniform_pool", "no_condition", "uniform_scales")


def job_settings(job):
    if job == "motifspan":
        return "motifspan", []
    switches = {"uniform_pool": ["--termiscale-pooling", "uniform"],
                "no_condition": ["--termiscale-pooling", "fixed"],
                "uniform_scales": ["--termiscale-routing", "uniform"]}
    return "termiscale", switches[job]


def command_for(job, task, seed, device, num_workers, batch_size, accumulation_steps):
    model, switches = job_settings(job)
    name = f"{model}_{task}_seed{seed}_{job}_followup"
    run = Path("runs")/name
    command = [sys.executable, "-u", "-m", "mfp_nova.cli", "train", "--model", model,
               "--task", task, "--seed", str(seed), "--device", device,
               "--precision", "fp32", "--batch-size", str(batch_size),
               "--accumulation-steps", str(accumulation_steps),
               "--num-workers", str(num_workers), "--metric-suite",
               "micro_f1" if task == "rich48" else "traditional5",
               "--threshold-strategy", "global", "--epochs", "100", "--name", name, *switches]
    if run.exists():
        command.append("--resume-in-place")
        if (run/"last.pt").exists():
            command.extend(["--resume", str(run/"last.pt")])
    return run, command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", nargs="+", choices=JOBS, default=list(JOBS))
    parser.add_argument("--task", choices=("major15", "fine39", "rich48"), default="fine39")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    args = parser.parse_args(argv)
    if args.batch_size*args.accumulation_steps != 64:
        parser.error("These controlled experiments keep effective batch=64")
    env = dict(os.environ, OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads))
    Path("runs").mkdir(exist_ok=True)
    for job in args.jobs:
        run, command = command_for(job, args.task, args.seed, args.device, args.num_workers,
                                   args.batch_size, args.accumulation_steps)
        if (run/"results.json").exists():
            result = json.loads((run/"results.json").read_text(encoding="utf-8"))
            print(f"Completed: {run}; best epoch={result['best_epoch']}", flush=True)
            continue
        log_path = run.with_suffix(".log")
        print(f"Running: {run}; log={log_path}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Finished: {run}", flush=True)


if __name__ == "__main__":
    main()
