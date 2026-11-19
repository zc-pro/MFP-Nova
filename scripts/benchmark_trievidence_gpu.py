"""Alternate two TriEvidence checkpoints on identical GPU-resident Validation batches."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from mfp_nova.models.trievidence import TriEvidence
from mfp_nova.research import to_device, validate_cache
from mfp_nova.trainer import write_json
from run_trievidence import batches, open_data


@torch.inference_mode()
def time_models(models, inputs, repeats):
    for batch in inputs:
        for model in models:
            model(batch)
    torch.cuda.synchronize()

    times = [[], []]
    for repeat in range(repeats):
        for i, batch in enumerate(inputs):
            first = (repeat + i) % 2
            events = []
            for j in (first, 1 - first):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                models[j](batch)
                end.record()
                events.append((j, start, end))
            events[-1][2].synchronize()
            for j, start, end in events:
                times[j].append(start.elapsed_time(end))
    return np.asarray(times)


def benchmark(args):
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Use a CUDA device for the CUDA-event benchmark")
    torch.cuda.set_device(device)
    paths = (args.baseline, args.candidate)
    baseline = torch.load(paths[0], map_location="cpu", weights_only=True)
    _, dataset = open_data(args, "val", baseline)
    if not 1 <= args.batches <= len(dataset) // args.batch_size or args.repeats < 1:
        raise ValueError("Choose a positive number of full batches and repeats")
    full_batches = len(dataset) // args.batch_size
    indices = np.sort(np.random.default_rng(42).choice(full_batches, args.batches, replace=False))
    positions = (indices[:, None] * args.batch_size + np.arange(args.batch_size)).ravel().tolist()
    loader = batches(Subset(dataset, positions), args.batch_size)
    inputs = [to_device(batch, device) for batch in loader]

    models = []
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True) if path != paths[0] else baseline
        if state["model_name"] != "trievidence" or "helixset_args" not in state["model_args"]:
            raise ValueError("Use the original and low-rank three-branch checkpoints")
        validate_cache(dataset, state["config"], state["encoder"])
        model = TriEvidence(**state["model_args"])
        model.load_state_dict(state["model_state"])
        model.reuse_input_compute = True
        models.append(model.to(device).eval())

    full, candidate = time_models(models, inputs, args.repeats)
    difference = candidate - full
    report = {"hardware": torch.cuda.get_device_name(device), "precision": "fp32", "reuse_input_compute": True,
              "batch_size": args.batch_size, "batches": args.batches, "repeats": args.repeats,
              "batch_indices": indices.tolist(),
              "paired_candidate_minus_baseline_p50_ms": float(np.median(difference)),
              "candidate_faster_pairs": int((difference < 0).sum()),
              "paired_measurements": len(full), "models": {}}
    for name, path, model, values in zip(("baseline", "candidate"), paths, models, (full, candidate)):
        report["models"][name] = {"checkpoint": str(path), "parameters": sum(p.numel() for p in model.parameters()),
                                  "p50_ms": float(np.median(values)), "p95_ms": float(np.quantile(values, .95)),
                                  "samples_ms": values.tolist()}
    output = args.output or Path("runs") / datetime.now().strftime("trievidence_gpu_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "results.json", report)
    for entry in report["models"].values():
        entry.pop("samples_ms")
    print(json.dumps(report, indent=2), output, sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    benchmark(args)
