"""Time actual 3-layer and 2-layer HelixSet decoders on matched Fine39 batches."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Subset

from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from mfp_nova.research import (
    epoch_loss_settings, load_model, make_optimizer, to_device, update_ema, validate_cache,
)


def time_steps(step, device, warmup, steps, samples_per_step):
    cuda = device.type == "cuda"
    elapsed = []
    for index in range(warmup + steps):
        if cuda and index == warmup:
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        step(index)
        if cuda:
            torch.cuda.synchronize(device)
        if index >= warmup:
            elapsed.append(perf_counter() - started)
    return {
        "mean_ms": 1000 * sum(elapsed) / steps,
        "median_ms": 1000 * median(elapsed),
        "samples_per_second": steps * samples_per_step / sum(elapsed),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if cuda else None,
        "step_ms": [1000 * duration for duration in elapsed],
    }


def benchmark(model, batches, config, device, warmup, steps, epoch):
    settings = config["train"]
    accumulation = settings["accumulation_steps"]
    batch_size = settings["batch_size"]
    loss_settings = epoch_loss_settings(config, epoch)
    torch.manual_seed(settings["seed"])
    model.eval()

    @torch.inference_mode()
    def inference_step(index):
        model(to_device(batches[index * accumulation], device))

    inference = time_steps(inference_step, device, warmup, steps, batch_size)

    model.train()
    optimizer = make_optimizer(model, settings)
    decay = settings.get("ema_decay")
    ema_start = settings.get("ema_start_epoch", 1)
    ema = None
    if decay is not None and epoch >= ema_start:
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
        ema.requires_grad_(False)

    def training_step(index):
        optimizer.zero_grad(set_to_none=True)
        start = index * accumulation
        for batch in batches[start:start + accumulation]:
            loss, _ = model.compute_loss(to_device(batch, device), loss_settings)
            (loss / accumulation).backward()
        clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        update_ema(ema, model, epoch, ema_start, skipped=False)

    training = time_steps(training_step, device, warmup, steps, batch_size * accumulation)
    return {
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "rdrop": loss_settings.get("rdrop", 0),
        "ema": ema is not None,
        "inference": inference,
        "training": training,
    }


def load_batches(checkpoint, config, args):
    data = config["data"]
    stats = checkpoint["statistics"]
    source = MultimodalData(args.data_root or data["root"], "fine39", stats,
                            localize_terminals=stats.get("localized_terminal_chemistry", False))
    dataset = MultimodalPeptides(
        source, args.features or data["features"], "train",
        peptide_feature_root=args.peptide_features or data.get("peptide_features"),
    )
    validate_cache(dataset, config, checkpoint["encoder"])
    settings = config["train"]
    rows = (args.warmup + args.steps) * settings["batch_size"] * settings["accumulation_steps"]
    if rows > len(dataset):
        raise ValueError(f"Benchmark requests {rows} samples but Train has {len(dataset)}")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(settings["seed"]))[:rows]
    loader = DataLoader(Subset(dataset, order.tolist()), batch_size=settings["batch_size"],
                        collate_fn=collate_multimodal)
    return list(loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=24, help="Epoch used for the saved loss and EMA schedules")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--peptide-features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.epoch) < 1:
        parser.error("--warmup, --steps, and --epoch must be positive")

    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint)
    if (checkpoint["model_name"] != "helixset" or checkpoint["track"] != "fine39"
            or len(model.latent_blocks) != 6 or len(model.decoder) != 3
            or hasattr(model, "early_evidence")):
        parser.error("Use the original Fine39 HelixSet best.pt with 6 latent and 3 decoder layers")
    config = checkpoint["config"]
    settings = config["train"]
    run = args.output_dir or Path("runs") / datetime.now().strftime("helixset_depth_speed_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    print("Loading matched Train batches", flush=True)
    batches = load_batches(checkpoint, config, args)
    del checkpoint
    result = {
        "checkpoint": str(args.checkpoint), "device": str(device), "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batch_size": settings["batch_size"], "accumulation_steps": settings["accumulation_steps"],
        "epoch_settings": args.epoch, "precision": "fp32", "seed": settings["seed"],
        "warmup": args.warmup, "steps": args.steps, "arms": {},
    }
    for depth in (3, 2):
        print(f"Benchmarking decoder depth {depth}", flush=True)
        model.decoder = model.decoder[:depth]
        arm = benchmark(model.to(device), batches, config, device, args.warmup, args.steps, args.epoch)
        result["arms"][f"decoder_{depth}"] = arm
        print(f"Decoder {depth} | inference {arm['inference']['mean_ms']:.1f} ms/batch | "
              f"training {arm['training']['mean_ms']:.1f} ms/optimizer step", flush=True)
        del model
        if depth == 3:
            model, checkpoint = load_model(args.checkpoint)
            del checkpoint

    full, slim = (result["arms"][key] for key in ("decoder_3", "decoder_2"))
    result["mean_time_reduction_pct"] = {
        mode: 100 * (1 - slim[mode]["mean_ms"] / full[mode]["mean_ms"])
        for mode in ("inference", "training")
    }
    report = run / "summary.json"
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Mean time reduction (%)", result["mean_time_reduction_pct"])
    print(report.resolve())


if __name__ == "__main__":
    main()
