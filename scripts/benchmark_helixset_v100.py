"""Measure HelixSet TheraPep49 optimizer throughput on one or two CUDA GPUs."""
import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Subset

from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from mfp_nova.research import epoch_loss_settings, make_model, make_optimizer, to_device, update_ema, validate_cache
from mfp_nova.therapep import configure_task
from mfp_nova.trainer import seed_everything


class TrainingLoss(torch.nn.Module):
    """Expose one complete training loss, including both R-Drop views, to DDP."""

    def __init__(self, model, settings):
        super().__init__()
        self.model = model
        self.settings = settings

    def forward(self, batch):
        loss, _ = self.model.compute_loss(batch, self.settings)
        return loss


def benchmark(steps, warmup, epoch, device, rank, world_size):
    config = yaml.safe_load(Path("configs/helixset.yaml").read_text(encoding="utf-8"))
    configure_task(config, "therapep49")
    data = config["data"]
    settings = config["train"]
    seed = settings["seed"]
    seed_everything(seed)
    source = MultimodalData(data["root"], data["track"])
    dataset = MultimodalPeptides(source, data["features"], "train", peptide_feature_root=data["peptide_features"])
    validate_cache(dataset, config)

    microbatch = settings["batch_size"]
    effective_batch = microbatch * settings["accumulation_steps"]
    accumulation = settings["accumulation_steps"] // world_size
    rows = (warmup + steps) * effective_batch
    if rows > len(dataset):
        raise ValueError(f"Benchmark requests {rows} rows, but Train has {len(dataset)}")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:rows]
    # Preserve each 16-row microbatch (and its padding) when distributing the work.
    local = order.view(-1, accumulation, world_size, microbatch)[:, :, rank, :].flatten().tolist()
    loader = iter(DataLoader(Subset(dataset, local), batch_size=microbatch, collate_fn=collate_multimodal))

    model = make_model("helixset", {
        "input_dim": dataset.meta["input_dim"],
        "peptide_input_dim": dataset.meta["peptide_encoder"]["input_dim"],
        "num_labels": len(source.data.labels),
        "vocab_size": len(source.statistics["chemical_vocab"]) + 2,
        "statistics": source.statistics["labels"],
        **config["model"],
    }).to(device)
    loss_settings = epoch_loss_settings(config, epoch)
    step_model = TrainingLoss(model, loss_settings)
    if world_size == 2:
        step_model = DistributedDataParallel(step_model, device_ids=[device.index],
                                             find_unused_parameters=True, broadcast_buffers=False)
    torch.cuda.manual_seed(seed + rank)
    optimizer = make_optimizer(model, settings)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(settings["ema_decay"]))
    ema.requires_grad_(False)
    if rank == 0:
        print(f"Starting | GPUs={world_size} Batch={effective_batch} RDrop={loss_settings['rdrop']:.3f} "
              f"EMA={epoch >= settings['ema_start_epoch']} Warmup={warmup} Steps={steps}", flush=True)

    for step in range(warmup + steps):
        if step == warmup:
            if world_size == 2:
                dist.barrier(device_ids=[device.index])
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for index in range(accumulation):
            batch = to_device(next(loader), device)
            sync = step_model.no_sync() if world_size == 2 and index + 1 < accumulation else nullcontext()
            with sync:
                (step_model(batch) / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        update_ema(ema, model, epoch, settings["ema_start_epoch"], skipped=False)
    torch.cuda.synchronize(device)

    result = torch.tensor([time.perf_counter() - started, torch.cuda.max_memory_allocated(device) / 2**30],
                          dtype=torch.float64, device=device)
    if world_size == 2:
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
    if rank == 0:
        seconds, peak = result.tolist()
        print(f"Benchmark | GPUs={world_size} EpochSettings={epoch} FP32 Batch={effective_batch} "
              f"Steps={steps} Seconds={seconds:.2f} SecPerStep={seconds / steps:.3f} "
              f"SamplesPerSecond={steps * effective_batch / seconds:.2f} MaxPeakGiB={peak:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--epoch", type=int, default=10, help="Epoch used to set R-Drop and EMA")
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.epoch) < 1:
        parser.error("steps, warmup and epoch must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size not in (1, 2):
        parser.error("Use one GPU or torchrun with two processes")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    if world_size == 2:
        dist.init_process_group("nccl")
    try:
        benchmark(args.steps, args.warmup, args.epoch, device, rank, world_size)
    finally:
        if world_size == 2:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
