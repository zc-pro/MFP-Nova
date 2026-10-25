"""Server-side real-cache probes and bounded full-model GPU measurements."""
import copy
import csv
import gc
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from .research import (autocast_context, make_model, to_device, train, validate_cache,
                       load_model, predict_loader, metrics, make_optimizer)
from .trainer import seed_everything, write_json


def gpu_inventory(device):
    index = torch.device(device).index or 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = visible.split(",")[index] if visible else str(index)
    query = subprocess.run(["nvidia-smi", "-i", physical, "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
                            "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    row = next(csv.reader(query.stdout.splitlines(), skipinitialspace=True))
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                                "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    active = [r for r in csv.reader(processes.stdout.splitlines(), skipinitialspace=True) if len(r)>=4 and r[0]==row[0] and r[1]!=str(os.getpid())]
    return {"host": socket.gethostname(), "device": device, "uuid": row[0], "name": row[1],
            "total_mib": float(row[2]), "used_mib": float(row[3]), "utilization_percent": float(row[4]), "driver": row[5],
            "other_compute_processes": active}


def inspect(feature_root, device):
    gpu = gpu_inventory(device)
    caches = {}
    for split in ("train", "val"):
        root = Path(feature_root)/split
        caches[split] = json.loads((root/"metadata.json").read_text(encoding="utf-8")) if (root/"metadata.json").exists() else None
    return {"gpu": gpu, "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "feature_root": str(feature_root), "caches": caches}


PROBES = {
    "trirelay": ["sequence.plm.weight", "geometry.layers.0.backbone.attention.in_proj_weight", "interactions.0.links.0.attention.in_proj_weight",
                 "feedback.0.attention.in_proj_weight", "relations.factor.0.0.weight"],
    "geofactor": ["sequence.plm.weight", "layers.0.member_up.weight", "layers.0.residue.contact.attention.in_proj_weight",
                  "factors.shared_factor.0.weight", "factors.damping"],
    "protoroute": ["sequence.plm.weight", "collaboration.0.links.0.attention.in_proj_weight", "prototypes.bank", "prototypes.residual", "route.0.weight"],
    "helixset": ["sequence.plm.weight", "geometry_blocks.0.read.attention.in_proj_weight",
                 "pair_geometry.weight", "pair_blocks.0.left.weight", "pair_blocks.0.triangle_left.weight",
                 "pair_blocks.0.geometry_left.weight",
                 "latent_blocks.0.combine.0.weight", "decoder.0.pair_read.queries.weight", "label_weights"]}


@torch.inference_mode()
def check_inputs(model, samples, device, precision):
    model.eval()
    batch = to_device(collate_multimodal(samples[:8]), device)
    with autocast_context(device, precision):
        original = model(batch)
        corrupted = dict(batch)
        corrupted["esm"] = batch["esm"].masked_fill(~batch["mask"][..., None], 100)
        corrupted["geometry"] = batch["geometry"].masked_fill(~batch["struct_mask"][..., None], 100)
        corrupted["coords"] = batch["coords"].masked_fill(~batch["struct_mask"][..., None], 100)
        padded = model(corrupted)
    torch.testing.assert_close(original["logits"], padded["logits"], rtol=.005 if precision!="fp32" else 1e-4, atol=.005 if precision!="fp32" else 1e-4)
    if not torch.isfinite(original["logits"]).all():
        raise FloatingPointError("Nonfinite diagnostic logits")
    for key in ("fusion", "routing"):
        if key in original:
            if not torch.isfinite(original[key]).all():
                raise FloatingPointError(f"Nonfinite {key}")
            missing = ~batch["struct_mask"].any(1)
            if missing.any() and original[key][missing, :, 1].count_nonzero():
                raise ValueError("Missing structures received nonzero fusion/routing weights")
    result = {"logits_shape": list(original["logits"].shape), "logits_finite": True,
              "padding_max_abs_difference": float((original["logits"]-padded["logits"]).abs().max()),
              "D_rows": int(batch["d"].any(1).sum()), "missing_structure_rows": int((~batch["struct_mask"].any(1)).sum())}
    model.train()
    return result


def profile(config, source, dataset, device, batches, precisions, measured_steps=3):
    torch.cuda.set_device(torch.device(device))
    seed_everything(config["train"]["seed"])
    arguments = {"input_dim": dataset.meta["input_dim"], "num_labels": len(source.data.labels),
                 "vocab_size": len(source.statistics["chemical_vocab"])+2, "statistics": source.statistics["labels"], **config["model"]}
    samples = [dataset[i] for i in range(len(dataset))]
    records = []
    for precision in precisions:
        model = make_model(config["model_name"], arguments).to(device).train()
        input_check = check_inputs(model, samples, device, precision)
        parameters = dict(model.named_parameters())
        probe_keys = list(PROBES[config["model_name"]])
        if getattr(model, "esm_tail", None) is not None:
            probe_keys.append("esm_tail.layers.0.attention.self.query.lora_B.weight")
        if hasattr(model, "early_evidence"):
            probe_keys.extend(["early_evidence.sequence_read.attention.in_proj_weight",
                               "early_evidence.property_read.attention.in_proj_weight",
                               "depth_gate.2.weight"])
        optimizer = make_optimizer(model, config["train"])
        scaler = torch.amp.GradScaler("cuda", enabled=precision=="fp16", init_scale=1024.)
        for size in batches:
            torch.cuda.reset_peak_memory_stats(device)
            measurements, module_updates = [], {}
            batch = loss = terms = before = None
            try:
                for step in range(measured_steps+1):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    selected = [samples[j%len(samples)] for j in range(size)]
                    batch = to_device(collate_multimodal(selected), device)
                    events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    optimizer.zero_grad(set_to_none=True)
                    before = {key: parameters[key].detach().clone() for key in probe_keys} if step==1 else {}
                    events[0].record()
                    with autocast_context(device, precision):
                        loss, terms = model.compute_loss(batch, config["loss"])
                    events[1].record()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite {precision} forward loss")
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if step==1:
                        for key in before:
                            gradient = parameters[key].grad
                            finite = bool(torch.isfinite(gradient).all()) if gradient is not None else False
                            module_updates[key] = {"gradient_finite": finite,
                                                   "gradient_abs_sum": float(gradient.float().abs().sum()) if finite else None}
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=precision!="fp16")
                    events[2].record()
                    scale = scaler.get_scale()
                    scaler.step(optimizer); scaler.update()
                    events[3].record()
                    torch.cuda.synchronize(device)
                    if step==1:
                        for key in before:
                            module_updates[key]["updated"] = bool((before[key]!=parameters[key]).any())
                    if step:
                        measurements.append({"loss": float(loss.detach()), "losses": {key: float(value.detach()) for key, value in terms.items()},
                                             "forward_ms": events[0].elapsed_time(events[1]), "backward_ms": events[1].elapsed_time(events[2]),
                                             "optimizer_ms": events[2].elapsed_time(events[3]), "step_seconds": time.perf_counter()-started,
                                             "gradient_finite": bool(torch.isfinite(norm)), "amp_update_skipped": scaler.get_scale()<scale,
                                             "padded_length": batch["aa"].shape[1]})
                successful = all(m["gradient_finite"] and not m["amp_update_skipped"] for m in measurements)
                mean_step = float(np.mean([m["step_seconds"] for m in measurements]))
                records.append({"precision": precision, "batch_size": size, "status": "ok" if successful else "nonfinite_gradient",
                                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/2**20,
                                "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/2**20,
                                "mean_forward_ms": float(np.mean([m["forward_ms"] for m in measurements])),
                                "mean_backward_ms": float(np.mean([m["backward_ms"] for m in measurements])),
                                "mean_step_seconds": mean_step, "samples_per_second": size/mean_step,
                                "estimated_train_epoch_seconds": len(source.data.indices("train"))/size*mean_step,
                                "module_updates": module_updates, "measurements": measurements})
                records[-1]["input_checks"] = input_check
                if not successful:
                    break
            except torch.OutOfMemoryError:
                records.append({"precision": precision, "batch_size": size, "status": "oom"})
                optimizer.zero_grad(set_to_none=True)
                batch = loss = terms = before = None
                break
            del batch, loss, terms, before
        del model, optimizer, scaler, parameters
        gc.collect(); torch.cuda.empty_cache()
    return records


def run(models, tasks, feature_root, device, output, esm="650m", batches=(2, 4, 8, 16),
        precisions=("fp32", "fp16"), steps=32, samples=128, data_root="data/therapeutic_2025"):
    environment = inspect(feature_root, device)
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)
    active = environment["gpu"]["other_compute_processes"]
    # WDDM also lists graphics apps; Linux compute jobs and Windows Python jobs are busy.
    busy = [p for p in active if "python" in p[2].lower() or "train" in p[2].lower() or (os.name!="nt" and p[3].strip().isdigit())]
    if busy:
        raise RuntimeError(f"GPU has other compute jobs: {busy}; select a free GPU")
    if not environment["cuda_available"]:
        raise RuntimeError("Current Python has no usable CUDA; activate the server MFP-Nova environment")
    if any(value is None for value in environment["caches"].values()):
        raise FileNotFoundError("Real Train/Val ESM cache missing; transfer the existing cache, do not substitute random features")
    torch.cuda.set_device(torch.device(device))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"environment.json", environment)
    summary = []
    for name in models:
        for task in tasks:
            config = yaml.safe_load(Path(f"configs/{name}.yaml").read_text(encoding="utf-8"))
            config["data"].update(root=str(data_root), track=task, features=str(feature_root))
            config["encoder"]["name"] = "facebook/esm2_t33_650M_UR50D" if esm=="650m" else "facebook/esm2_t12_35M_UR50D"
            source = MultimodalData(config["data"]["root"], task)
            data = MultimodalPeptides(source, feature_root, "train", samples)
            validate_cache(data, config)
            groups = source.groups(data.indices, source.data.y[data.indices])
            integration = {"rows": len(data), "esm_dim": data.meta["input_dim"], "encoder": data.meta,
                           "statistics_train_rows": source.statistics["train_rows"],
                           "label_order": source.data.labels, "groups": {k: int(v.sum()) for k, v in groups.items()},
                           "empirical_factors": len(source.statistics["labels"]["hyperedges"]),
                           "prototype_counts": source.statistics["labels"]["prototype_counts"]}
            print(f"Checking {name}/{task}: {len(data)} real rows", flush=True)
            records = profile(config, source, data, device, batches, precisions)
            write_json(root/f"{name}_{task}_profile.json", {"integration": integration, "profile": records})
            valid = [r for r in records if r["status"]=="ok" and all(p["gradient_finite"] and p["gradient_abs_sum"]>0 and p["updated"] for p in r["module_updates"].values())]
            if not valid:
                raise RuntimeError(f"{name}/{task} has no successful complete-model profile; inspect saved results")
            chosen = min(valid, key=lambda r: r["mean_step_seconds"]/r["batch_size"])
            config["train"].update(device=device, precision=chosen["precision"], batch_size=chosen["batch_size"],
                                   limit_samples=samples, max_steps=steps, epochs=1000, patience=0,
                                   output=str(root), warmup_epochs=0, aux_warmup_epochs=1)
            del data, source
            gc.collect(); torch.cuda.empty_cache()
            result = train(copy.deepcopy(config), f"{name}_{task}")
            continued = copy.deepcopy(config)
            continued["train"].update(max_steps=steps+2, epochs=result["epochs_completed"]+2)
            restored = train(continued, f"{name}_{task}_resume", resume=root/f"{name}_{task}"/"last.pt")
            model, checkpoint = load_model(root/f"{name}_{task}_resume"/"best.pt", device)
            source = MultimodalData(config["data"]["root"], task, checkpoint["statistics"])
            val = MultimodalPeptides(source, feature_root, "val", min(samples, 64))
            loader = torch.utils.data.DataLoader(val, batch_size=chosen["batch_size"], collate_fn=collate_multimodal)
            y, p = predict_loader(model, loader, device, chosen["precision"])
            evaluation = metrics(y, p, source.statistics, checkpoint["threshold"])
            item = {"model": name, "task": task, "recommendation": {k: chosen[k] for k in ("batch_size", "precision", "peak_allocated_mib", "samples_per_second", "mean_step_seconds")},
                    "train": result, "resume": restored, "independent_validation_rows": len(val), "validation": evaluation,
                    "performance_claim": "diagnostic only; no formal performance conclusion"}
            summary.append(item)
            write_json(root/"summary.json", summary)
            del model, source, val, loader
            gc.collect(); torch.cuda.empty_cache()
    return summary
