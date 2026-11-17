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
from .therapep import TRACKS as THERAPEP_TRACKS, ROOTS as THERAPEP_ROOTS

import numpy as np
import torch
import yaml
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from .models import HH_MODELS, HH_V16, HH_V20, PEPTIDE_V6, PEPTIDE_V7
from .research import (autocast_context, make_model, to_device, train, validate_cache,
                       load_model, predict_loader, metrics, make_optimizer, update_ema, v16_count_support)
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
    PEPTIDE_V6: ["cnn.convs.0.0.weight", "primary.1.weight", "peptide.1.weight",
                 "local.0.0.weight", "interaction.weight", "label_query"],
    PEPTIDE_V7: ["cnn.convs.0.0.weight", "features.projection.weight",
                 "attention.in_proj_weight", "label.weight"],
    "hh_setreread": ["backbone.sequence.plm.weight", "query", "shared_readout.weight", "private_readout.weight"],
    "hh_setreread_v12": ["residue_projection.weight", "esm_projection.weight", "peptide_projection.weight",
                       "fine_queries", "evidence_readout.2.weight"],
    HH_V16: ["backbone.sequence.plm.weight", "label_weights", "count.weight",
             "fragment_evidence.fragment.weight"],
    HH_V20: ["cnn.convs1.0.weight", "peptide.1.weight", "family_queries", "fine_weights"],
    "setreread": ["features.projection.weight", "cnn.convs1.0.weight", "add_head.2.weight"],
    "termiscale": ["features.projection.weight", "blocks.0.router.weight", "label.weight", "condition_weights"],
    "motifspan": ["features.projection.weight", "velocity_out.weight", "left.weight", "detail_query.weight"],
    "privilegedbridge": ["sequence.features.projection.weight", "sequence.label_weight"],
    "trirelay": ["sequence.plm.weight", "geometry.layers.0.backbone.attention.in_proj_weight", "interactions.0.links.0.attention.in_proj_weight",
                 "feedback.0.attention.in_proj_weight", "relations.factor.0.0.weight"],
    "geofactor": ["sequence.plm.weight", "layers.0.member_up.weight", "layers.0.residue.contact.attention.in_proj_weight",
                  "factors.shared_factor.0.weight", "factors.damping"],
    "protoroute": ["sequence.plm.weight", "collaboration.0.links.0.attention.in_proj_weight", "prototypes.bank", "prototypes.residual", "route.0.weight"],
    "helixset": ["sequence.plm.weight", "geometry_blocks.0.read.attention.in_proj_weight",
                 "pair_geometry.weight", "pair_blocks.0.left.weight", "pair_blocks.0.triangle_left.weight",
                 "pair_blocks.0.geometry_left.weight",
                 "latent_blocks.0.combine.0.weight", "decoder.0.pair_read.queries.weight", "label_weights"],
    "theraset": ["cnn.convs1.0.weight", "sequence.plm.weight", "pairs.left.weight",
                 "pairs.geometry.weight", "prototype_bank", "decoder.0.pair.queries.weight",
                 "decoder.0.factor_read.attention.in_proj_weight", "label_weights"]}


@torch.inference_mode()
def check_inputs(model, samples, device, precision):
    model.eval()
    batch = to_device(collate_multimodal(samples[:8]), device)
    with autocast_context(device, precision):
        original = model(batch)
        corrupted = dict(batch)
        corrupted["esm"] = batch["esm"].masked_fill(~batch["mask"][..., None], 100)
        if "esm_peptide" in batch:
            corrupted["esm_peptide"] = batch["esm_peptide"].masked_fill(~batch["mask"][..., None], 100)
        if "geometry" in batch:
            corrupted["geometry"] = batch["geometry"].masked_fill(~batch["struct_mask"][..., None], 100)
            corrupted["coords"] = batch["coords"].masked_fill(~batch["struct_mask"][..., None], 100)
        padded = model(corrupted)
    torch.testing.assert_close(original["logits"], padded["logits"], rtol=.005 if precision!="fp32" else 1e-4, atol=.005 if precision!="fp32" else 1e-4)
    if not torch.isfinite(original["logits"]).all():
        raise FloatingPointError("Nonfinite diagnostic logits")
    if "energies" in original:
        for key in ("energies", "count_logits", "mixture_logits"):
            torch.testing.assert_close(original[key], padded[key],
                                       rtol=.005 if precision != "fp32" else 1e-4,
                                       atol=.005 if precision != "fp32" else 1e-4)
    for key in ("fusion", "routing"):
        if key in original:
            if not torch.isfinite(original[key]).all():
                raise FloatingPointError(f"Nonfinite {key}")
            missing = ~batch["struct_mask"].any(1)
            if missing.any() and original[key][missing, :, 1].count_nonzero():
                raise ValueError("Missing structures received nonzero fusion/routing weights")
    result = {"logits_shape": list(original["logits"].shape), "logits_finite": True,
              "padding_max_abs_difference": float((original["logits"]-padded["logits"]).abs().max()),
              "D_rows": int(batch["d"].any(1).sum()) if "d" in batch else 0,
              "missing_structure_rows": int((~batch["struct_mask"].any(1)).sum()) if "struct_mask" in batch else None}
    model.train()
    return result


def profile(config, source, dataset, device, batches, precisions, measured_steps=3,
            batch_scope="diagnostic"):
    torch.cuda.set_device(torch.device(device))
    seed_everything(config["train"]["seed"])
    arguments = {"input_dim": dataset.meta["input_dim"], "num_labels": len(source.data.labels),
                 "vocab_size": len(source.statistics["chemical_vocab"])+2, "statistics": source.statistics["labels"], **config["model"]}
    if config["model_name"] == "setreread":
        arguments["function_projection"] = source.statistics["function_projection"]
    if config["model_name"] == "hh_setreread":
        from .models.hh_setreread import axis_statistics
        arguments["axis_statistics"] = axis_statistics(source)
    if config["model_name"] == HH_V16:
        arguments["count_support"] = v16_count_support(source.data)
    if config["model_name"] == HH_V20:
        arguments["family_statistics"] = source.statistics["function_auxiliary"]
    if config["model_name"] == PEPTIDE_V7 and arguments.get("head", "structured") == "structured":
        from .models.peptide_v7 import fit_axis_counts
        arguments["axis_counts"] = fit_axis_counts(source.data)
    if "peptide_encoder" in dataset.meta:
        arguments["peptide_input_dim"] = dataset.meta["peptide_encoder"]["input_dim"]
    samples = [dataset[i] for i in range(len(dataset))] if batch_scope == "diagnostic" else None
    if batch_scope == "longest_train":
        lengths = [len(source.data.sequences[row]) for row in dataset.indices]
        longest = np.argsort(-np.asarray(lengths), kind="stable")
    input_samples = samples if samples is not None else [dataset[i] for i in range(min(len(dataset), 8))]
    records = []
    for precision in precisions:
        rng = np.random.default_rng(config["train"]["seed"])
        model = make_model(config["model_name"], arguments).to(device).train()
        if config["model_name"] == "privilegedbridge" and model.stage != "A":
            previous, saved = load_model(config["train"]["initialize_from"], device)
            model.initialize_from(previous)
            del previous
        input_check = check_inputs(model, input_samples, device, precision)
        parameters = dict(model.named_parameters())
        probe_keys = list(PROBES[config["model_name"]])
        if config["model_name"] == PEPTIDE_V7 and model.head == "structured":
            probe_keys.extend(("state.weight", "child.weight"))
        if config["model_name"] == "privilegedbridge":
            if model.stage == "B":
                probe_keys = ["teacher.residual.0.0.weight", "teacher.residual.1.0.weight"]
            elif model.stage == "C":
                probe_keys = ["sequence.features.projection.weight", "bridges.0.2.weight", "bridges.1.2.weight"]
        if config["data"]["track"] in THERAPEP_TRACKS:
            probe_keys = [key for key in probe_keys if not any(
                absent in key for absent in ("geometry", "interactions.", "collaboration."))]
        if getattr(model, "esm_tail", None) is not None:
            probe_keys.append("esm_tail.layers.0.attention.self.query.lora_B.weight")
        if hasattr(model, "early_evidence"):
            probe_keys.extend(["early_evidence.sequence_read.attention.in_proj_weight",
                               "early_evidence.property_read.attention.in_proj_weight",
                               "depth_gate.2.weight"])
        optimizer = make_optimizer(model, config["train"])
        scaler = torch.amp.GradScaler("cuda", enabled=precision=="fp16", init_scale=1024.)
        ema_decay = config["train"].get("ema_decay") if samples is None else None
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)).requires_grad_(False) if ema_decay else None
        for size in batches:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            measurements, module_updates = [], {}
            batch = loss = terms = before = None
            try:
                for step in range(measured_steps+2):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    if samples is not None:
                        selected = [samples[j%len(samples)] for j in range(size)]
                    else:
                        positions = (longest[:size] if batch_scope == "longest_train" else
                                     rng.choice(len(dataset), size, replace=False))
                        selected = [dataset[int(position)] for position in positions]
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
                    skipped = scaler.get_scale() < scale
                    update_ema(ema, model, epoch=1, start_epoch=1, skipped=skipped)
                    events[3].record()
                    torch.cuda.synchronize(device)
                    if step==1:
                        for key in before:
                            module_updates[key]["updated"] = bool((before[key]!=parameters[key]).any())
                        # V1.2 and V6 already probe each encoder's separate projection in PROBES.
                        if "peptide_encoder" in dataset.meta and config["model_name"] not in ("hh_setreread_v12", PEPTIDE_V6):
                            key = ("features.projection.weight" if config["model_name"] in ("termiscale", "motifspan", PEPTIDE_V7) else
                                   "sequence.features.projection.weight" if config["model_name"] == "privilegedbridge" else
                                   "backbone.sequence.plm.weight" if config["model_name"] in ("hh_setreread", HH_V16) else
                                   "sequence.plm.weight")
                            boundary = dataset.meta["input_dim"]
                            for view, columns in (("primary", slice(None, boundary)), ("peptide", slice(boundary, None))):
                                gradient = parameters[key].grad[:, columns]
                                module_updates[f"{key}:{view}"] = {
                                    "gradient_finite": bool(torch.isfinite(gradient).all()),
                                    "gradient_abs_sum": float(gradient.float().abs().sum()),
                                    "updated": bool((before[key][:, columns] != parameters[key][:, columns]).any())}
                    if step >= 2:
                        measurements.append({"loss": float(loss.detach()), "losses": {key: float(value.detach()) for key, value in terms.items()},
                                             "forward_ms": events[0].elapsed_time(events[1]), "backward_ms": events[1].elapsed_time(events[2]),
                                             "optimizer_ms": events[2].elapsed_time(events[3]), "step_seconds": time.perf_counter()-started,
                                             "gradient_finite": bool(torch.isfinite(norm)), "amp_update_skipped": skipped,
                                             "padded_length": batch["aa"].shape[1]})
                successful = all(m["gradient_finite"] and not m["amp_update_skipped"] for m in measurements)
                mean_step = float(np.mean([m["step_seconds"] for m in measurements]))
                records.append({"precision": precision, "batch_size": size, "batch_scope": batch_scope,
                                "status": "ok" if successful else "nonfinite_gradient",
                                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/2**20,
                                "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/2**20,
                                "mean_forward_ms": float(np.mean([m["forward_ms"] for m in measurements])),
                                "mean_backward_ms": float(np.mean([m["backward_ms"] for m in measurements])),
                                "mean_step_seconds": mean_step, "samples_per_second": size/mean_step,
                                "estimated_train_epoch_seconds": (None if batch_scope == "longest_train" else
                                                                  len(source.data.indices("train"))/size*mean_step),
                                "ema_included": ema is not None,
                                "module_updates": module_updates, "measurements": measurements})
                records[-1]["input_checks"] = input_check
                if not successful:
                    break
            except torch.OutOfMemoryError:
                records.append({"precision": precision, "batch_size": size, "batch_scope": batch_scope, "status": "oom"})
                optimizer.zero_grad(set_to_none=True)
                batch = loss = terms = before = None
                break
            del batch, loss, terms, before
        del model, optimizer, scaler, parameters, ema
        gc.collect(); torch.cuda.empty_cache()
    return records


def run(models, tasks, feature_root, device, output, esm="650m", batches=(2, 4, 8, 16),
        precisions=("fp32", "fp16"), steps=32, samples=128, data_root="data/therapeutic_2025", peptide_feature_root=None,
        profile_only=False):
    if any(name in (*HH_MODELS, HH_V16) for name in models) and (not profile_only or list(tasks) != ["fine39"] or esm != "650m"):
        raise ValueError("Use --profile-only --tasks fine39 with the default ESM650M for HH GPU profiling")
    if len(set(tasks)) == 1 and tasks[0] in (*THERAPEP_TRACKS, "rich48"):
        if str(data_root) == "data/therapeutic_2025":
            data_root = "data/therapeutic_2025/rich48" if tasks[0] == "rich48" else THERAPEP_ROOTS[tasks[0]]
        if str(feature_root) == f"features/esm2_{esm}":
            feature_root = f"features/rich48/esm2_{esm}" if tasks[0] == "rich48" else f"features/therapep_ai/esm2_{esm}"
    environment = inspect(feature_root, device)
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)
    active = environment["gpu"]["other_compute_processes"]
    # WDDM also lists graphics apps; Linux compute jobs and Windows Python jobs are busy.
    busy = [p for p in active if "python" in p[2].lower() or "train" in p[2].lower() or (os.name!="nt" and p[3].strip().isdigit())]
    if busy:
        raise RuntimeError(f"GPU has other compute jobs: {busy}; select a free GPU")
    if not environment["cuda_available"]:
        raise RuntimeError("Current Python has no usable CUDA; activate the server MFP-Nova environment")
    required_splits = ("train",) if profile_only else ("train", "val")
    if any(environment["caches"][split] is None for split in required_splits):
        raise FileNotFoundError(f"Real ESM cache required for {', '.join(required_splits)}")
    torch.cuda.set_device(torch.device(device))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/"environment.json", environment)
    summary = []
    for name in models:
        for task in tasks:
            config = yaml.safe_load(Path(f"configs/{name}.yaml").read_text(encoding="utf-8"))
            if name in (HH_V20, PEPTIDE_V6, PEPTIDE_V7) and task != "fine39":
                raise ValueError(f"Use --tasks fine39 for {name}")
            if name in (*HH_MODELS, HH_V16):
                base = yaml.safe_load(Path("configs/theraset.yaml").read_text(encoding="utf-8"))
                for key in ("encoder", "peptide_encoder"):
                    config[key] = base[key]
                config["model"] = {**base["model"], "label_context": False, "prototype_readout": False, **config["model"]}
                if name == "hh_setreread":
                    config["loss"] = {**base["loss"], **config["loss"]}
            from .therapep import configure_task
            configure_task(config, task)
            if task not in (*THERAPEP_TRACKS, "rich48") or str(data_root) != "data/therapeutic_2025":
                config["data"]["root"] = str(data_root)
            if task not in (*THERAPEP_TRACKS, "rich48") or str(feature_root) != f"features/esm2_{esm}":
                config["data"]["features"] = str(feature_root)
            if peptide_feature_root is not None and name in ("helixset", "theraset", "termiscale", "motifspan",
                                                            "privilegedbridge", *HH_MODELS, HH_V16, PEPTIDE_V6, PEPTIDE_V7):
                config["data"]["peptide_features"] = str(peptide_feature_root)
            if name == "setreread":
                config["train"]["correction_start_epoch"] = 1
                if str(feature_root) == f"features/esm2_{esm}":
                    config["data"]["features"] = "features/esm2_peptide_150m"
            elif name != HH_V20:
                config["encoder"]["name"] = "facebook/esm2_t33_650M_UR50D" if esm=="650m" else "facebook/esm2_t12_35M_UR50D"
            sequence_only = name in ("termiscale", "motifspan", "setreread", HH_V20, PEPTIDE_V6, PEPTIDE_V7)
            source = MultimodalData(config["data"]["root"], task, sequence_only=sequence_only,
                                   sequence_chemistry=name in ("setreread", HH_V20),
                                   localize_terminals=name == HH_V20)
            if name == HH_V20:
                source.enable_function_auxiliary()
            task_features = config["data"]["features"]
            data = MultimodalPeptides(source, task_features, "train", None if profile_only else samples,
                                     config["data"].get("peptide_features"),
                                     sequence_only=sequence_only or (name == "privilegedbridge" and config["model"].get("stage", "A") == "A"))
            validate_cache(data, config)
            groups = source.groups(data.indices, source.data.y[data.indices])
            integration = {"rows": len(data), "esm_dim": data.meta["input_dim"], "encoder": data.meta,
                           "statistics_train_rows": source.statistics["train_rows"],
                           "label_order": source.data.labels, "groups": {k: int(v.sum()) for k, v in groups.items()},
                           "empirical_factors": len(source.statistics["labels"]["hyperedges"]),
                           "prototype_counts": source.statistics["labels"]["prototype_counts"]}
            if config["loss"].get("rdrop", 0):
                integration["rdrop_profile_weight"] = config["loss"]["rdrop"]
                integration["rdrop_profile_scope"] = "Both dropout graphs retained; configured maximum weight, no warmup"
                print(f"RDrop profile | weight={config['loss']['rdrop']:.4f} views=2 (maximum, no warmup)", flush=True)
            print(f"Checking {name}/{task}: {len(data)} real rows", flush=True)
            if profile_only:
                records = []
                for scope in ("random_train", "longest_train"):
                    print(f"Profiling {name}/{task} | {scope} | {steps} measured steps per batch size", flush=True)
                    records.extend(profile(config, source, data, device, batches, precisions,
                                           measured_steps=steps, batch_scope=scope))
                write_json(root/f"{name}_{task}_profile.json", {"integration": integration, "profile": records})
                summary.append({"model": name, "task": task, "profile": records})
                continue
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
            source = MultimodalData(config["data"]["root"], task, checkpoint["statistics"],
                                   sequence_only=sequence_only,
                                   sequence_chemistry=name in ("setreread", HH_V20),
                                   localize_terminals=name == HH_V20)
            val = MultimodalPeptides(source, task_features, "val", min(samples, 64),
                                     config["data"].get("peptide_features"), sequence_only=sequence_only)
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
