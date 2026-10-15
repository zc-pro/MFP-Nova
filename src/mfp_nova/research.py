"""Shared research training, checkpoint resume, evaluation and prediction."""
import json
import math
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import evaluate_metrics, fit_global_threshold, fit_label_thresholds
from .multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from .trainer import seed_everything, write_json


def make_model(name, arguments):
    if name == "trirelay":
        from .trirelay import TriRelay
        return TriRelay(**arguments)
    if name == "geofactor":
        from .geofactor import GeoFactor
        return GeoFactor(**arguments)
    if name == "protoroute":
        from .protoroute import ProtoRoute
        return ProtoRoute(**arguments)
    raise ValueError(f"Unknown research model: {name}")


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def autocast_context(device, precision="fp32"):
    device = torch.device(device)
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("fp16/bf16 training requires a CUDA device")
    if precision == "bf16" and torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("This GPU does not support BF16; use fp16 or fp32")
    return torch.autocast("cuda", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[precision])


@torch.inference_mode()
def predict_loader(model, loader, device, precision="fp32"):
    model.eval()
    targets, probabilities = [], []
    for batch in loader:
        with autocast_context(device, precision):
            output = model(to_device(batch, device))
        targets.append(batch["y"].numpy())
        probabilities.append(output["logits"].float().sigmoid().cpu().numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def metrics(y, probabilities, statistics, threshold=.5):
    return evaluate_metrics(y, probabilities, threshold, statistics["labels"]["support"], statistics["function_projection"])


def validate_cache(dataset, config, reference=None):
    meta = dataset.meta
    if meta["name"] != config["encoder"]["name"] or meta["max_length"] != config["encoder"]["max_length"]:
        raise ValueError("Configured ESM differs from cache")
    if config["encoder"]["revision"] not in (meta["requested_revision"], meta["resolved_revision"]):
        raise ValueError("Configured ESM revision differs from cache")
    if reference is not None and any(meta[key] != reference[key] for key in (
            "resolved_revision", "input_dim", "dataset", "extraction_dtype", "storage_dtype")):
        raise ValueError("Split feature definitions differ")


def train(config, name, resume=None):
    settings = config["train"]
    device = torch.device(settings["device"])
    if device.type=="cuda":
        torch.cuda.set_device(device)
    seed_everything(settings["seed"])
    checkpoint = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if checkpoint is not None:
        # Resume preserves the trained model/priors/protocol; only execution controls change.
        original = checkpoint["config"]
        if config["model_name"] != original["model_name"] or config["data"]["track"] != original["data"]["track"]:
            raise ValueError("Resume model/task differs from checkpoint")
        for key in ("model", "loss", "encoder"):
            config[key] = original[key]
        for key, value in original["train"].items():
            if key not in ("device", "output", "epochs", "num_workers", "max_steps"):
                settings[key] = value
    source = MultimodalData(config["data"]["root"], config["data"]["track"],
                           checkpoint["statistics"] if checkpoint else None,
                           config.get("min_factor_support", 20), config.get("max_factors", 32))
    limit = settings.get("limit_samples")
    diagnostic = bool(limit or settings.get("max_steps") or (checkpoint and checkpoint["config"]["train"].get("max_steps")))
    # Diagnostic optimization subsets retain full original-Train priors/statistics:
    # shrinking the subset must not erase empirical factors or prototype capacity.
    datasets = [MultimodalPeptides(source, config["data"]["features"], split, limit) for split in ("train", "val")]
    validate_cache(datasets[0], config)
    validate_cache(datasets[1], config, datasets[0].meta)
    if datasets[0].meta["dataset"] != source.statistics["dataset"]:
        raise ValueError("Cache dataset version differs from multimodal data")
    generator = torch.Generator().manual_seed(settings["seed"])
    loaders = [DataLoader(data, batch_size=settings["batch_size"], shuffle=shuffle, num_workers=settings["num_workers"],
                          collate_fn=collate_multimodal, generator=generator if shuffle else None)
               for data, shuffle in zip(datasets, (True, False))]
    args = {"input_dim": datasets[0].meta["input_dim"], "num_labels": len(source.data.labels),
            "vocab_size": len(source.statistics["chemical_vocab"])+2,
            "statistics": source.statistics["labels"], **config["model"]}
    model = make_model(config["model_name"], args).to(device)
    run = Path(settings["output"])/name
    run.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    precision = settings.get("precision", "fp32")
    accumulation = settings.get("accumulation_steps", 1)
    scaler = torch.amp.GradScaler("cuda", enabled=precision=="fp16", init_scale=1024.)
    planned_epochs = checkpoint.get("planned_epochs", settings["epochs"]) if checkpoint else settings["epochs"]
    warmup = settings.get("warmup_epochs", 5)
    def schedule(epoch):
        if warmup and epoch<warmup:
            return (epoch+1)/warmup
        return .05+.95*.5*(1+math.cos(math.pi*min(1, max(0, (epoch-warmup)/max(1, planned_epochs-warmup)))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start, best_key, best_epoch, stale = 1, (-float("inf"), -float("inf")), 0, 0
    best_state = None
    global_step = 0
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        global_step = checkpoint.get("global_step", 0)
        start = checkpoint["epoch"]+1
        best_key, best_epoch, stale = tuple(checkpoint["best_key"]), checkpoint["best_epoch"], checkpoint["stale"]
        best_state = checkpoint["best_model_state"]
        random.setstate(checkpoint["python_rng"])
        np_state = checkpoint["numpy_rng"]
        np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["loader_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
        if start>settings["epochs"]:
            raise ValueError("--epochs must exceed the resumed epoch")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata = {"model": config["model_name"], "task": source.data.track, "seed": settings["seed"], "config": config,
                "git_commit": git.stdout.strip(), "git_dirty": bool(status.stdout.strip()), "trainable_parameters": params,
                "frozen_plm_parameters_included": False, "torch": str(torch.__version__), "numpy": str(np.__version__),
                "python": sys.version, "packages": {p: version(p) for p in ("transformers", "scikit-learn", "PyYAML")},
                "gpu": torch.cuda.get_device_name(device) if device.type=="cuda" else None,
                "device": str(device), "cuda": torch.version.cuda, "labels": source.data.labels,
                "encoder": datasets[0].meta, "train_rows": len(datasets[0]), "val_rows": len(datasets[1]),
                "dataset": source.statistics["dataset"], "cohort": source.statistics["cohort"],
                "label_semantics": "Observed annotations; 0 is unannotated, not experimentally confirmed negative",
                "statistics_train_rows": source.statistics["train_rows"],
                "resume": str(resume) if resume else None, "diagnostic_subset": diagnostic}
    write_json(run/"metadata.json", metadata)
    static = {"model_name": config["model_name"], "model_args": args, "config": config, "statistics": source.statistics,
              "labels": source.data.labels, "track": source.data.track, "encoder": datasets[0].meta, "planned_epochs": planned_epochs}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    elapsed_before = checkpoint.get("elapsed_seconds", 0) if checkpoint else 0
    for epoch in range(start, settings["epochs"]+1):
        model.train()
        totals, count, skipped = {}, 0, 0
        epoch_start = time.perf_counter()
        loss_settings = dict(config["loss"])
        ramp = min(1, epoch/max(1, settings.get("aux_warmup_epochs", 5)))
        for key in ("direct", "branch", "reconstruction", "prototype", "diversity", "balance", "factor"):
            if key in loss_settings:
                loss_settings[key] *= ramp
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loaders[0]):
            batch = to_device(batch, device)
            with autocast_context(device, precision):
                loss, components = model.compute_loss(batch, loss_settings)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite research loss")
            group_size = min(accumulation, len(loaders[0])-(batch_index//accumulation)*accumulation)
            scaler.scale(loss/group_size).backward()
            boundary = (batch_index+1)%accumulation==0 or batch_index+1==len(loaders[0])
            if boundary:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("gradient_clip", 5), error_if_nonfinite=precision!="fp16")
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale()<old_scale)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            n = len(batch["y"])
            for key, value in {"total": loss, **components}.items():
                totals[key] = totals.get(key, 0)+float(value.detach())*n
            count += n
            if settings.get("max_steps") and global_step>=settings["max_steps"]:
                break
        y, probabilities = predict_loader(model, loaders[1], device, precision)
        report = metrics(y, probabilities, source.statistics)
        if report["macro_ap"] is None:
            raise ValueError("No positive Validation labels for checkpoint selection")
        key = (report["macro_ap"], report["micro_ap"])
        if key>best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({**static, "model_state": best_state, "best_epoch": best_epoch}, run/"best.pt")
        else:
            stale += 1
        history = {"epoch": epoch, "losses": {k: v/count for k, v in totals.items()}, "lr": optimizer.param_groups[0]["lr"],
                   "global_step": global_step, "skipped_amp_updates": skipped,
                   "validation": report, "seconds": time.perf_counter()-epoch_start}
        with (run/"history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(history, allow_nan=False)+"\n")
        scheduler.step()
        numpy_rng = np.random.get_state()
        torch.save({**static, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "best_model_state": best_state,
                    "scaler_state": scaler.state_dict(), "global_step": global_step,
                    "best_key": best_key, "best_epoch": best_epoch, "epoch": epoch, "stale": stale,
                    "python_rng": random.getstate(), "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                    "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type=="cuda" else None,
                    "elapsed_seconds": elapsed_before+time.perf_counter()-started}, run/"last.pt")
        print(f"{config['model_name']} epoch={epoch} loss={totals['total']/count:.6f} val_macro_ap={key[0]:.6f}", flush=True)
        if settings["patience"] and stale>=settings["patience"]:
            break
        if settings.get("max_steps") and global_step>=settings["max_steps"]:
            break
    model.load_state_dict(best_state)
    y, probabilities = predict_loader(model, loaders[1], device, precision)
    threshold = fit_label_thresholds(y, probabilities) if settings.get("threshold_strategy", "global")=="per_label" else fit_global_threshold(y, probabilities)
    torch.save({**static, "model_state": best_state, "best_epoch": best_epoch, "threshold": threshold}, run/"best.pt")
    result = {"model": config["model_name"], "task": source.data.track, "seed": settings["seed"],
              "best_epoch": best_epoch, "epochs_completed": epoch, "threshold": threshold, "threshold_fit_split": "val",
              "global_step": global_step, "precision": precision, "accumulation_steps": accumulation,
              "threshold_strategy": settings.get("threshold_strategy", "global"), "trainable_parameters": params,
              "elapsed_seconds": elapsed_before+time.perf_counter()-started,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
              "diagnostic_subset": diagnostic, "validation": metrics(y, probabilities, source.statistics, threshold),
              "annotation_based_metrics": True,
              "checkpoint": str(run/"best.pt")}
    write_json(run/"results.json", result)
    return result


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["model_name"], checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), checkpoint


def evaluate(checkpoint_path, data_root, feature_root, split, device, output, predictions=None, batch_size=32):
    model, checkpoint = load_model(checkpoint_path, device)
    source = MultimodalData(data_root, checkpoint["track"], checkpoint["statistics"])
    dataset = MultimodalPeptides(source, feature_root, split)
    validate_cache(dataset, checkpoint["config"], checkpoint["encoder"])
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_multimodal)
    y, probabilities = predict_loader(model, loader, device)
    threshold = checkpoint["threshold"]
    grouped = {}
    for name, mask in source.groups(dataset.indices, y).items():
        grouped[name] = {"rows": int(mask.sum()), "metrics": metrics(y[mask], probabilities[mask], source.statistics, threshold) if mask.any() else None}
    result = {"model": checkpoint["model_name"], "task": checkpoint["track"], "split": split, "rows": len(dataset),
              "labels": source.data.labels, "threshold": threshold, "checkpoint": str(checkpoint_path),
              "metrics": metrics(y, probabilities, source.statistics, threshold), "groups": grouped,
              "annotation_based_metrics": True, "experimental_structure_group": None}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    if predictions:
        Path(predictions).parent.mkdir(parents=True, exist_ok=True)
        with Path(predictions).open("xb") as f:
            np.savez_compressed(f, row_id=source.data.rows[dataset.indices], ID=source.data.ids[dataset.indices],
                                targets=y, probabilities=probabilities, label_names=np.array(source.data.labels), threshold=threshold)
    return result


def predict_inputs(checkpoint_path, input_path, device="cpu", batch_size=32):
    """Inference on portable tensor inputs exported by the shared preprocessing API.

    The NPZ contains batched collate_multimodal fields (no labels required). Numeric
    inputs must use the checkpoint's saved preprocessing. The API also accepts raw
    multimodal samples through MultimodalData.sample -> collate_multimodal.
    """
    model, checkpoint = load_model(checkpoint_path, device)
    with np.load(input_path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files if k not in ("y", "row_id", "ID")}
    if "phys_raw" in raw:
        from .multimodal import prepare_inputs
        inputs = prepare_inputs(raw, checkpoint["statistics"])
    else:
        inputs = {k: torch.from_numpy(v) for k, v in raw.items()}
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(inputs["aa"]), batch_size):
            batch = {k: v[start:start+batch_size].to(device) for k, v in inputs.items()}
            probabilities.extend(model(batch)["logits"].sigmoid().cpu().tolist())
    return prediction_records(checkpoint, probabilities, "provided_multimodal_fields")


def prediction_records(checkpoint, probabilities, scope):
    return [{"probabilities": dict(zip(checkpoint["labels"], p)), "evidence_scope": scope,
             "predicted_labels": [label for label, value, threshold in zip(checkpoint["labels"], p,
                                     np.broadcast_to(checkpoint["threshold"], (len(p),))) if value>=threshold]}
            for p in probabilities]


def predict_sequences(checkpoint_path, sequences, device="cpu"):
    """Missing-modality inference; never fabricates structure or chemical descriptors."""
    from .encoder import FrozenEncoder
    from .multimodal import prepare_inputs
    model, checkpoint = load_model(checkpoint_path, device)
    info = checkpoint["encoder"]
    encoder = FrozenEncoder(info["name"], info["resolved_revision"], info["max_length"], device)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    output = []
    size = checkpoint["config"]["encoder"].get("batch_size", 8 if info["input_dim"]==1280 else 32)
    for start in range(0, len(sequences), size):
        selected = sequences[start:start+size]
        states, mask = encoder.encode(selected)
        residues = [states[i, mask[i]].half().float().cpu() for i in range(len(selected))]
        esm = torch.nn.utils.rnn.pad_sequence(residues, batch_first=True).numpy()
        length = esm.shape[1]
        aa = np.zeros((len(selected), length), np.int64)
        stereo = np.zeros_like(aa)
        valid = np.zeros_like(aa, dtype=bool)
        for i, s in enumerate(selected):
            aa[i, :len(s)] = [alphabet.index(c.upper())+1 for c in s]
            stereo[i, :len(s)] = [c.islower() for c in s]
            valid[i, :len(s)] = True
        raw = {"esm": esm, "aa": aa, "d": stereo, "mask": valid,
               "phys_raw": np.full((len(selected), 515), np.nan),
               "residue_chem_tokens": np.array([[c if j<len(s) else "" for j, c in enumerate(s.ljust(length))] for s in selected])}
        inputs = prepare_inputs(raw, checkpoint["statistics"])
        with torch.inference_mode():
            probs = model(to_device(inputs, device))["logits"].sigmoid().cpu().tolist()
        records = prediction_records(checkpoint, probs, "sequence_and_stereochemistry_only")
        for s, record in zip(selected, records):
            record["sequence"] = s
        output.extend(records)
    return output


def summarize_results(runs, split="val"):
    """Plain JSON comparison. Never selects a model or threshold from Test."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for directory in runs:
        directory = Path(directory)
        metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
        result = json.loads((directory/"results.json").read_text(encoding="utf-8"))
        if result.get("diagnostic_subset"):
            raise ValueError(f"Diagnostic subset is not a formal comparison: {directory}")
        score = result["validation"] if split=="val" else json.loads((directory/"test.json").read_text(encoding="utf-8"))["metrics"]
        config = metadata["config"]
        protocol = {"model": metadata["model"], "task": metadata["task"], "encoder": {key: metadata["encoder"][key] for key in ("name", "resolved_revision")},
                    "dataset": metadata["dataset"], "cohort": metadata["cohort"], "model_settings": config["model"],
                    "loss": config["loss"], "labels": metadata["labels"], "threshold_strategy": result["threshold_strategy"],
                    "training": {k: v for k, v in config["train"].items() if k not in ("seed", "device", "output", "num_workers")}}
        buckets[json.dumps(protocol, sort_keys=True)].append({"run": str(directory), "seed": metadata["seed"], "metrics": score,
            "trainable_parameters": result["trainable_parameters"], "elapsed_seconds": result["elapsed_seconds"],
            "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"]})
    def scalar_metrics(mapping, prefix=""):
        values = {}
        for key, value in mapping.items():
            name = prefix+key
            if isinstance(value, dict):
                values.update(scalar_metrics(value, name+"."))
            elif isinstance(value, (int, float)):
                values[name] = value
        return values
    def aggregate(values):
        present = [v for v in values if v is not None]
        return {"mean": float(np.mean(present)), "std": float(np.std(present, ddof=1)) if len(present)>1 else 0.,
                "available_seeds": len(present)} if present else None
    summary = []
    for protocol, records in buckets.items():
        if len({r["seed"] for r in records}) != len(records):
            raise ValueError("Repeated seed within one comparison protocol; distinguish the configurations")
        scalars = [scalar_metrics(r["metrics"]) for r in records]
        names = sorted(set().union(*(s.keys() for s in scalars)))
        score_summary = {name: aggregate([s.get(name) for s in scalars]) for name in names}
        per_label = {key: [aggregate([record["metrics"][key][j] for record in records]) for j in range(len(records[0]["metrics"][key]))]
                     for key in ("label_ap", "label_f1", "label_mcc", "label_auroc", "label_positive_support")}
        summary.append({"protocol": json.loads(protocol), "split": split, "seeds": [r["seed"] for r in records],
                        "metrics": score_summary, "per_label": per_label,
                        "resources": {key: aggregate([r[key] for r in records]) for key in ("trainable_parameters", "elapsed_seconds", "peak_cuda_memory_bytes")},
                        "runs": [r["run"] for r in records]})
    return {"split": split, "std_ddof": 1, "comparisons": summary}
