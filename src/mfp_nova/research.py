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
from .therapep import TRACKS as THERAPEP_TRACKS

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .metrics import evaluate_metrics, evaluate_groups, format_epoch, fit_global_threshold, fit_label_thresholds
from .multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from .research_layers import observed_label_loss
from .consistency import bernoulli_consistency, consistency_weight
from .trainer import seed_everything, write_json
from .metric_protocol import protocol_description, primary_metrics, selection, format_primary, final_threshold


def make_model(name, arguments):
    if name == "theraset":
        from .models.theraset import TheraSet
        return TheraSet(**arguments)
    if name == "helixset":
        from .models.helixset import HelixSet
        return HelixSet(**arguments)
    if name == "trirelay":
        from .models.trirelay import TriRelay
        return TriRelay(**arguments)
    if name == "geofactor":
        from .models.geofactor import GeoFactor
        return GeoFactor(**arguments)
    if name == "protoroute":
        from .models.protoroute import ProtoRoute
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
def predict_loader(model, loader, device, precision="fp32", loss_settings=None):
    model.eval()
    targets, probabilities = [], []
    loss_total, rows = 0., 0
    for batch in loader:
        moved = to_device(batch, device)
        with autocast_context(device, precision):
            output = model(moved)
            if loss_settings is not None:
                loss = observed_label_loss(output["logits"], moved["y"], model.statistics,
                                           loss_settings.get("unlabeled_weight", .3))
                loss_total += float(loss)*len(batch["y"])
                rows += len(batch["y"])
        targets.append(batch["y"].numpy())
        probabilities.append(output["logits"].float().sigmoid().cpu().numpy())
    result = np.concatenate(targets), np.concatenate(probabilities)
    return (*result, loss_total/rows) if loss_settings is not None else result


def metrics(y, probabilities, statistics, threshold=.5):
    return evaluate_metrics(y, probabilities, threshold, statistics["labels"]["support"], statistics["function_projection"])


def export_validation(run, stem, source, dataset, y, probabilities, threshold, epoch, weight_source, objective):
    """Reuse collected Val predictions; no extra model forward or Test access."""
    rows = dataset.indices
    path = run/f"{stem}_validation.npz"
    np.savez_compressed(path, row_id=source.data.rows[rows], ID=source.data.ids[rows],
                        cluster_id=source.data.clusters[rows], targets=y, probabilities=probabilities,
                        label_names=np.array(source.data.labels), threshold=threshold, split="val",
                        task=source.data.track, epoch=epoch, weight_source=weight_source,
                        selection_objective=objective)
    actual, predicted = y.astype(bool), probabilities >= threshold
    from .multimodal import independent_functions
    projection = source.statistics["function_projection"]
    summary = {
        "split": "val", "epoch": epoch, "weight_source": weight_source, "selection_objective": objective,
        "threshold": threshold, "rows": len(y), "empty_predictions": int((~predicted.any(1)).sum()),
        "mean_predicted_labels": float(predicted.sum(1).mean()),
        "mean_observed_labels": float(actual.sum(1).mean()),
        "per_label": [{"label": label, "tp": int((actual[:, j] & predicted[:, j]).sum()),
                       "observed_fp": int((~actual[:, j] & predicted[:, j]).sum()),
                       "fn": int((actual[:, j] & ~predicted[:, j]).sum())}
                      for j, label in enumerate(source.data.labels)],
        "scope": "Validation-selected checkpoint and threshold; FP means unannotated, not confirmed inactive",
    }
    if projection is not None:
        actual_functions = independent_functions(actual, projection)
        predicted_functions = independent_functions(predicted, projection)
        multifunction = actual_functions.sum(1) >= 2
        summary.update(
            multifunction_support=int(multifunction.sum()),
            multifunction_missed_detection=int((multifunction & (predicted_functions.sum(1) < 2)).sum()),
            multifunction_with_two_observed_functions_recovered=int(
                (multifunction & ((actual_functions & predicted_functions).sum(1) >= 2)).sum()))
    else:
        summary["scope"] = "Published 49-bit targets; decoy zeros are assigned negatives, not verified inactivity"
    write_json(run/f"{stem}_diagnostics.json", summary)
    return path.name


def validate_cache(dataset, config, reference=None):
    meta = dataset.meta
    if meta["name"] != config["encoder"]["name"] or meta["max_length"] != config["encoder"]["max_length"]:
        raise ValueError("Configured ESM differs from cache")
    if config["encoder"]["revision"] not in (meta["requested_revision"], meta["resolved_revision"]):
        raise ValueError("Configured ESM revision differs from cache")
    layer = config["encoder"].get("representation_layer")
    if meta.get("representation_layer") != layer or (layer is not None and not meta.get("special_tokens_cached")):
        raise ValueError("ESM prefix cache differs from config; extract the configured layer including CLS/EOS")
    tail = config.get("model", {}).get("esm_tail")
    if tail is not None and (tail["representation_layer"] != layer or tail["name"] != meta["name"]
                             or tail["revision"] != meta["resolved_revision"]):
        raise ValueError("Trainable ESM tail must continue the cached layer from the same pretrained revision")
    if reference is not None and any(meta[key] != reference[key] for key in (
            "resolved_revision", "input_dim", "dataset", "extraction_dtype", "storage_dtype")):
        raise ValueError("Split feature definitions differ")
    peptide = meta.get("peptide_encoder")
    settings = config.get("peptide_encoder")
    if (peptide is None) != (settings is None):
        raise ValueError("Peptide encoder configuration and cache must be supplied together")
    if peptide is not None:
        if config["model_name"] not in ("helixset", "theraset"):
            raise ValueError("Dual frozen encoders are supported by HelixSet and TheraSet")
        if peptide["name"] != settings["name"] or peptide["max_length"] != settings["max_length"]:
            raise ValueError("Configured peptide encoder differs from cache")
        if settings["revision"] not in (peptide["requested_revision"], peptide["resolved_revision"]):
            raise ValueError("Configured peptide encoder revision differs from cache")
        if peptide.get("representation_layer") is not None:
            raise ValueError("Peptide view requires final-layer features")
        if peptide["dataset"] != meta["dataset"]:
            raise ValueError("Primary and peptide caches use different datasets")
    if reference is not None:
        other = reference.get("peptide_encoder")
        if (peptide is None) != (other is None):
            raise ValueError("Checkpoint/splits differ in peptide encoder use")
        if peptide is not None and any(peptide[key] != other[key] for key in (
                "name", "resolved_revision", "input_dim", "max_length", "dataset", "extraction_dtype", "storage_dtype")):
            raise ValueError("Peptide feature definitions differ across splits/checkpoint")


def epoch_loss_settings(config, epoch):
    """Original warmup, with optional reconstruction-only cosine decay."""
    settings = config['train']
    loss = dict(config['loss'])
    if 'branch_supervision' in settings:
        loss['branch_supervision'] = settings['branch_supervision']
    ramp = min(1, epoch/max(1, settings.get('aux_warmup_epochs', 5)))
    for key in ('direct', 'branch', 'reconstruction', 'prototype', 'diversity', 'balance', 'factor',
                'evidence', 'annotation_set'):
        if key in loss:
            loss[key] *= ramp
    if "rdrop" in loss:
        start = settings.get("rdrop_start_epoch", 6)
        duration = settings.get("rdrop_ramp_epochs", 5)
        loss["rdrop"] *= min(1., max(0., (epoch-start+1)/duration))
    decay = settings.get('reconstruction_decay')
    if decay is not None:
        start, end = decay
        floor = settings.get('reconstruction_floor', .2)
        if end <= start or start < settings.get('aux_warmup_epochs', 5) or not 0 <= floor <= 1:
            raise ValueError('Reconstruction decay requires warmup <= start < end and floor in [0, 1]')
        progress = min(1, max(0, (epoch-start)/(end-start)))
        loss['reconstruction'] *= floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
    if settings.get("consistency_weight", 0) > 0:
        loss["consistency"] = consistency_weight(settings, epoch)
    if settings.get("function_aux_weight", 0) > 0:
        loss["function_aux"] = settings["function_aux_weight"]*ramp
        loss["function_hierarchy"] = settings.get("function_hierarchy_weight", .02)*ramp
    if settings.get("relation_view_weight", 0) > 0:
        loss["relation_view"] = settings["relation_view_weight"]*min(1., max(0., (epoch-settings.get("warmup_epochs", 5))/5))
        loss["relation_dropout"] = settings.get("relation_dropout", .2)
    return loss


def update_ema(ema, model, epoch, start_epoch, skipped):
    """Average successful optimizer updates, never microbatches or AMP skips."""
    if ema is not None and epoch >= start_epoch and not skipped:
        ema.update_parameters(model)


def frozen_tail_keys(model):
    """Pretrained tail tensors are reloaded by revision, not saved per checkpoint."""
    return {key for key, _ in model.named_parameters()
            if key.removeprefix("module.").startswith("esm_tail.") and ".lora_" not in key}


def checkpoint_state(model, cpu=False):
    omitted = frozen_tail_keys(model)
    return {key: value.detach().cpu().clone() if cpu else value
            for key, value in model.state_dict().items() if key not in omitted}


def restore_model_state(model, state):
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != frozen_tail_keys(model) - state.keys() or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint/model mismatch: {result}")


def make_optimizer(model, settings):
    parameters = [(key, value) for key, value in model.named_parameters() if value.requires_grad]
    if getattr(model, "esm_tail", None) is not None:
        groups = [{"params": [p for key, p in parameters if not key.startswith("esm_tail.")]},
                  {"params": [p for key, p in parameters if key.startswith("esm_tail.")],
                   "lr": settings.get("esm_learning_rate", 1e-5)}]
    else:
        groups = [p for _, p in parameters]
    return torch.optim.AdamW(groups, lr=settings["learning_rate"], weight_decay=settings["weight_decay"])


def save_last_checkpoint(checkpoint, path):
    """Keep the previous complete epoch if saving is interrupted."""
    path = Path(path)
    temporary = path.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def restore_history(run, checkpoint):
    """Discard uncommitted lines and recover the epoch saved just before interruption."""
    path = run / 'history.jsonl'
    records = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break  # A process can stop halfway through the final append.
            if record['epoch'] <= checkpoint['epoch']:
                records.append(record)
    if 'history_record' in checkpoint:
        records = [r for r in records if r['epoch'] < checkpoint['epoch']]
        records.append(checkpoint['history_record'])
    path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records), encoding='utf-8')


def train(config, name, resume=None, resume_in_place=False, stop_file=None):
    settings = config["train"]
    device = torch.device(settings["device"])
    if device.type=="cuda":
        torch.cuda.set_device(device)
    seed_everything(settings["seed"])
    checkpoint = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if checkpoint is not None:
        # Resume preserves the trained model/priors/protocol; only execution controls change.
        original = checkpoint["config"]
        if ("lr_decay_epochs" in settings and settings["lr_decay_epochs"] !=
                original["train"].get("lr_decay_epochs", checkpoint["planned_epochs"])):
            raise ValueError("Changing the LR decay horizon requires a new run, not resume")
        if settings.get("metric_suite", "task_specific") != original["train"].get("metric_suite", "task_specific"):
            raise ValueError("Metric suite changes checkpoint selection; start a new run or resume with its saved suite")
        if "branch_supervision" in settings and settings["branch_supervision"] != original["train"].get("branch_supervision", "mixed_private"):
            raise ValueError("Branch supervision changes the experiment; resume its own last.pt or start a new run")
        if "shared_private" in config["model"] and config["model"]["shared_private"] != original["model"].get("shared_private", False):
            raise ValueError("Shared/private architecture differs from checkpoint; start a new run")
        if "evidence_fusion" in config["model"] and config["model"]["evidence_fusion"] != original["model"].get("evidence_fusion", "residual"):
            raise ValueError("Evidence fusion differs from checkpoint; resume its own last.pt or start a new run")
        if settings.get("ema_decay") is not None and original["train"].get("ema_decay") is None:
            raise ValueError("EMA changes the experiment; start a new training run instead of resuming a non-EMA checkpoint")
        for key in ("consistency_weight", "consistency_start_epoch", "consistency_ramp_epochs"):
            if key in settings and settings[key] != original["train"].get(key, 0 if key == "consistency_weight" else None):
                raise ValueError("Consistency changes the experiment; resume its own last.pt or start a new run")
        for key in ("function_aux_weight", "function_hierarchy_weight"):
            if key in settings and settings[key] != original["train"].get(key, 0 if key == "function_aux_weight" else .02):
                raise ValueError("Function auxiliary supervision changes the experiment; start a new run")
        for key, default in (("reconstruction_gradient", "sum"), ("reconstruction_grad_ratio", 1.)):
            if key in settings and settings[key] != original["train"].get(key, default):
                raise ValueError("Reconstruction gradient optimization changes the experiment; start a new run")
        for key, default in (("relation_view_weight", 0.), ("relation_dropout", .2)):
            if key in settings and settings[key] != original["train"].get(key, default):
                raise ValueError("Relation-view supervision changes the experiment; start a new run")
        if config["model_name"] != original["model_name"] or config["data"]["track"] != original["data"]["track"]:
            raise ValueError("Resume model/task differs from checkpoint")
        for key in ("model", "loss", "encoder"):
            config[key] = original[key]
        if "peptide_encoder" in original:
            config["peptide_encoder"] = original["peptide_encoder"]
            config["data"].setdefault("peptide_features", original["data"]["peptide_features"])
        else:
            config.pop("peptide_encoder", None)
            config["data"].pop("peptide_features", None)
        for key, value in original["train"].items():
            if key not in ("device", "output", "epochs", "num_workers", "max_steps"):
                settings[key] = value
    suite = settings.get("metric_suite", "task_specific")
    rdrop = config["loss"].get("rdrop", 0)
    if not math.isfinite(rdrop) or rdrop < 0:
        raise ValueError("R-Drop weight must be finite and nonnegative")
    if rdrop:
        if config["model_name"] not in ("helixset", "theraset"):
            raise ValueError("R-Drop currently supports HelixSet and TheraSet")
        if settings.get("rdrop_start_epoch", 6) < 1 or settings.get("rdrop_ramp_epochs", 5) < 1:
            raise ValueError("R-Drop start epoch and ramp length must be positive")
    protocol = protocol_description(suite, settings.get("threshold_strategy", "global"))
    branch_supervision = settings.get("branch_supervision", "mixed_private")
    if branch_supervision not in ("mixed_private", "shared_private"):
        raise ValueError("branch_supervision must be mixed_private or shared_private")
    if branch_supervision == "shared_private" and (config["model_name"] != "trirelay" or not config["model"].get("shared_private")):
        raise ValueError("Shared/private branch supervision requires shared-private TriRelay")
    evidence_fusion = config["model"].get("evidence_fusion", "residual")
    if evidence_fusion not in ("residual", "normalized_competitive", "normalized_residual"):
        raise ValueError("Evidence fusion must be residual, normalized_competitive or normalized_residual")
    if evidence_fusion != "residual" and (config["model_name"] != "trirelay" or not config["model"].get("shared_private")):
        raise ValueError("Non-residual evidence fusion requires shared-private TriRelay")
    consistency = settings.get("consistency_weight", 0)
    if not math.isfinite(consistency) or consistency < 0:
        raise ValueError("consistency_weight must be finite and nonnegative")
    if consistency:
        if config["model_name"] != "trirelay" or settings.get("ema_decay") is None:
            raise ValueError("Consistency currently requires TriRelay with EMA")
        ema_start = settings.get("ema_start_epoch", settings.get("warmup_epochs", 5)+1)
        settings.setdefault("consistency_start_epoch", ema_start+1)
        settings.setdefault("consistency_ramp_epochs", 5)
        if settings["consistency_start_epoch"] <= ema_start or settings["consistency_ramp_epochs"] < 1:
            raise ValueError("Consistency starts after EMA initialization and requires a positive ramp length")
    elif any(key in settings for key in ("consistency_start_epoch", "consistency_ramp_epochs")):
        raise ValueError("Consistency scheduling requires a positive consistency_weight")
    function_aux = settings.get("function_aux_weight", 0)
    relation_view_weight = settings.get("relation_view_weight", 0.)
    relation_dropout = settings.get("relation_dropout", .2)
    if not math.isfinite(relation_view_weight) or relation_view_weight < 0 or not 0 < relation_dropout < 1:
        raise ValueError("Relation-view weight must be finite/nonnegative and dropout between 0 and 1")
    if relation_view_weight and (config["model_name"] != "trirelay" or not config["model"].get("relations", True)):
        raise ValueError("Relation views require TriRelay with relations enabled")
    hierarchy = settings.get("function_hierarchy_weight", .02)
    if not math.isfinite(function_aux) or function_aux < 0 or not math.isfinite(hierarchy) or hierarchy < 0:
        raise ValueError("Function auxiliary weights must be finite and nonnegative")
    if function_aux and config["model_name"] != "trirelay":
        raise ValueError("Function auxiliary supervision currently supports TriRelay")
    if function_aux:
        config["model"]["function_aux"] = True
    reconstruction_gradient = settings.get("reconstruction_gradient", "sum")
    gradient_ratio = settings.get("reconstruction_grad_ratio", 1.)
    if reconstruction_gradient not in ("sum", "cap", "project") or not math.isfinite(gradient_ratio) or gradient_ratio < 0:
        raise ValueError("Use reconstruction_gradient=sum/cap/project and a finite nonnegative gradient ratio")
    if reconstruction_gradient != "sum" and config["model_name"] != "trirelay":
        raise ValueError("Reconstruction gradient balancing currently supports TriRelay")
    source = MultimodalData(config["data"]["root"], config["data"]["track"],
                           checkpoint["statistics"] if checkpoint else None,
                           config.get("min_factor_support", 20), config.get("max_factors", 32))
    if function_aux:
        source.enable_function_auxiliary()
    limit = settings.get("limit_samples")
    diagnostic = bool(limit or settings.get("max_steps") or (checkpoint and checkpoint["config"]["train"].get("max_steps")))
    # Diagnostic optimization subsets retain full original-Train priors/statistics:
    # shrinking the subset must not erase empirical factors or prototype capacity.
    datasets = [MultimodalPeptides(source, config["data"]["features"], split, limit,
                                   config["data"].get("peptide_features")) for split in ("train", "val")]
    validate_cache(datasets[0], config, checkpoint["encoder"] if checkpoint else None)
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
    if "peptide_encoder" in datasets[0].meta:
        args["peptide_input_dim"] = datasets[0].meta["peptide_encoder"]["input_dim"]
    if function_aux:
        args["function_aux_statistics"] = source.statistics["function_auxiliary"]
    model = make_model(config["model_name"], args).to(device)
    run = Path(settings["output"])/name
    if resume_in_place:
        if (run/'results.json').exists():
            raise ValueError('This run is already complete; use a new name for another experiment')
        if resume is not None:
            if Path(resume).resolve() != (run/'last.pt').resolve():
                raise ValueError('In-place resume requires this run directory\'s last.pt')
            restore_history(run, checkpoint)
        elif (run/'last.pt').exists() or (run/'history.jsonl').exists():
            raise ValueError('Continue this run using --resume last.pt')
        else:
            # An interruption before epoch one committed has no state to resume.
            run.mkdir(parents=True, exist_ok=True)
            print('Restarting the uncommitted first epoch.', flush=True)
    else:
        run.mkdir(parents=True, exist_ok=False)
    optimizer = make_optimizer(model, settings)
    precision = settings.get("precision", "fp32")
    accumulation = settings.get("accumulation_steps", 1)
    scaler = torch.amp.GradScaler("cuda", enabled=precision=="fp16", init_scale=1024.)
    planned_epochs = checkpoint.get("planned_epochs", settings["epochs"]) if checkpoint else settings["epochs"]
    warmup = settings.get("warmup_epochs", 5)
    decay_epochs = settings.get("lr_decay_epochs", planned_epochs)
    if "lr_decay_epochs" in settings and decay_epochs <= warmup:
        raise ValueError("lr_decay_epochs must exceed warmup_epochs")
    save_candidates = settings.get("save_selection_candidates", False) and suite != "micro_f1"
    alternate_suite = "task_specific" if suite == "traditional5" else "traditional5"
    alternate_stem = "best_ap" if suite == "traditional5" else "best_accuracy"
    alternate_best = checkpoint.get("alternate_best") if checkpoint and save_candidates else None
    candidate_start = (checkpoint.get("selection_tracking_start_epoch", checkpoint["epoch"]+1)
                       if checkpoint else 1)
    ema_decay = settings.get("ema_decay")
    ema_start = settings.get("ema_start_epoch", warmup+1)
    if ema_decay is not None and (not 0 < ema_decay < 1 or not 1 <= ema_start <= planned_epochs):
        raise ValueError("EMA requires 0 < decay < 1 and a start epoch within the training schedule")
    if ema_decay is None and settings.get("ema_start_epoch") is not None:
        raise ValueError("ema_start_epoch requires ema_decay")
    # These models use LayerNorm; buffers are Train priors/indices, not BN moments.
    # Keep buffers copied from the online model, rather than averaging their values.
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay)) if ema_decay is not None else None
    if ema is not None:
        ema.requires_grad_(False)
    online_best_state, online_best_key, online_best_epoch = None, (-float("inf"), -float("inf")), 0
    best_weights = "online"
    def schedule(epoch):
        if warmup and epoch<warmup:
            return (epoch+1)/warmup
        return .05+.95*.5*(1+math.cos(math.pi*min(1, max(0, (epoch-warmup)/max(1, decay_epochs-warmup)))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start, best_key, best_epoch, stale = 1, (-float("inf"), -float("inf")), 0, 0
    best_state = None
    best_ranking = (None, None)
    best_selection_threshold = None
    global_step = 0
    if checkpoint:
        restore_model_state(model, checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        global_step = checkpoint.get("global_step", 0)
        start = checkpoint["epoch"]+1
        best_key, best_epoch, stale = tuple(checkpoint["best_key"]), checkpoint["best_epoch"], checkpoint["stale"]
        best_state = checkpoint["best_model_state"]
        best_ranking = tuple(checkpoint.get("best_ranking", best_key))
        best_selection_threshold = checkpoint.get("best_selection_threshold")
        best_weights = checkpoint.get("best_weights", "online")
        if ema is not None:
            restore_model_state(ema, checkpoint["ema_state"])
            online_best_state = checkpoint["online_best_state"]
            online_best_key = tuple(checkpoint["online_best_key"])
            online_best_epoch = checkpoint["online_best_epoch"]
        random.setstate(checkpoint["python_rng"])
        np_state = checkpoint["numpy_rng"]
        np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["loader_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], device)
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
                "resume": str(resume) if resume else None, "diagnostic_subset": diagnostic,
                "monitor_threshold": .5, "train_metrics_scope": "online_augmented_changing_weights",
                "validation_loss_scope": "main_classification_clean_eval; no stochastic auxiliary reconstruction"}
    if source.data.track in THERAPEP_TRACKS:
        metadata["label_semantics"] = ("Positive-only 48 functional targets; 0 is unannotated; no therapeutic flag"
            if source.data.track == "therapep48" else
            "Author-published 49-bit targets; synthetic backgrounds when present are assigned zeros; therapeutic flag is supervised")
        metadata["split_protocol"] = "Published CSV members, including duplicates; no cluster isolation imposed"
        metadata["modality_scope"] = "Sequence only for every source; no positive-only structure/descriptor availability shortcut"
    if getattr(model, "esm_tail", None) is not None:
        metadata["esm_adaptation"] = {
            "mode": "frozen_prefix_cached_trainable_tail_lora",
            "prefix_layer": config["encoder"]["representation_layer"],
            "trainable_parameters": sum(p.numel() for p in model.esm_tail.parameters() if p.requires_grad),
            "pretrained_weights_in_checkpoint": False,
            "learning_rate": settings.get("esm_learning_rate", 1e-5)}
    if function_aux:
        aux_stats = source.statistics["function_auxiliary"]
        metadata["function_auxiliary"] = {"task": aux_stats["track"], "labels": aux_stats["labels"],
                                          "statistics_train_rows": aux_stats["train_rows"]}
    if reconstruction_gradient != "sum":
        metadata["gradient_optimization"] = f"reconstruction gradient mode={reconstruction_gradient}; logged total is the original weighted loss sum"
    if branch_supervision == "shared_private":
        metadata["branch_supervision_scope"] = "First read before mixed-label feedback: shared/private equally share the original branch weight; no extra parameters"
    if evidence_fusion == "normalized_competitive":
        metadata["evidence_fusion_scope"] = "Both reads: parameter-free RMS(shared/private), same normalized evidence for branch supervision, convex (1-g)*shared+g*private; no extra parameters"
    elif evidence_fusion == "normalized_residual":
        metadata["evidence_fusion_scope"] = "Both reads: same parameter-free RMS, gate inputs and branch supervision as normalized_competitive; only mixing changes to shared+g*private; no extra parameters"
    metadata["metric_protocol"] = protocol
    if rdrop:
        metadata["rdrop_scope"] = {
            "coefficient": rdrop, "start_epoch": settings.get("rdrop_start_epoch", 6),
            "ramp_epochs": settings.get("rdrop_ramp_epochs", 5),
            "objective": "mean supervised loss of two dropout forwards + symmetric Bernoulli KL over all labels",
            "inputs": "same modality-dropout view; independent ordinary network dropout masks",
            "train_metrics": "first stochastic forward only", "validation": "single clean forward; main ASL only"}
    metadata["phys_missing_values"] = "zero_after_scaling" if source.statistics["phys"].get("zero_missing", False) else "legacy_median_after_scaling"
    if save_candidates:
        metadata["selection_candidates"] = {"primary": suite, "alternate": alternate_suite,
                                            "tracking_start_epoch": candidate_start,
                                            "stopping": "primary suite only; one shared optimizer trajectory"}
    if not resume_in_place or checkpoint is None:
        write_json(run/"metadata.json", metadata)
    static = {"model_name": config["model_name"], "model_args": args, "config": config, "statistics": source.statistics,
              "labels": source.data.labels, "track": source.data.track, "encoder": datasets[0].meta, "planned_epochs": planned_epochs}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    elapsed_before = checkpoint.get("elapsed_seconds", 0) if checkpoint else 0
    peak_memory = (checkpoint.get("peak_cuda_memory_bytes", 0) or 0) if checkpoint else 0
    validation_groups = source.groups(datasets[1].indices, source.data.y[datasets[1].indices])
    epoch = checkpoint["epoch"] if checkpoint else 0
    # A crash after the final last.pt but before results.json needs finalization,
    # not an extra optimizer epoch or a restarted early-stopping counter.
    terminal_resume = checkpoint is not None and (start>settings["epochs"] or
                        (settings["patience"] and stale>=settings["patience"]))
    for epoch in (() if terminal_resume else range(start, settings["epochs"]+1)):
        model.train()
        totals, count, skipped = {}, 0, 0
        gradient_records = []
        epoch_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        online_targets, online_probabilities = [], []
        loss_settings = epoch_loss_settings(config, epoch)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loaders[0]):
            batch = to_device(batch, device)
            teacher_logits = None
            if loss_settings.get("consistency", 0) > 0 and int(ema.n_averaged) > 0:
                ema.module.eval()
                with torch.no_grad(), autocast_context(device, precision):
                    teacher_logits = ema.module(batch)["logits"]
            with autocast_context(device, precision):
                loss, components, output = model.compute_loss(batch, loss_settings, return_output=True)
            if consistency:
                if teacher_logits is None:
                    agreement = output["logits"].sum()*0
                    agreement_terms = {"consistency_high": agreement, "consistency_low": agreement}
                else:
                    agreement, agreement_terms = bernoulli_consistency(output["logits"], teacher_logits, batch["y"])
                loss = loss+loss_settings["consistency"]*agreement
                components = {**components, "consistency": agreement, **agreement_terms}
            del teacher_logits
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite research loss")
            group_size = min(accumulation, len(loaders[0])-(batch_index//accumulation)*accumulation)
            if reconstruction_gradient != "sum":
                from .gradient_balance import backward_reconstruction
                auxiliary = loss_settings.get("reconstruction", .05)*components["reconstruction"]
                # Construct the primary graph directly: subtracting reconstruction
                # from total leaves zero-but-present gradients on its private head.
                main = (components["classification"]+loss_settings.get("direct", .2)*components["direct"]
                        +loss_settings.get("branch", .1)*components["branch"])
                for key in ("function_aux", "function_hierarchy", "consistency", "relation_view"):
                    if key in components:
                        main = main+loss_settings.get(key, 0)*components[key]
                gradient_records.append(backward_reconstruction(scaler.scale(main/group_size),
                    scaler.scale(auxiliary/group_size), model.parameters(), gradient_ratio,
                    project_conflicts=reconstruction_gradient=="project"))
            else:
                scaler.scale(loss/group_size).backward()
            boundary = (batch_index+1)%accumulation==0 or batch_index+1==len(loaders[0])
            if boundary:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("gradient_clip", 5), error_if_nonfinite=precision!="fp16")
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_skipped = scaler.get_scale()<old_scale
                skipped += int(step_skipped)
                update_ema(ema, model, epoch, ema_start, step_skipped)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            n = len(batch["y"])
            online_targets.append(batch["y"].detach().cpu().numpy())
            online_probabilities.append(output["logits"].detach().float().sigmoid().cpu().numpy())
            del output
            for key, value in {"total": loss, **components}.items():
                totals[key] = totals.get(key, 0)+float(value.detach())*n
            count += n
            if settings.get("max_steps") and global_step>=settings["max_steps"]:
                break
        train_report = metrics(np.concatenate(online_targets), np.concatenate(online_probabilities), source.statistics)
        del online_targets, online_probabilities
        y, probabilities, validation_loss = predict_loader(model, loaders[1], device, precision, config["loss"])
        report = metrics(y, probabilities, source.statistics)
        selected_model, validation_weights = model, "online"
        online_report, online_loss = report, validation_loss
        if ema is not None:
            online_selection = selection(y, probabilities, report, suite, protocol["threshold_strategy"])
            online_key = online_selection[0]
            if online_key > online_best_key:
                online_best_key, online_best_epoch = online_key, epoch
                online_best_state = checkpoint_state(model, cpu=True)
            validation_weights = "online_warmup"
            if int(ema.n_averaged) > 0:
                selected_model, validation_weights = ema.module, "ema"
                # An extra DataLoader iterator consumes a CPU base seed, even with
                # zero workers. Preserve it so this observation does not change Train RNG.
                with torch.random.fork_rng(devices=[]):
                    y, probabilities, validation_loss = predict_loader(selected_model, loaders[1], device, precision, config["loss"])
                report = metrics(y, probabilities, source.statistics)
        groups = evaluate_groups(y, probabilities, validation_groups, .5, source.statistics["labels"]["support"],
                                 source.statistics["function_projection"])
        if report["macro_ap"] is None:
            raise ValueError("No positive Validation labels for checkpoint selection")
        key, selection_threshold, selection_report = (
            online_selection if ema is not None and selected_model is model
            else selection(y, probabilities, report, suite, protocol["threshold_strategy"]))
        if key>best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_ranking = (report["macro_ap"], report["micro_ap"])
            best_selection_threshold = selection_threshold
            best_state = checkpoint_state(selected_model, cpu=True)
            best_weights = validation_weights
            torch.save({**static, "model_state": best_state, "best_epoch": best_epoch, "weight_source": best_weights}, run/"best.pt")
        else:
            stale += 1
        if save_candidates:
            alternate_key, alternate_threshold, _ = selection(
                y, probabilities, report, alternate_suite, protocol["threshold_strategy"])
            if alternate_best is None or alternate_key > tuple(alternate_best["key"]):
                if alternate_threshold is None:
                    alternate_threshold = (fit_label_thresholds(y, probabilities)
                                           if protocol["threshold_strategy"] == "per_label"
                                           else fit_global_threshold(y, probabilities))
                alternate_best = {
                    "key": list(alternate_key), "epoch": epoch, "threshold": alternate_threshold,
                    "weight_source": validation_weights, "validation_loss": validation_loss,
                    "model_state": checkpoint_state(selected_model, cpu=True),
                    "targets": torch.from_numpy(y.copy()),
                    "probabilities": torch.from_numpy(probabilities.copy()),
                }
        epoch_peak = torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
        peak_memory = max(peak_memory, epoch_peak or 0)
        history = {"epoch": epoch, "losses": {k: v/count for k, v in totals.items()}, "loss_settings": loss_settings,
                   "lr": optimizer.param_groups[0]["lr"],
                   "global_step": global_step, "skipped_amp_updates": skipped,
                   "train": train_report, "train_rows": count, "train_metrics_scope": "online_augmented_changing_weights",
                   "validation": report, "validation_loss": validation_loss,
                   "validation_loss_scope": "main_classification_clean_eval; no stochastic auxiliary reconstruction",
                   "validation_rows": len(y), "validation_groups": groups, "monitor_threshold": .5,
                   "best_macro_ap": best_ranking[0], "best_micro_ap": best_ranking[1], "best_epoch": best_epoch,
                   "early_stopping_count": stale, "peak_cuda_memory_bytes": epoch_peak}
        if getattr(model, "esm_tail", None) is not None:
            history["esm_lr"] = optimizer.param_groups[1]["lr"]
        if "metric_suite" in settings:
            history.update(metric_suite=suite, selection_key=list(key), best_selection_key=list(best_key),
                           selection_threshold=selection_threshold, selection_metrics=selection_report,
                           best_selection_threshold=best_selection_threshold)
        if save_candidates:
            history["alternate_selection"] = {"suite": alternate_suite, "best_epoch": alternate_best["epoch"],
                                               "best_key": alternate_best["key"]}
        if gradient_records:
            valid = [r for r in gradient_records if r["finite"]]
            history["reconstruction_gradient"] = {
                "mode":reconstruction_gradient, "batches":len(gradient_records), "finite_batches":len(valid),
                "conflicting_batches":sum(r["conflict"] for r in valid),
                "projected_batches":sum(r["projected"] for r in valid),
                "capped_batches":sum(r["capped"] for r in valid), "max_ratio":gradient_ratio}
            for key in ("cosine", "weighted_aux_main_ratio", "applied_aux_main_ratio"):
                observed = [r[key] for r in valid if r[key] is not None]
                history["reconstruction_gradient"]["mean_"+key] = sum(observed)/len(observed) if observed else None
        scheduler.step()
        ema_checkpoint = {}
        if ema is not None:
            history.update(online_validation=online_report, online_validation_loss=online_loss,
                           validation_weights=validation_weights, ema_updates=int(ema.n_averaged),
                           online_best_epoch=online_best_epoch)
            ema_checkpoint = {"ema_state": checkpoint_state(ema), "online_best_state": online_best_state,
                              "online_best_key": online_best_key, "online_best_epoch": online_best_epoch}
        numpy_rng = np.random.get_state()
        candidate_checkpoint = ({"alternate_best": alternate_best, "selection_tracking_start_epoch": candidate_start}
                                if save_candidates else {})
        history.update(seconds=time.perf_counter()-epoch_start, elapsed_seconds=elapsed_before+time.perf_counter()-started)
        save_last_checkpoint({**static, **ema_checkpoint, **candidate_checkpoint, "history_record": history,
                    "best_weights": best_weights, "model_state": checkpoint_state(model), "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(), "best_model_state": best_state,
                    "scaler_state": scaler.state_dict(), "global_step": global_step,
                    "best_key": best_key, "best_epoch": best_epoch, "epoch": epoch, "stale": stale,
                    "best_ranking": best_ranking, "best_selection_threshold": best_selection_threshold,
                    "python_rng": random.getstate(), "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                    "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type=="cuda" else None,
                    "elapsed_seconds": elapsed_before+time.perf_counter()-started,
                    "peak_cuda_memory_bytes": peak_memory if device.type=="cuda" else None}, run/"last.pt")
        with (run/"history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(history, allow_nan=False)+"\n")
        print(format_epoch(config["model_name"], source.data.track, settings["seed"], settings["epochs"], history, settings["patience"]), flush=True)
        if "metric_suite" in settings:
            print(f"Selection | suite={suite} key={key} best={best_key}@{best_epoch} "
                  f"threshold={selection_threshold} | {format_primary(suite, selection_report)}", flush=True)
        if branch_supervision == "shared_private":
            print(f"Branches | mode=shared_private stage=pre_feedback "
                  f"shared={history['losses']['branch_shared']:.6f} private={history['losses']['branch_private']:.6f} "
                  f"weight_each={loss_settings.get('branch', .1)/2:.3f} fusion={evidence_fusion}", flush=True)
        if ema is not None:
            mfp_text = (f"{online_report['mfp_detection']['f1']:.4f}"
                        if 'mfp_detection' in online_report else 'NA')
            print(f"Weights | Val={validation_weights} EMA-updates={int(ema.n_averaged)} | OnlineVal "
                  f"MacroAP={online_report['macro_ap']:.4f} MicroAP={online_report['micro_ap']:.4f} "
                  f"MFP-F1={mfp_text} "
                  f"Acc={online_report['accuracy']:.4f} Prec={online_report['precision']:.4f} "
                  f"Cov={online_report['coverage']:.4f} AT={online_report['absolute_true']:.4f} AF={online_report['absolute_false']:.4f}", flush=True)
        if consistency:
            print(f"Consistency | weight={loss_settings['consistency']:.4f} "
                  f"KL={history['losses']['consistency']:.6f} "
                  f"high={history['losses']['consistency_high']:.6f} "
                  f"low={history['losses']['consistency_low']:.6f}", flush=True)
        if rdrop:
            print(f"RDrop | weight={loss_settings['rdrop']:.4f} "
                  f"KL={history['losses']['rdrop']:.6f} "
                  f"weighted={loss_settings['rdrop']*history['losses']['rdrop']:.6f} "
                  f"views={2 if loss_settings['rdrop'] > 0 else 1}", flush=True)
        if function_aux:
            print(f"FunctionAux | task={source.statistics['function_auxiliary']['track']} "
                  f"weight={loss_settings['function_aux']:.4f} "
                  f"loss={history['losses']['function_aux']:.6f} "
                  f"hierarchy_weight={loss_settings['function_hierarchy']:.4f} "
                  f"hierarchy={history['losses']['function_hierarchy']:.6f}", flush=True)
        if relation_view_weight:
            print(f"RelationView | weight={loss_settings.get('relation_view', 0):.4f} "
                  f"stability={history['losses'].get('relation_stability', 0):.6f} "
                  f"anchor={history['losses'].get('relation_anchor', 0):.6f} dropout={relation_dropout:g}", flush=True)
        if gradient_records:
            g = history["reconstruction_gradient"]
            fmt = lambda value: "NA" if value is None else f"{value:.4f}"
            print(f"ReconGrad | mode={reconstruction_gradient} conflict={g['conflicting_batches']}/{g['finite_batches']} "
                  f"projected={g['projected_batches']}/{g['finite_batches']} "
                  f"capped={g['capped_batches']}/{g['finite_batches']} "
                  f"cosine={fmt(g['mean_cosine'])} ratio={fmt(g['mean_weighted_aux_main_ratio'])} "
                  f"applied={fmt(g['mean_applied_aux_main_ratio'])} max={gradient_ratio:g}", flush=True)
        if settings["patience"] and stale>=settings["patience"]:
            break
        if settings.get("max_steps") and global_step>=settings["max_steps"]:
            break
        if stop_file is not None and Path(stop_file).exists() and epoch < settings['epochs']:
            print(f'Paused after epoch {epoch}; continue from {run / "last.pt"}', flush=True)
            return {'paused': True, 'epoch': epoch, 'checkpoint': str(run/'last.pt')}
    restore_model_state(model, best_state)
    y, probabilities, validation_loss = predict_loader(model, loaders[1], device, precision, config["loss"])
    threshold = final_threshold(y, probabilities, suite, settings.get("threshold_strategy", "global"))
    torch.save({**static, "model_state": best_state, "best_epoch": best_epoch, "threshold": threshold,
                "weight_source": best_weights}, run/"best.pt")
    if device.type=="cuda":
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
    if terminal_resume:
        save_last_checkpoint({**checkpoint, **static, "elapsed_seconds": elapsed_before+time.perf_counter()-started}, run/"last.pt")
    result = {"model": config["model_name"], "task": source.data.track, "seed": settings["seed"],
              "best_epoch": best_epoch, "epochs_completed": epoch, "threshold": threshold, "threshold_fit_split": protocol["threshold_fit_split"],
              "global_step": global_step, "precision": precision, "accumulation_steps": accumulation,
              "threshold_strategy": settings.get("threshold_strategy", "global"), "trainable_parameters": params,
              "elapsed_seconds": elapsed_before+time.perf_counter()-started,
              "peak_cuda_memory_bytes": peak_memory if device.type=="cuda" else None,
              "diagnostic_subset": diagnostic, "validation": metrics(y, probabilities, source.statistics, threshold),
              "validation_loss": validation_loss, "validation_loss_scope": "main_classification_clean_eval",
              "validation_groups": evaluate_groups(y, probabilities, validation_groups, threshold,
                  source.statistics["labels"]["support"], source.statistics["function_projection"]),
              "labels": source.data.labels, "monitor_threshold": .5,
              "final_metrics_scope": ("best_checkpoint_at_fixed_0_5" if suite == "micro_f1"
                                      else "best_checkpoint_with_validation_fitted_threshold"),
              "annotation_based_metrics": True,
              "checkpoint": str(run/"best.pt")}
    result.update(metric_protocol=protocol, primary_metrics=primary_metrics(result["validation"], suite),
                  best_selection_key=list(best_key))
    export_predictions = settings.get("export_validation_predictions", False)
    if export_predictions:
        result["validation_predictions"] = export_validation(
            run, "best", source, datasets[1], y, probabilities, threshold, best_epoch, best_weights, suite)
    if save_candidates and alternate_best is not None:
        candidate = alternate_best
        alternate_path = run/f"{alternate_stem}.pt"
        torch.save({**static, "model_state": candidate["model_state"], "best_epoch": candidate["epoch"],
                    "threshold": candidate["threshold"], "weight_source": candidate["weight_source"],
                    "selection_suite": alternate_suite, "selection_key": candidate["key"]}, alternate_path)
        alternate_y, alternate_p = candidate["targets"].numpy(), candidate["probabilities"].numpy()
        result["alternate_selection"] = {
            "suite": alternate_suite, "best_epoch": candidate["epoch"], "key": candidate["key"],
            "threshold": candidate["threshold"], "threshold_fit_split": "val",
            "weight_source": candidate["weight_source"], "validation_loss": candidate["validation_loss"],
            "validation": metrics(alternate_y, alternate_p, source.statistics, candidate["threshold"]),
            "validation_groups": evaluate_groups(alternate_y, alternate_p, validation_groups, candidate["threshold"],
                source.statistics["labels"]["support"], source.statistics["function_projection"]),
            "checkpoint": str(alternate_path), "tracking_start_epoch": candidate_start,
            "scope": "Same optimizer trajectory; early stopping follows primary suite; Validation only",
        }
        if export_predictions:
            result["alternate_selection"]["validation_predictions"] = export_validation(
                run, alternate_stem, source, datasets[1], alternate_y, alternate_p, candidate["threshold"],
                candidate["epoch"], candidate["weight_source"], alternate_suite)
    if ema is not None:
        restore_model_state(model, online_best_state)
        raw_y, raw_p, raw_loss = predict_loader(model, loaders[1], device, precision, config["loss"])
        raw_threshold = final_threshold(raw_y, raw_p, suite, settings.get("threshold_strategy", "global"))
        torch.save({**static, "model_state": online_best_state, "best_epoch": online_best_epoch,
                    "threshold": raw_threshold, "weight_source": "online"}, run/"online_best.pt")
        result.update(weight_source=best_weights, ema_updates=int(ema.n_averaged),
                      online_reference={"best_epoch": online_best_epoch, "threshold": raw_threshold,
                          "threshold_fit_split": protocol["threshold_fit_split"], "validation_loss": raw_loss,
                          "validation": metrics(raw_y, raw_p, source.statistics, raw_threshold),
                          "validation_groups": evaluate_groups(raw_y, raw_p, validation_groups, raw_threshold,
                              source.statistics['labels']['support'], source.statistics['function_projection']),
                          "checkpoint": str(run/"online_best.pt"),
                          "scope": "same optimizer trajectory; stopping follows primary Validation, not an independent baseline training"})
        if device.type == "cuda":
            result['peak_cuda_memory_bytes'] = max(peak_memory, torch.cuda.max_memory_allocated(device))
    if source.data.track in THERAPEP_TRACKS:
        from .therapep import published_metrics
        result["therapep_validation"] = published_metrics(y, probabilities, threshold)
    result["elapsed_seconds"] = elapsed_before+time.perf_counter()-started
    write_json(run/"results.json.tmp", result)
    (run/"results.json.tmp").replace(run/"results.json")
    return result


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["model_name"], checkpoint["model_args"]).to(device)
    restore_model_state(model, checkpoint["model_state"])
    return model.eval(), checkpoint


def evaluate(checkpoint_path, data_root, feature_root, split, device, output, predictions=None, batch_size=32, decision_path=None,
             without_private_evidence=False, peptide_feature_root=None):
    model, checkpoint = load_model(checkpoint_path, device)
    if without_private_evidence:
        if checkpoint["model_name"] != "trirelay" or not hasattr(model, "evidence_partition"):
            raise ValueError("Private evidence ablation requires a shared-private TriRelay checkpoint")
        model.private_evidence_enabled = False
    source = MultimodalData(data_root, checkpoint["track"], checkpoint["statistics"])
    dataset = MultimodalPeptides(source, feature_root, split,
                                  peptide_feature_root=peptide_feature_root or checkpoint["config"]["data"].get("peptide_features"))
    validate_cache(dataset, checkpoint["config"], checkpoint["encoder"])
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_multimodal)
    y, probabilities = predict_loader(model, loader, device)
    threshold = checkpoint["threshold"]
    decision, predicted = None, None
    if decision_path:
        from .decisions import apply_decision
        decision = json.loads(Path(decision_path).read_text(encoding='utf-8'))
        predicted = apply_decision(probabilities, source.data.labels, checkpoint['track'], decision)
    grouped = {}
    for name, mask in source.groups(dataset.indices, y).items():
        grouped[name] = {"rows": int(mask.sum()), "metrics": evaluate_metrics(
            y[mask], probabilities[mask], threshold, source.statistics['labels']['support'],
            source.statistics['function_projection'], None if predicted is None else predicted[mask]) if mask.any() else None}
    result = {"model": checkpoint["model_name"], "task": checkpoint["track"], "split": split, "rows": len(dataset),
              "labels": source.data.labels, "threshold": threshold, "checkpoint": str(checkpoint_path),
              "weight_source": checkpoint.get("weight_source", "online"),
              "metrics": evaluate_metrics(y, probabilities, threshold, source.statistics['labels']['support'],
                  source.statistics['function_projection'], predicted), "groups": grouped,
              "annotation_based_metrics": True, "experimental_structure_group": None}
    if hasattr(model, "evidence_partition"):
        fusion_mode = model.evidence_partition.fusion
        ablation_scope = {
            'residual':'Zero private evidence residual in both reads; feedback/context and fusion recomputed; weights unchanged',
            'normalized_residual':'Zero normalized private residual in both reads; normalized shared coefficient remains 1; feedback/context and fusion recomputed; weights unchanged',
            'normalized_competitive':'Use normalized shared evidence alone in both reads (g=0, shared coefficient=1); feedback/context and fusion recomputed; weights unchanged'}
        result['inference_ablation'] = {
            'private_evidence_enabled': not without_private_evidence,
            'evidence_fusion': fusion_mode,
            'scope': ablation_scope[fusion_mode] if without_private_evidence else 'Unmodified shared-private checkpoint inference',
            'threshold_refit': False}
    if decision is not None:
        result['decision'] = decision
    if checkpoint["track"] in THERAPEP_TRACKS:
        from .therapep import published_metrics
        result["therapep"] = published_metrics(y, probabilities, threshold)
    settings = checkpoint["config"]["train"]
    suite = checkpoint.get("selection_suite", settings.get("metric_suite", "task_specific"))
    result["metric_protocol"] = protocol_description(suite, settings.get("threshold_strategy", "global"))
    result["primary_metrics"] = primary_metrics(result["metrics"], suite)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    if predictions:
        Path(predictions).parent.mkdir(parents=True, exist_ok=True)
        with Path(predictions).open("xb") as f:
            np.savez_compressed(f, row_id=source.data.rows[dataset.indices], ID=source.data.ids[dataset.indices],
                                targets=y, probabilities=probabilities, label_names=np.array(source.data.labels), threshold=threshold,
                                split=split, task=checkpoint['track'], decision_json=json.dumps(decision),
                                inference_ablation_json=json.dumps(result.get('inference_ablation')))
    return result


def predict_inputs(checkpoint_path, input_path, device="cpu", batch_size=32, decision_path=None):
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
    return prediction_records(checkpoint, probabilities, "provided_multimodal_fields", decision_path)


def prediction_records(checkpoint, probabilities, scope, decision_path=None):
    predicted = np.asarray(probabilities) >= checkpoint['threshold']
    if decision_path:
        from .decisions import apply_decision
        decision = json.loads(Path(decision_path).read_text(encoding='utf-8'))
        predicted = apply_decision(probabilities, checkpoint['labels'], checkpoint['track'], decision)
    return [{"probabilities": dict(zip(checkpoint["labels"], p)), "evidence_scope": scope,
             "predicted_labels": [label for label, keep in zip(checkpoint['labels'], selected) if keep]}
            for p, selected in zip(probabilities, predicted)]


def predict_sequences(checkpoint_path, sequences, device="cpu", decision_path=None):
    """Missing-modality inference; never fabricates structure or chemical descriptors."""
    from .encoder import FrozenEncoder
    from .multimodal import prepare_inputs
    model, checkpoint = load_model(checkpoint_path, device)
    info = checkpoint["encoder"]
    prefix = info.get("representation_layer")
    encoder = FrozenEncoder(info["name"], info["resolved_revision"], info["max_length"], device, prefix)
    peptide = info.get("peptide_encoder")
    peptide_encoder = (FrozenEncoder(peptide["name"], peptide["resolved_revision"], peptide["max_length"], device)
                       if peptide is not None else None)
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
        if checkpoint["track"] in THERAPEP_TRACKS:
            raw.pop("residue_chem_tokens")
        if peptide_encoder is not None:
            peptide_states, peptide_mask = peptide_encoder.encode(selected)
            peptide_residues = [peptide_states[i, peptide_mask[i]].half().float().cpu() for i in range(len(selected))]
            raw["esm_peptide"] = torch.nn.utils.rnn.pad_sequence(peptide_residues, batch_first=True).numpy()
        if prefix is not None:
            raw["esm_cls"] = states[:, 0].half().float().cpu().numpy()
            raw["esm_eos"] = torch.stack([states[i, len(s) + 1] for i, s in enumerate(selected)]).half().float().cpu().numpy()
        inputs = prepare_inputs(raw, checkpoint["statistics"])
        with torch.inference_mode():
            probs = model(to_device(inputs, device))["logits"].sigmoid().cpu().tolist()
        records = prediction_records(checkpoint, probs, "sequence_and_stereochemistry_only", decision_path)
        for s, record in zip(selected, records):
            record["sequence"] = s
        output.extend(records)
    return output


def summarize_results(runs, split="val"):
    """Plain JSON comparison. Never selects a model or threshold from Test."""
    from collections import defaultdict
    from .experiments import scalar_metrics, aggregate, metric_aliases
    buckets = defaultdict(list)
    for directory in runs:
        directory = Path(directory)
        metadata = json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
        result = json.loads((directory/"results.json").read_text(encoding="utf-8"))
        if result.get("diagnostic_subset"):
            raise ValueError(f"Diagnostic subset is not a formal comparison: {directory}")
        score = metric_aliases(result["validation"] if split=="val" else json.loads((directory/"test.json").read_text(encoding="utf-8"))["metrics"])
        config = metadata["config"]
        protocol = {"model": metadata["model"], "task": metadata["task"], "encoder": {key: metadata["encoder"][key] for key in ("name", "resolved_revision")},
                    "dataset": metadata["dataset"], "cohort": metadata["cohort"], "model_settings": config["model"],
                    "loss": config["loss"], "labels": metadata["labels"], "threshold_strategy": result["threshold_strategy"],
                    "training": {k: v for k, v in config["train"].items() if k not in ("seed", "device", "output", "num_workers")}}
        if "peptide_encoder" in metadata["encoder"]:
            protocol["peptide_encoder"] = {key: metadata["encoder"]["peptide_encoder"][key]
                                            for key in ("name", "resolved_revision")}
        buckets[json.dumps(protocol, sort_keys=True)].append({"run": str(directory), "seed": metadata["seed"], "metrics": score,
            "trainable_parameters": result["trainable_parameters"], "elapsed_seconds": result["elapsed_seconds"],
            "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"]})
    summary = []
    for protocol, records in buckets.items():
        if len({r["seed"] for r in records}) != len(records):
            raise ValueError("Repeated seed within one comparison protocol; distinguish the configurations")
        scalars = [scalar_metrics(r["metrics"]) for r in records]
        names = sorted(set().union(*(s.keys() for s in scalars)))
        score_summary = {name: aggregate([s.get(name) for s in scalars]) for name in names}
        per_label = {key: [aggregate([record["metrics"][key][j] for record in records]) for j in range(len(records[0]["metrics"][key]))]
                     for key in ("label_ap", "label_f1", "label_mcc", "label_auroc", "label_positive_support",
                                 "label_precision", "label_recall", "label_sensitivity", "label_specificity")}
        summary.append({"protocol": json.loads(protocol), "split": split, "seeds": [r["seed"] for r in records],
                        "metrics": score_summary, "per_label": per_label,
                        "resources": {key: aggregate([r[key] for r in records]) for key in ("trainable_parameters", "elapsed_seconds", "peak_cuda_memory_bytes")},
                        "runs": [r["run"] for r in records]})
    return {"split": split, "std_ddof": 1, "comparisons": summary}
