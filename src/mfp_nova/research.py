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
from .models import (HH_MODELS, HH_V15_RANKER, HH_V16, HH_V17, HH_V20, HH_V20_CHEMICAL_MODES,
                     PEPTIDE_V3, PEPTIDE_V6, PEPTIDE_V7, SEQUENCE_ONLY_MODELS, XBRIDGE, THERASET_BRIDGE,
                     THERAHELIX, THERAHELIX_JOINT)
from .multimodal import BackbonePeptides, MultimodalData, MultimodalPeptides, backbone_inputs, collate_multimodal
from .research_layers import observed_label_loss
from .consistency import bernoulli_consistency, consistency_weight
from .trainer import seed_everything, write_json
from .metric_protocol import (FIXED_THRESHOLD_SUITES, JACCARD_ORDER, protocol_description,
                              primary_metrics, selection, format_primary, final_threshold)


def make_model(name, arguments):
    if name == THERAHELIX_JOINT:
        from .models.therahelix_fusion import TheraHelixFusion
        shared = {key: arguments[key] for key in
                  ("input_dim", "peptide_input_dim", "num_labels", "vocab_size", "statistics")}
        helix = dict(arguments["helixset"], triangles=False, mix_latents=False, compact_inactive=True)
        return TheraHelixFusion(dict(shared, **arguments["theraset"]), dict(shared, **helix),
                                train_readers=True, branch_thresholds=arguments["branch_thresholds"])
    if name == THERAHELIX:
        from .models.therahelix import TheraHelix
        return TheraHelix(**arguments)
    if name == THERASET_BRIDGE:
        from .models.theraset_bridge import TheraSetBridge
        return TheraSetBridge(**arguments)
    if name == XBRIDGE:
        from .models.xbridge import XBridge
        return XBridge(**arguments)
    if name == PEPTIDE_V3:
        from .models.peptide_v3 import PeptideV3
        return PeptideV3(**arguments)
    if name == PEPTIDE_V6:
        from .models.peptide_v6 import PeptideV6
        arguments = dict(arguments)
        arguments.pop("vocab_size", None)
        return PeptideV6(**arguments)
    if name == HH_V20:
        from .models.hh_setreread_v20 import HHSetRereadV20
        return HHSetRereadV20(**arguments)
    if name == PEPTIDE_V7:
        from .models.peptide_v7 import PeptideV7
        arguments = dict(arguments)
        arguments.pop("vocab_size", None)
        return PeptideV7(**arguments)
    if name == HH_V17:
        from .models.hh_setreread_v17 import HHSetRereadV17
        return HHSetRereadV17(**arguments)
    if name == HH_V16:
        from .models.hh_setreread_v16 import HHSetRereadV16
        return HHSetRereadV16(**arguments)
    if name == HH_V15_RANKER:
        from .models.hh_setreread_v15_ranker import HHSetRereadV15Ranker
        return HHSetRereadV15Ranker(**arguments)
    if name == "hh_setreread_v14":
        from .models.hh_setreread_v14 import HHSetRereadV14
        return HHSetRereadV14(**arguments)
    if name == "hh_setreread_v13":
        from .models.hh_setreread_v13 import HHSetRereadV13
        return HHSetRereadV13(**arguments)
    if name == "hh_setreread_v12":
        from .models.hh_setreread_v12 import HHSetRereadV12
        return HHSetRereadV12(**arguments)
    if name == "hh_setreread":
        from .models.hh_setreread import HHSetReread
        return HHSetReread(**arguments)
    if name == "setreread":
        from .models.setreread import SetReread
        return SetReread(**arguments)
    if name == "termiscale":
        from .models.termiscale import TermiScale
        return TermiScale(**arguments)
    if name == "motifspan":
        from .models.motifspan import MotifSpan
        return MotifSpan(**arguments)
    if name == "privilegedbridge":
        from .models.privilegedbridge import PrivilegedBridge
        return PrivilegedBridge(**arguments)
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
    non_blocking = torch.device(device).type == "cuda"
    return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}


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
def predict_loader(model, loader, device, precision="fp32", loss_settings=None, decode_sets=False,
                   include_auxiliary=False, residue_nll=None, view_outputs=None, v5_forward=None,
                   v6_outputs=None, v7_outputs=None, xbridge_outputs=None):
    model.eval()
    targets, probabilities, decisions, count_decisions = [], [], [], []
    component_weights, component_counts = [], []
    auxiliary = []
    loss_total, rows = 0., 0
    residue_totals = {}
    view_probabilities, view_available = [], []
    if v6_outputs is not None:
        from .models.peptide_v6 import DIAGNOSTIC_NAMES, EVIDENCE_NAMES
        v6_batches = {f"{name}_probabilities": [] for name in DIAGNOSTIC_NAMES}
        v6_batches["evidence_weights"] = []
    if v7_outputs is not None:
        state_probabilities = []
    if xbridge_outputs is not None:
        xbridge_batches = {key: [] for key in ("auxiliary_logits", "reference_logits")}
    for batch in loader:
        moved = to_device(batch, device)
        with autocast_context(device, precision):
            if v6_outputs is not None:
                output = model(moved, diagnostics=True)
            elif residue_nll is None:
                output = model(moved, **(v5_forward or {}))
            else:
                update, totals = model.order_evidence(moved["aa"], moved["mask"], moved["y"])
                output = model(moved, order_update=update)
                for key, (numerator, denominator) in totals.items():
                    old_numerator, old_denominator = residue_totals.get(key, (0, 0))
                    residue_totals[key] = old_numerator + numerator, old_denominator + denominator
            if loss_settings is not None:
                if decode_sets:
                    loss = model.set_nll(output, moved["y"]).mean()
                else:
                    loss = observed_label_loss(output["logits"], moved["y"], model.statistics,
                                               loss_settings.get("unlabeled_weight", .3))
                loss_total += float(loss)*len(batch["y"])
                rows += len(batch["y"])
        # A NumPy view keeps the worker's shared tensor (and its FD) alive.
        targets.append(batch["y"].numpy().copy())
        if decode_sets:
            marginal, predicted, count_predicted = model.decode(output)
            probabilities.append(marginal.cpu().numpy())
            decisions.append(predicted.cpu().numpy())
            count_decisions.append(count_predicted.cpu().numpy())
            if include_auxiliary:
                auxiliary.append(output["logits"].float().sigmoid().cpu().numpy())
            weight = output["mixture_logits"].float().softmax(-1)
            count = output["count_logits"].float().softmax(-1)
            component_weights.append(weight.cpu().numpy())
            component_counts.append((weight[..., None] * count).sum(0).cpu().numpy())
        else:
            probabilities.append(output["logits"].float().sigmoid().cpu().numpy())
        if view_outputs is not None:
            view_probabilities.append(output["view_logits"].sigmoid().cpu().numpy())
            view_available.append(output["view_available"].cpu().numpy())
        if v6_outputs is not None:
            for name in DIAGNOSTIC_NAMES:
                v6_batches[f"{name}_probabilities"].append(output[name].sigmoid().cpu().numpy())
            v6_batches["evidence_weights"].append(output["evidence_weights"].cpu().numpy())
        if v7_outputs is not None:
            state_probabilities.append(output["state_logits"].softmax(-1).cpu().numpy())
        if xbridge_outputs is not None:
            for key, batches in xbridge_batches.items():
                if key in output:
                    batches.append(output[key].float().cpu().numpy())
    if xbridge_outputs is not None:
        xbridge_outputs.update({key: np.concatenate(batches) for key, batches in xbridge_batches.items() if batches})
    if v7_outputs is not None:
        from .models.peptide_v7 import AXIS_NAMES, STATE_NAMES
        v7_outputs.update(state_probabilities=np.concatenate(state_probabilities),
                          axis_names=np.array(AXIS_NAMES), state_names=np.array(STATE_NAMES))
    if v6_outputs is not None:
        v6_outputs.update({key: np.concatenate(batches) for key, batches in v6_batches.items()})
        v6_outputs["evidence_names"] = np.array(EVIDENCE_NAMES)
    if view_outputs is not None:
        from .models.theraset_v5 import V5_VIEWS
        view_outputs.update(view_probabilities=np.concatenate(view_probabilities),
                            view_available=np.concatenate(view_available), view_names=np.array(V5_VIEWS))
    if residue_nll is not None:
        for key, (numerator, denominator) in residue_totals.items():
            denominator = float(denominator)
            residue_nll[key] = float(numerator) / denominator if denominator else None
    result = np.concatenate(targets), np.concatenate(probabilities)
    if loss_settings is not None:
        result = (*result, loss_total / rows)
    if not decode_sets:
        return result
    weights = np.concatenate(component_weights)
    counts = np.sum(component_counts, axis=0)
    components = {"mean_weights": weights.mean(axis=0).tolist(),
                  "dominant_fraction": (np.bincount(weights.argmax(1), minlength=weights.shape[1]) / len(weights)).tolist(),
                  "weighted_count_distribution": (counts / counts.sum(axis=1, keepdims=True).clip(min=1e-8)).tolist()}
    result = (*result, np.concatenate(decisions), np.concatenate(count_decisions), components)
    return (*result, np.concatenate(auxiliary)) if include_auxiliary else result


def metrics(y, probabilities, statistics, threshold=.5, predicted=None):
    return evaluate_metrics(y, probabilities, threshold, statistics["labels"]["support"],
                            statistics["function_projection"], predicted)


def joint_set_protocol():
    return {"metric_suite": "jaccard", "threshold_strategy": "joint_expected_jaccard",
            "threshold_fit_split": None, "selection_threshold": None,
            "selection": [f"val_decoded_{key}" for key in JACCARD_ORDER] + ["earlier_epoch"]}


def v16_count_support(data):
    count = data.y[data.indices("train")].sum(axis=1, dtype=int)
    return np.bincount(count, minlength=len(data.labels) + 1)[1:].tolist()


def export_validation(run, stem, source, dataset, y, probabilities, threshold, epoch, weight_source, objective,
                      predicted=None, count_predicted=None, view_outputs=None, split="val"):
    """Reuse collected Val predictions; no extra model forward or Test access."""
    rows = dataset.indices
    path = run/f"{stem}_validation.npz"
    extra = {} if predicted is None else {"predicted": predicted, "prediction_rule": "joint_expected_jaccard",
                                         "count_predicted": count_predicted}
    np.savez_compressed(path, row_id=source.data.rows[rows], ID=source.data.ids[rows],
                        cluster_id=source.data.clusters[rows], targets=y, probabilities=probabilities,
                        label_names=np.array(source.data.labels), threshold=threshold, split=split,
                        task=source.data.track, epoch=epoch, weight_source=weight_source,
                        selection_objective=objective, **extra, **(view_outputs or {}))
    actual = y.astype(bool)
    predicted = probabilities >= threshold if predicted is None else predicted
    from .multimodal import independent_functions
    projection = source.statistics["function_projection"]
    summary = {
        "split": split, "epoch": epoch, "weight_source": weight_source, "selection_objective": objective,
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
    if config.get("model_name") in (PEPTIDE_V7, XBRIDGE, THERASET_BRIDGE, THERAHELIX, THERAHELIX_JOINT) and peptide is None:
        raise ValueError(f"{config['model_name']} requires both frozen ESM caches")
    if peptide is not None:
        if config["model_name"] not in (
                "helixset", "theraset", "privilegedbridge", *SEQUENCE_ONLY_MODELS,
                *HH_MODELS, HH_V16, HH_V17, XBRIDGE, THERASET_BRIDGE, THERAHELIX, THERAHELIX_JOINT):
            raise ValueError("This architecture does not support dual frozen encoders")
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
            if key == "branch" and config.get("model_name") == "privilegedbridge" and config["model"].get("stage") == "B":
                continue  # PB's specified teacher objective uses a fixed 0.2 branch term.
            loss[key] *= ramp
    if "rdrop" in loss:
        start = settings.get("rdrop_start_epoch", 6)
        duration = settings.get("rdrop_ramp_epochs", 5)
        loss["rdrop"] *= min(1., max(0., (epoch-start+1)/duration))
    if config.get("model_name") == XBRIDGE:
        for key in ("bridge", "axis_coverage"):
            if key in loss:
                loss[key] *= ramp
    if config.get("model_name") == THERASET_BRIDGE:
        start = settings.get("warmup_epochs", 5)
        loss["bridge"] = loss.get("bridge", 0.) * min(1., max(0., (epoch - start) / 5))
    if config.get("model_name") == "setreread" and epoch < settings.get("correction_start_epoch", 6):
        loss.update(edit_gain=0., recovery=0.)
    if config.get("model_name") == "privilegedbridge" and config["model"].get("stage") == "C":
        loss["kd"] = loss.get("kd", .2)*min(1., epoch/5)
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
            if key.removeprefix("module.").startswith(("esm_tail.", "peptide_lora.encoder.")) and ".lora_" not in key}


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
    if "v16_backbone_lr" in settings:
        groups = [{"params": model.backbone.parameters(), "lr": settings["v16_backbone_lr"]},
                  {"params": [p for key, p in parameters if not key.startswith("backbone.")]}]
    elif hasattr(model, "backbone_frozen"):
        groups = [
            {"params": [p for key, p in parameters if key.startswith("backbone.")]},
            {"params": [p for key, p in parameters if not key.startswith("backbone.")],
             "lr": settings["hh_frozen_learning_rate"] if model.backbone_frozen else settings["hh_head_learning_rate"]},
        ]
        groups = [group for group in groups if group["params"]]
    elif getattr(model, "stage", None) == "C":
        groups = [{"params": [p for key, p in parameters if key.startswith("sequence.")]},
                  {"params": [p for key, p in parameters if key.startswith("bridges.")],
                   "lr": settings.get("bridge_learning_rate", 2e-4)}]
        groups = [g for g in groups if g["params"]]
    elif getattr(model, "esm_tail", None) is not None or getattr(model, "peptide_lora", None) is not None:
        prefix = ("esm_tail.", "peptide_lora.")
        groups = [{"params": [p for key, p in parameters if not key.startswith(prefix)]},
                  {"params": [p for key, p in parameters if key.startswith(prefix)],
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
    fold = settings.get("hh_oof_fold")
    if fold is not None and (resume or resume_in_place or stop_file or config["model_name"] != "theraset"
                             or config["data"]["track"] != "fine39" or settings.get("initialize_from")):
        raise ValueError("HH V1.3 fold training requires a fresh Fine39 TheraSet from random initialization")
    device = torch.device(settings["device"])
    if device.type=="cuda":
        torch.cuda.set_device(device)
    seed_everything(settings["seed"])
    checkpoint = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    pb = config["model_name"] == "privilegedbridge"
    hh = config["model_name"] == "hh_setreread"
    hh_v12 = config["model_name"] == "hh_setreread_v12"
    hh_model = hh or hh_v12
    hh_v16 = config["model_name"] == HH_V16
    hh_v20 = config["model_name"] == HH_V20
    xbridge = config["model_name"] == XBRIDGE
    inner_fold = settings.get("xbridge_inner_fold")
    peptide_v3 = config["model_name"] == PEPTIDE_V3
    if peptide_v3 and config["data"]["track"] not in ("fine39", "major15", "rich48"):
        raise ValueError("V3 uses the project Fine39, Major15 or Rich48 split")
    if config["model_name"] in (PEPTIDE_V6, PEPTIDE_V7, XBRIDGE, THERAHELIX_JOINT) and config["data"]["track"] != "fine39":
        raise ValueError(f"{config['model_name']} uses the project Fine39 split")
    anchor = None
    if xbridge:
        anchor_path = settings.get("xbridge_reference")
        for key in ("bridge", "axis_coverage", "evidence_classification"):
            value = config["loss"].get(key, 0.3 if key == "evidence_classification" else 0.)
            if not math.isfinite(value) or value < 0 or (anchor_path and key == "evidence_classification" and value == 0):
                raise ValueError(f"Invalid XBridge loss weight: {key}")
        if not 0 < config["loss"].get("axis_temperature", .2) < math.inf or not -1 < config["loss"].get("bridge_margin", .5) < 1:
            raise ValueError("Use a positive axis temperature and cosine margin in (-1, 1)")
        if bool(anchor_path) != bool(settings.get("oof_files")):
            raise ValueError("XBridge residual training requires a frozen TheraSet and all three OOF exports")
        if anchor_path and checkpoint is None:
            anchor = torch.load(anchor_path, map_location="cpu", weights_only=True)
            if (anchor["model_name"] != "theraset" or anchor["track"] != "fine39"
                    or anchor["config"]["train"].get("hh_oof_fold") is not None
                    or anchor["config"]["train"].get("xbridge_inner_fold") != inner_fold):
                raise ValueError("The frozen reference must be the full TheraSet for the current Fine39 training scope")
        online = config["model"].get("online_encoder")
        if online and any(online[key] != config["peptide_encoder"][key] for key in ("name", "revision", "max_length")):
            raise ValueError("XBridge online ESM must match the peptide cache tokenizer and revision")
    if hh_v20 and config["data"]["track"] != "fine39":
        raise ValueError("HH V2.0 predicts Fine39 and uses Major15 as auxiliary supervision")
    ranker = config["model_name"] == HH_V15_RANKER
    initial = None
    initialize = settings.get("initialize_from")
    helix_recovery = checkpoint is None and config["model_name"] == "helixset" and initialize
    if helix_recovery:
        initial = torch.load(initialize, map_location="cpu", weights_only=True)
        original_args = initial["model_args"]
        if (initial["model_name"] != "helixset" or initial["track"] != "fine39"
                or config["data"]["track"] != "fine39" or original_args.get("latent_depth") != 6
                or original_args.get("decoder_depth") != 3 or original_args.get("evidence_readout")):
            raise ValueError("HelixSet recovery requires the original Fine39 6-latent/3-decoder checkpoint")
        config["model"] = {key: value for key, value in original_args.items()
                           if key not in ("input_dim", "peptide_input_dim", "vocab_size", "num_labels", "statistics")}
        config["model"]["decoder_depth"] = 2
    if checkpoint is None and (hh_model or hh_v16 or (config["model_name"] == "theraset" and initialize)):
        if not initialize:
            raise ValueError("HH-SetReread requires --initialize-from pointing to the Fine39 TheraSet best.pt")
        initial = torch.load(initialize, map_location="cpu", weights_only=True)
        if initial["model_name"] != "theraset" or initial["track"] != config["data"]["track"]:
            raise ValueError("Initialization requires a TheraSet checkpoint for the same task")
        backbone = {k: v for k, v in initial["model_args"].items()
                    if k not in ("input_dim", "num_labels", "vocab_size", "statistics", "peptide_input_dim")}
        if (hh_model or hh_v16) and (backbone.get("label_context", True) or not backbone.get("evidence_decoder", True)
                         or not initial["model_state"]["active"].sum(1).eq(1).all()):
            raise ValueError("HH requires the no_label_context_single_prototype checkpoint")
        if hh_v16:
            config["model"] = {**backbone, **config["model"]}
        else:
            hh_options = (("relation", "lambda_pair", "fixed_overlap") if hh else
                          ("readout", "esm_bypass", "condition_query", "split_evidence", "background_dropout"))
            config["model"] = {**backbone, **{key: config["model"][key] for key in hh_options
                                            if key in config["model"]}} if hh_model else backbone
        original = initial["config"]
        if hh:
            config["loss"] = {**original["loss"], **config["loss"]}
        elif not hh_v12 and not hh_v16:
            config["loss"] = dict(original["loss"])
        config["encoder"] = original["encoder"]
        if "peptide_encoder" in original:
            config["peptide_encoder"] = original["peptide_encoder"]
        else:
            config.pop("peptide_encoder", None)
            config["data"].pop("peptide_features", None)
    if pb and checkpoint is None and config["model"].get("stage", "A") in ("B", "C"):
        if not initialize:
            raise ValueError("PB B/C require initialize_from pointing to the selected preceding phase")
        initial = torch.load(initialize, map_location="cpu", weights_only=True)
        if initial["model_name"] != "privilegedbridge" or initial["track"] != config["data"]["track"]:
            raise ValueError("PB initialization model/task differs")
        if config["model"].get("stage") == "C":
            config["model"]["teacher_mode"] = initial["model_args"].get("teacher_mode", "zero_reference")
    if checkpoint is not None:
        # Resume preserves the trained model/priors/protocol; only execution controls change.
        original = checkpoint["config"]
        if config["model_name"] == "termiscale":
            for key, default in (("routing", "content"), ("pooling", "conditional")):
                if config["model"].get(key, default) != original["model"].get(key, default):
                    raise ValueError(f"TermiScale {key} differs from the resumed experiment; use its original switch")
        if (config["model_name"] == "motifspan" and
                config["model"].get("global_only", False) != original["model"].get("global_only", False)):
            raise ValueError("MotifSpan global-only mode differs from the checkpoint; use its original switch")
        if pb and config["model"].get("stage", "A") != original["model"].get("stage", "A"):
            raise ValueError("PB phase changes require initialize_from, not resume")
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
        inner_fold = original["train"].get("xbridge_inner_fold")
    suite = settings.get("metric_suite", "task_specific")
    fixed_threshold = settings.get("fixed_threshold", .5)
    if fixed_threshold != .5 and (config["model_name"] != "helixset" or suite != "jaccard"
                                  or not 0 < fixed_threshold < 1):
        raise ValueError("Custom fixed threshold requires HelixSet with the Jaccard suite")
    theraset_bridge = config["loss"].get("theraset_bridge")
    if config["model_name"] == THERASET_BRIDGE:
        if (config["data"]["track"] != "fine39" or suite != "jaccard"
                or theraset_bridge not in ("off", "align", "positive")):
            raise ValueError("TheraSet-Bridge requires Fine39, fixed-.50 Jaccard and an explicit bridge mode")
        weight = config["loss"].get("bridge", 0.)
        if (not math.isfinite(weight) or (weight > 0.) != (theraset_bridge != "off") or weight < 0.
                or not -1 < config["loss"].get("bridge_margin", .5) < 1):
            raise ValueError("Set a nonnegative bridge weight matching its mode and a margin in (-1, 1)")
    elif theraset_bridge is not None:
        raise ValueError("The bridge experiment uses model_name theraset_bridge")
    gram_weighting = settings.get("gram_positive_weighting")
    if gram_weighting is not None and (
            gram_weighting not in ("cluster", "cluster_context") or config["model_name"] != "theraset"
            or config["data"]["track"] != "fine39" or settings.get("v5_mode") != "baseline"
            or config["model"].get("v5_mode") is not None or fold is not None):
        raise ValueError("Gram positive weighting requires the Fine39 TheraSet baseline")
    if hh_model and (config["data"]["track"] != "fine39" or suite != "jaccard"
                     or config["encoder"]["max_length"] != 80):
        raise ValueError("HH requires Fine39, max_length=80 and the fixed-threshold jaccard protocol")
    if hh_v12 and (not config["data"].get("peptide_features")
                   or config.get("peptide_encoder", {}).get("max_length") != 80):
        raise ValueError("HH V1.2 requires length-80 caches for both encoders")
    if hh_v16 and (config["data"]["track"] != "fine39" or suite != "jaccard"
                    or config["encoder"]["max_length"] != 80
                    or not config["data"].get("peptide_features")
                    or config.get("peptide_encoder", {}).get("max_length") != 80):
        raise ValueError("HH V1.6 requires Fine39, dual length-80 caches and the jaccard protocol")
    hard_weight = config["loss"].get("hard_pair", 0.) if ranker else 0.
    if ranker:
        if (config["data"]["track"] != "fine39" or suite != "jaccard"
                or config["encoder"]["max_length"] != 80
                or not config["data"].get("peptide_features")
                or config.get("peptide_encoder", {}).get("max_length") != 80):
            raise ValueError("The V1.5 ranker uses Fine39, dual length-80 caches and the jaccard protocol")
        if not math.isfinite(hard_weight) or hard_weight < 0:
            raise ValueError("V1.5 hard pair loss weight must be finite and nonnegative")
        if hard_weight and len(settings.get("oof_files", ())) != 3:
            raise ValueError("V1.5 hard pair training requires three TheraSet --hh-oof-files")
    frozen_epochs = settings["hh_frozen_epochs"] if hh else settings.get("v16_frozen_epochs", 0)
    setreread = config["model_name"] == "setreread"
    if setreread and (config["data"]["track"] not in ("major15", "fine39") or suite != "micro_f1"):
        raise ValueError("SetReread uses major15 or fine39 with the fixed-0.50 micro_f1 protocol")
    rdrop = config["loss"].get("rdrop", 0)
    if pb and config["model"].get("stage", "A") == "B" and (rdrop or settings.get("ema_decay") is not None):
        raise ValueError("PB teacher phase B uses deterministic training without R-Drop/EMA")
    if not math.isfinite(rdrop) or rdrop < 0:
        raise ValueError("R-Drop weight must be finite and nonnegative")
    if rdrop:
        if config["model_name"] not in ("helixset", "theraset", "termiscale", "motifspan", "privilegedbridge",
                                       "hh_setreread", PEPTIDE_V3, PEPTIDE_V6, PEPTIDE_V7, THERASET_BRIDGE, THERAHELIX):
            raise ValueError("This model does not support R-Drop")
        if settings.get("rdrop_start_epoch", 6) < 1 or settings.get("rdrop_ramp_epochs", 5) < 1:
            raise ValueError("R-Drop start epoch and ramp length must be positive")
    protocol = protocol_description(suite, settings.get("threshold_strategy", "global"), fixed_threshold)
    if hh_v16:
        protocol = joint_set_protocol()
    setreread_start = settings.get("correction_start_epoch", 6) if setreread else 1
    if setreread:
        protocol["selection_start_epoch"] = setreread_start if settings["epochs"] >= setreread_start else 1
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
    reference_checkpoint = checkpoint or initial or anchor
    family_supervision = hh_v20 and config["model"]["mode"] in ("auxiliary", "full")
    localize_terminals = hh_v20 and config["model"]["mode"] not in HH_V20_CHEMICAL_MODES
    sequence_only = setreread or config["model_name"] in SEQUENCE_ONLY_MODELS
    train_indices = held_indices = None
    if inner_fold is not None:
        from .data import PeptideData
        from .hypothesis_data import family_folds

        if (config["model_name"] not in (XBRIDGE, "theraset", THERASET_BRIDGE)
                or config["data"]["track"] != "fine39" or inner_fold not in range(5)):
            raise ValueError("The internal holdout requires a Fine39 XBridge or TheraSet model and fold 0–4")
        data = PeptideData(config["data"]["root"], "fine39")
        pool = data.indices("train")
        held = family_folds(data.clusters[pool], 5) == inner_fold
        train_indices, held_indices = pool[~held], pool[held]
    oof_pool = train_indices
    if fold is not None:
        from .hh_v13 import fold_indices
        train_indices, held_indices = fold_indices(config["data"]["root"], fold, oof_pool)
    source = MultimodalData(config["data"]["root"], config["data"]["track"],
                           reference_checkpoint["statistics"] if reference_checkpoint else None,
                           config.get("min_factor_support", 20), config.get("max_factors", 32),
                           sequence_only=sequence_only, sequence_chemistry=setreread or hh_v20,
                           train_indices=train_indices, localize_terminals=localize_terminals)
    if pb and config["model"].get("stage", "A") != "A" and source.data.track in THERAPEP_TRACKS:
        raise ValueError("PB B/C need observed auxiliary inputs; use rich48 or stage A for author sequence tasks")
    if function_aux or family_supervision:
        source.enable_function_auxiliary()
    if anchor is not None and list(anchor["labels"]) != list(source.data.labels):
        raise ValueError("Reference label order differs from Fine39")
    limit = settings.get("limit_samples")
    diagnostic = bool(limit or settings.get("max_steps") or (checkpoint and checkpoint["config"]["train"].get("max_steps")))
    if anchor is not None and not diagnostic and any(anchor["config"]["train"].get(key) for key in ("limit_samples", "max_steps")):
        raise ValueError("Full-scope XBridge requires a full-scope reference checkpoint")
    use_oof = bool((xbridge and settings.get("oof_files")) or hard_weight or (hh_v16 and
                   (config["loss"].get("hard_row_weight", 0.) or config["loss"].get("retain", 0.))))
    if use_oof:
        if len(settings.get("oof_files", ())) != 3:
            raise ValueError("Provide three TheraSet OOF exports")
        from .hh_v13 import ScoredPeptides, load_oof
        train_indices, oof_scores = load_oof(source.data, settings["oof_files"], diagnostic, oof_pool)
    # Diagnostic optimization subsets retain full original-Train priors/statistics:
    # shrinking the subset must not erase empirical factors or prototype capacity.
    if peptide_v3:
        datasets = [BackbonePeptides(source, split, config["encoder"]["max_length"], limit) for split in ("train", "val")]
    else:
        datasets = [MultimodalPeptides(source, config["data"]["features"], "train" if (fold is not None or inner_fold is not None) else split, limit,
                                       config["data"].get("peptide_features"),
                                       sequence_only=sequence_only or (pb and config["model"].get("stage", "A") == "A"),
                                       indices=indices) for split, indices in (("train", train_indices), ("val", held_indices))]
        validate_cache(datasets[0], config, reference_checkpoint["encoder"] if reference_checkpoint else None)
        validate_cache(datasets[1], config, datasets[0].meta)
        if datasets[0].meta["dataset"] != source.statistics["dataset"]:
            raise ValueError("Cache dataset version differs from multimodal data")
    generator = torch.Generator().manual_seed(settings["seed"])
    num_workers = settings["num_workers"]
    pin_memory = device.type == "cuda" and num_workers > 0
    training_data = datasets[0]
    gram_weight_summary = None
    if gram_weighting is not None:
        from .multimodal import gram_positive_multipliers
        multipliers, gram_weight_summary = gram_positive_multipliers(source.data, gram_weighting)
        training_data.positive_multipliers = torch.from_numpy(multipliers[training_data.indices])
        print(f"Gram positive weighting | {gram_weighting} | {json.dumps(gram_weight_summary)}", flush=True)
    if use_oof:
        training_data = ScoredPeptides(training_data, oof_scores[np.searchsorted(train_indices, training_data.indices)])
    loader_settings = dict(num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_multimodal)
    batch_sampler = None
    if xbridge or theraset_bridge is not None:
        from .xbridge_data import BridgePeptides, PositivePairBatches

        rows = datasets[0].indices
        clusters = source.data.clusters[rows]
        training_data = BridgePeptides(training_data, clusters)
        if theraset_bridge is not None or settings.get("xbridge_pair_batches"):
            batch_sampler = PositivePairBatches(source.data.y[rows], clusters, settings["batch_size"], generator)
    train_loader = (DataLoader(training_data, batch_sampler=batch_sampler, **loader_settings)
                    if batch_sampler is not None else DataLoader(training_data, batch_size=settings["batch_size"],
                                                                shuffle=True, generator=generator, **loader_settings))
    loaders = [train_loader, DataLoader(datasets[1], batch_size=settings["batch_size"], **loader_settings)]
    print(f"DataLoader | workers={num_workers} pin_memory={pin_memory}", flush=True)
    args = {"num_labels": len(source.data.labels), "statistics": source.statistics["labels"], **config["model"]}
    if not peptide_v3:
        args.update(input_dim=datasets[0].meta["input_dim"], vocab_size=len(source.statistics["chemical_vocab"])+2)
    if hh_v16:
        args["count_support"] = v16_count_support(source.data)
    if xbridge:
        args["function_projection"] = source.statistics["function_projection"]
        args["reference_args"] = (checkpoint["model_args"]["reference_args"] if checkpoint else
                                  anchor["model_args"] if anchor is not None else None)
    if config["model_name"] == PEPTIDE_V7 and args.get("head", "structured") == "structured":
        from .models.peptide_v7 import fit_axis_counts
        args["axis_counts"] = checkpoint["model_args"]["axis_counts"] if checkpoint else fit_axis_counts(source.data)
    if setreread:
        args["function_projection"] = source.statistics["function_projection"]
    if hh:
        from .models.hh_setreread import axis_statistics
        args["axis_statistics"] = checkpoint["model_args"]["axis_statistics"] if checkpoint else axis_statistics(source)
    if "peptide_encoder" in datasets[0].meta:
        args["peptide_input_dim"] = datasets[0].meta["peptide_encoder"]["input_dim"]
    if function_aux:
        args["function_aux_statistics"] = source.statistics["function_auxiliary"]
    if family_supervision:
        args["family_statistics"] = source.statistics["function_auxiliary"]
    model = make_model(config["model_name"], args).to(device)
    if anchor is not None:
        restore_model_state(model.reference, anchor["model_state"])
        del anchor
    if initial is not None and not pb:
        if helix_recovery:
            if list(initial["labels"]) != list(source.data.labels):
                raise ValueError("HelixSet initialization labels differ from Fine39")
            initial["model_state"] = {name: value for name, value in initial["model_state"].items()
                                      if not name.startswith("decoder.2.")}
        restore_model_state(model.backbone if hh_model or hh_v16 else model, initial["model_state"])
        del initial
    elif initial is not None:
        previous = make_model("privilegedbridge", initial["model_args"]).to(device)
        restore_model_state(previous, initial["model_state"])
        model.initialize_from(previous)
        del previous, initial
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
    if hh:
        model.freeze_backbone(bool(frozen_epochs and (checkpoint is None or checkpoint["epoch"] <= frozen_epochs)))
    if hh_v16:
        model.freeze_backbone(bool(frozen_epochs and (checkpoint is None or checkpoint["epoch"] < frozen_epochs)))
    optimizer = make_optimizer(model, settings)
    precision = settings.get("precision", "fp32")
    accumulation = settings.get("accumulation_steps", 1)
    scaler = torch.amp.GradScaler("cuda", enabled=precision=="fp16", init_scale=1024.)
    planned_epochs = checkpoint.get("planned_epochs", settings["epochs"]) if checkpoint else settings["epochs"]
    warmup = settings.get("warmup_epochs", 5)
    decay_epochs = settings.get("lr_decay_epochs", planned_epochs)
    if "lr_decay_epochs" in settings and decay_epochs <= warmup:
        raise ValueError("lr_decay_epochs must exceed warmup_epochs")
    save_candidates = settings.get("save_selection_candidates", False) and suite != "micro_f1" and not hh_v16
    alternate_suite = {"traditional5": "task_specific", "jaccard": "micro_f1"}.get(suite, "traditional5")
    alternate_stem = {"traditional5": "best_ap", "jaccard": "best_micro_f1"}.get(suite, "best_accuracy")
    alternate_best = checkpoint.get("alternate_best") if checkpoint and save_candidates else None
    candidate_start = (checkpoint.get("selection_tracking_start_epoch", checkpoint["epoch"]+1)
                       if checkpoint else 1)
    ema_decay = settings.get("ema_decay")
    ema_start = settings.get("ema_start_epoch", warmup+1)
    if ema_decay is not None and (not 0 < ema_decay < 1 or ema_start < 1
                                 or (suite != "jaccard" and ema_start > planned_epochs)):
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

    def phase_schedule(optimizer, frozen):
        if suite != "jaccard":
            return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        length = decay_epochs if hh_v16 else (frozen_epochs if frozen else decay_epochs-frozen_epochs)
        def ratio(index):
            if index < warmup:
                return (index+1)/warmup
            progress = min(1., (index-warmup+1)/max(1, length-warmup))
            return .05+.95*.5*(1+math.cos(math.pi*progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, ratio)

    scheduler = phase_schedule(optimizer, hh and model.backbone_frozen)
    start, best_key, best_epoch, stale = 1, (-float("inf"), -float("inf")), 0, 0
    best_state = None
    best_ranking = (None, None)
    best_selection_threshold = None
    global_step = 0
    phase_best = -float("inf")
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
        phase_best = checkpoint.get("phase_best", best_key[0])
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
    params = sum(p.numel() for p in model.parameters() if p.requires_grad or hh)
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
                "monitor_threshold": fixed_threshold, "train_metrics_scope": "online_augmented_changing_weights",
                "validation_loss_scope": "main_classification_clean_eval; no stochastic auxiliary reconstruction"}
    if gram_weight_summary is not None:
        metadata["gram_positive_weights"] = gram_weight_summary
    if source.data.track in THERAPEP_TRACKS:
        metadata["label_semantics"] = ("Positive-only 48 functional targets; 0 is unannotated; no therapeutic flag"
            if source.data.track == "therapep48" else
            "Author-published 49-bit targets; synthetic backgrounds when present are assigned zeros; therapeutic flag is supervised")
        metadata["split_protocol"] = "Published CSV members, including duplicates; no cluster isolation imposed"
        metadata["modality_scope"] = "Sequence only for every source; no positive-only structure/descriptor availability shortcut"
    if source.data.track == "rich48":
        metadata["label_semantics"] = "48 author functional targets; 0 is unannotated; no therapeutic flag"
        metadata["split_protocol"] = "Author positive members retained; no homology isolation imposed"
        metadata["modality_scope"] = ("Canonical backbone only" if peptide_v3 else
                                      "Source-matched raw structure/properties and annotation masks; Train-only preprocessing")
    if getattr(model, "esm_tail", None) is not None:
        metadata["esm_adaptation"] = {
            "mode": "frozen_prefix_cached_trainable_tail_lora",
            "prefix_layer": config["encoder"]["representation_layer"],
            "trainable_parameters": sum(p.numel() for p in model.esm_tail.parameters() if p.requires_grad),
            "pretrained_weights_in_checkpoint": False,
            "learning_rate": settings.get("esm_learning_rate", 1e-5)}
    if function_aux or family_supervision:
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
            "inputs": ("same sequence and cached residues; independent ordinary network dropout masks"
                       if config["model_name"] in (PEPTIDE_V6, PEPTIDE_V7) else
                       "same modality-dropout view; independent ordinary network dropout masks"),
            "train_metrics": "first stochastic forward only", "validation": "single clean forward; main ASL only"}
    metadata["phys_missing_values"] = "zero_after_scaling" if source.statistics["phys"].get("zero_missing", False) else "legacy_median_after_scaling"
    if save_candidates:
        metadata["selection_candidates"] = {"primary": suite, "alternate": alternate_suite,
                                            "tracking_start_epoch": candidate_start,
                                            "stopping": "primary suite only; one shared optimizer trajectory"}
    if not resume_in_place or checkpoint is None:
        if inner_fold is not None:
            metadata["validation_scope"] = "internal_fine39_train_clusters"
            metadata["internal_fold"] = inner_fold
        if fold is not None:
            metadata["val_rows"] = 0
            metadata["held_rows"] = len(datasets[1])
            metadata["metric_protocol"] = {"selection": "fixed_epoch_oof", "fold": fold}
        write_json(run/"metadata.json", metadata)
    static = {"model_name": config["model_name"], "model_args": args, "config": config, "statistics": source.statistics,
              "labels": source.data.labels, "track": source.data.track, "encoder": datasets[0].meta, "planned_epochs": planned_epochs}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    elapsed_before = checkpoint.get("elapsed_seconds", 0) if checkpoint else 0
    peak_memory = (checkpoint.get("peak_cuda_memory_bytes", 0) or 0) if checkpoint else 0
    validation_groups = (source.groups(datasets[1].indices, source.data.y[datasets[1].indices])
                         if fold is None else {})

    def collect_validation(candidate, view_outputs=None, xbridge_outputs=None):
        residue_nll = {} if getattr(candidate, "order_evidence", None) is not None else None
        values = predict_loader(candidate, loaders[1], device, precision, config["loss"],
                                decode_sets=hh_v16, residue_nll=residue_nll, view_outputs=view_outputs,
                                xbridge_outputs=xbridge_outputs)
        return values if hh_v16 else (*values, None, None, residue_nll)

    epoch = checkpoint["epoch"] if checkpoint else 0
    if suite == "jaccard" and checkpoint is None and fold is None and not hh_v16:
        y, probabilities, validation_loss, _, _, _ = collect_validation(model)
        report = metrics(y, probabilities, source.statistics, fixed_threshold)
        best_key, best_selection_threshold, _ = selection(y, probabilities, report, suite, protocol["threshold_strategy"],
                                                           fixed_threshold)
        phase_best = best_key[0]
        best_ranking = (report["macro_ap"], report["micro_ap"])
        best_state = checkpoint_state(model, cpu=True)
        online_best_state, online_best_key = best_state, best_key
        if save_candidates:
            alternate_report = report if fixed_threshold == .5 else metrics(y, probabilities, source.statistics)
            alternate_best = {"key": [alternate_report["micro_f1"]], "epoch": 0, "threshold": .5,
                              "weight_source": "online", "validation_loss": validation_loss,
                              "model_state": best_state, "targets": torch.from_numpy(y),
                              "probabilities": torch.from_numpy(probabilities)}
            candidate_start = 0
    # A crash after the final last.pt but before results.json needs finalization,
    # not an extra optimizer epoch or a restarted early-stopping counter.
    terminal_resume = checkpoint is not None and (start>settings["epochs"] or
                        (start>frozen_epochs+1 and settings["patience"] and stale>=settings["patience"]))
    for epoch in (() if terminal_resume else range(start, settings["epochs"]+1)):
        if hh_v16 and frozen_epochs and epoch == frozen_epochs + 1:
            model.freeze_backbone(False)
            stale = 0
        if hh and frozen_epochs and epoch == frozen_epochs+1:
            model.freeze_backbone(False)
            optimizer = make_optimizer(model, settings)
            scheduler = phase_schedule(optimizer, False)
            scaler = torch.amp.GradScaler("cuda", enabled=precision=="fp16", init_scale=1024.)
            if ema is not None:
                ema.n_averaged.zero_()
            previous = checkpoint["history_record"] if checkpoint and epoch == start else history
            phase_best = previous["online_validation" if ema is not None else "validation"]["sample_jaccard"]
            stale = 0
        phase_epoch = epoch-frozen_epochs if hh and epoch > frozen_epochs else epoch
        if setreread and epoch == setreread_start:
            # Keep warmup metrics, then compare all SetReread variants over the same trained phase.
            best_key = online_best_key = (-float("inf"), -float("inf"))
            stale = 0
        model.train()
        totals, count, skipped = {}, 0, 0
        gradient_records = []
        epoch_start = time.perf_counter()
        epoch_start_step = global_step
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        online_targets, online_probabilities = [], []
        defer_metrics = config["model_name"] == THERAHELIX
        batch_losses = {}
        pending_losses = []
        loss_settings = epoch_loss_settings(config, phase_epoch)
        if hh:
            loss_settings["hh_ramp"] = min(1., epoch/5)
            if model.backbone_frozen:
                loss_settings["rdrop"] = 0.
        optimizer.zero_grad(set_to_none=True)
        valid_microbatches = 0
        data_wait_seconds = 0.
        batch_done = time.perf_counter()
        for batch_index, batch in enumerate(loaders[0]):
            data_wait_seconds += time.perf_counter() - batch_done
            cpu_targets = batch["y"]
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
            if defer_metrics:
                pending_losses.append(loss.detach())
            elif not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite research loss")
            backward_enabled = not (pb and model.stage == "B" and not output["observed"].any())
            valid_microbatches += int(backward_enabled)
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
            elif backward_enabled:
                scaler.scale(loss/group_size).backward()
            boundary = (batch_index+1)%accumulation==0 or batch_index+1==len(loaders[0])
            if defer_metrics and boundary:
                if not torch.stack(pending_losses).isfinite().all():
                    raise FloatingPointError("Nonfinite research loss")
                pending_losses.clear()
            if boundary and valid_microbatches:
                if pb and model.stage == "B":
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(group_size/valid_microbatches)
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("gradient_clip", 5), error_if_nonfinite=precision!="fp16")
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_skipped = scaler.get_scale()<old_scale
                skipped += int(step_skipped)
                update_ema(ema, model, phase_epoch, ema_start, step_skipped)
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if boundary:
                valid_microbatches = 0
            n = len(batch["y"])
            if defer_metrics:
                if fold is None:
                    online_targets.append(cpu_targets.numpy())
                    online_probabilities.append(output["logits"].detach().float().sigmoid())
                for key, value in {"total": loss, **components}.items():
                    batch_losses.setdefault(key, []).append((value.detach(), n))
            elif fold is None:
                online_targets.append(batch["y"].detach().cpu().numpy())
                online_probabilities.append(output["logits"].detach().float().sigmoid().cpu().numpy())
            del output
            if not defer_metrics:
                for key, value in {"total": loss, **components}.items():
                    totals[key] = totals.get(key, 0)+float(value.detach())*n
            count += n
            if settings.get("max_steps") and global_step>=settings["max_steps"]:
                break
            batch_done = time.perf_counter()
        if defer_metrics:
            for key, entries in batch_losses.items():
                values, sizes = zip(*entries)
                totals[key] = float(np.dot(torch.stack(values).cpu().numpy().astype(np.float64), sizes))
            if fold is None:
                online_probabilities = [torch.cat(online_probabilities).cpu().numpy()]
        train_seconds = time.perf_counter() - epoch_start
        if fold is not None:
            scheduler.step()
            print(f"Fold {fold} | epoch {epoch}/{settings['epochs']} | "
                  f"loss={totals['total']/count:.6f} | {train_seconds:.1f}s", flush=True)
            if settings.get("max_steps") and global_step >= settings["max_steps"]:
                break
            continue
        train_report = metrics(np.concatenate(online_targets), np.concatenate(online_probabilities), source.statistics,
                               fixed_threshold)
        del online_targets, online_probabilities
        validation_start = time.perf_counter()
        y, probabilities, validation_loss, predicted, _, components = collect_validation(model)
        report = metrics(y, probabilities, source.statistics, fixed_threshold, predicted=predicted)
        selected_model, validation_weights = model, "online"
        online_report, online_loss = report, validation_loss
        if ema is not None:
            online_selection = selection(y, probabilities, report, suite, protocol["threshold_strategy"], fixed_threshold)
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
                    y, probabilities, validation_loss, predicted, _, components = collect_validation(selected_model)
                report = metrics(y, probabilities, source.statistics, fixed_threshold, predicted=predicted)
        groups = evaluate_groups(y, probabilities, validation_groups, fixed_threshold, source.statistics["labels"]["support"],
                                 source.statistics["function_projection"], predicted)
        validation_seconds = time.perf_counter() - validation_start
        if report["macro_ap"] is None:
            raise ValueError("No positive Validation labels for checkpoint selection")
        key, selection_threshold, selection_report = (
            online_selection if ema is not None and selected_model is model
            else selection(y, probabilities, report, suite, protocol["threshold_strategy"], fixed_threshold))
        if hh_v16:
            selection_threshold = None
        if key>best_key:
            best_key, best_epoch = key, epoch
            if suite != "jaccard":
                stale = 0
            best_ranking = (report["macro_ap"], report["micro_ap"])
            best_selection_threshold = selection_threshold
            best_state = checkpoint_state(selected_model, cpu=True)
            best_weights = validation_weights
            torch.save({**static, "model_state": best_state, "best_epoch": best_epoch, "weight_source": best_weights}, run/"best.pt")
        elif suite != "jaccard" and epoch >= setreread_start:
            stale += 1
        if suite == "jaccard":
            if key[0] > phase_best:
                phase_best, stale = key[0], 0
            else:
                stale += 1
        if save_candidates:
            alternate_report = report if fixed_threshold == .5 else metrics(y, probabilities, source.statistics)
            alternate_key, alternate_threshold, _ = selection(
                y, probabilities, alternate_report, alternate_suite, protocol["threshold_strategy"])
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
                   "validation_rows": len(y), "validation_groups": groups, "monitor_threshold": fixed_threshold,
                   "best_macro_ap": best_ranking[0], "best_micro_ap": best_ranking[1], "best_epoch": best_epoch,
                   "early_stopping_count": stale, "peak_cuda_memory_bytes": epoch_peak}
        if hh:
            history.update(hh_phase="frozen" if model.backbone_frozen else "joint",
                           hh_phase_epoch=phase_epoch, hh_head_lr=optimizer.param_groups[-1]["lr"],
                           phase_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        if hh_v16:
            history.update(validation_loss_scope="recorded_set_nll", decision_rule="joint_expected_jaccard",
                           train_metrics_scope="online_auxiliary_label_logits",
                           v16_phase="frozen" if model.backbone_frozen else "joint",
                           v16_head_lr=optimizer.param_groups[-1]["lr"], joint_components=components)
        elif components is not None:
            history["validation_order_nll"] = components
        if getattr(model, "esm_tail", None) is not None:
            history["esm_lr"] = optimizer.param_groups[1]["lr"]
        if pb and model.stage == "C":
            history["bridge_lr"] = optimizer.param_groups[-1]["lr"]
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
        if global_step > epoch_start_step:
            scheduler.step()
        ema_checkpoint = {}
        if ema is not None:
            history.update(online_validation=online_report, online_validation_loss=online_loss,
                           validation_weights=validation_weights, ema_updates=int(ema.n_averaged),
                           online_best_epoch=online_best_epoch)
            if suite == "jaccard" and phase_epoch == ema_start-1:
                ema.update_parameters(model)
            ema_checkpoint = {"ema_state": checkpoint_state(ema), "online_best_state": online_best_state,
                              "online_best_key": online_best_key, "online_best_epoch": online_best_epoch}
        numpy_rng = np.random.get_state()
        candidate_checkpoint = ({"alternate_best": alternate_best, "selection_tracking_start_epoch": candidate_start}
                                if save_candidates else {})
        history.update(seconds=time.perf_counter()-epoch_start, elapsed_seconds=elapsed_before+time.perf_counter()-started,
                       train_seconds=train_seconds, data_wait_seconds=data_wait_seconds, validation_seconds=validation_seconds)
        phase_checkpoint = {"phase_best": phase_best} if suite == "jaccard" else {}
        save_last_checkpoint({**static, **ema_checkpoint, **candidate_checkpoint, **phase_checkpoint, "history_record": history,
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
        if hh_v16:
            print(f"V1.6 | Phase={history['v16_phase']} BackboneLR={history['lr']:.3g} "
                  f"HeadLR={history['v16_head_lr']:.3g}", flush=True)
        print(f"Timing | Train={train_seconds:.1f}s DataWait={data_wait_seconds:.1f}s Val={validation_seconds:.1f}s", flush=True)
        if theraset_bridge is not None and "bridge" in history["losses"]:
            terms = history["losses"]
            print(f"Bridge | mode={theraset_bridge} weight={loss_settings['bridge']:.4f} "
                  f"loss={terms['bridge']:.6f} cluster_pairs={terms['bridge_cluster_pairs']:.1f} "
                  f"active={terms['bridge_active']:.1%}", flush=True)
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
        if epoch>frozen_epochs and settings["patience"] and stale>=settings["patience"]:
            break
        if settings.get("max_steps") and global_step>=settings["max_steps"]:
            break
        if stop_file is not None and Path(stop_file).exists() and epoch < settings['epochs']:
            print(f'Paused after epoch {epoch}; continue from {run / "last.pt"}', flush=True)
            return {'paused': True, 'epoch': epoch, 'checkpoint': str(run/'last.pt')}
    if fold is not None:
        from .hh_v13 import export_fold
        selected = ema.module if ema is not None and int(ema.n_averaged)>0 else model
        fold_checkpoint = {**static, "model_state": checkpoint_state(selected), "threshold": .5,
                           "best_epoch": epoch, "weight_source": "ema" if selected is not model else "online"}
        torch.save(fold_checkpoint, run/"best.pt")
        return export_fold(run, selected, loaders[1], device, fold, epoch, diagnostic, time.perf_counter()-started)
    restore_model_state(model, best_state)
    view_outputs = {} if getattr(model, "v5_readout", None) is not None else None
    xbridge_outputs = {} if xbridge else None
    y, probabilities, validation_loss, predicted, count_predicted, components = collect_validation(model, view_outputs, xbridge_outputs)
    threshold = final_threshold(y, probabilities, suite, settings.get("threshold_strategy", "global"), fixed_threshold)
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
              "diagnostic_subset": diagnostic, "validation": metrics(y, probabilities, source.statistics, threshold, predicted),
              "validation_loss": validation_loss, "validation_loss_scope": "main_classification_clean_eval",
              "validation_groups": evaluate_groups(y, probabilities, validation_groups, threshold,
                  source.statistics["labels"]["support"], source.statistics["function_projection"], predicted),
              "labels": source.data.labels, "monitor_threshold": fixed_threshold,
              "final_metrics_scope": (f"best_checkpoint_at_{protocol['threshold_strategy']}" if suite in FIXED_THRESHOLD_SUITES
                                      else "best_checkpoint_with_validation_fitted_threshold"),
              "annotation_based_metrics": True,
              "checkpoint": str(run/"best.pt")}
    result.update(metric_protocol=protocol, primary_metrics=primary_metrics(result["validation"], suite),
                  best_selection_key=list(best_key))
    if hh_v16:
        result.update(threshold_strategy="joint_expected_jaccard", validation_loss_scope="recorded_set_nll",
                      final_metrics_scope="best_checkpoint_with_decoded_label_sets",
                      validation_marginal_fixed_threshold=metrics(y, probabilities, source.statistics, .5),
                      validation_marginal_top_count=metrics(y, probabilities, source.statistics, .5, count_predicted),
                      joint_components=components)
    if setreread:
        result["correction_active"] = bool(model.correction_active)
    if "v4_mode" in settings:
        result["v4_mode"] = settings["v4_mode"]
    if "v5_mode" in settings:
        result["v5_mode"] = settings["v5_mode"]
    if gram_weight_summary is not None:
        result["gram_positive_weighting"] = gram_weighting
        result["gram_positive_weights"] = gram_weight_summary
    if getattr(model, "v5_readout", None) is not None and model.v5_readout.relation is not None:
        from .models.theraset_v5 import function_adjacency
        actual = model.v5_readout.relation.cpu().numpy() > 0
        semantic = function_adjacency(source.data.labels)
        result["v5_graph"] = {"undirected_edges": int(actual.sum() // 2),
                              "overlap_with_semantic": float((actual & semantic).sum() / max(actual.sum(), 1))}
    if not hh_v16 and components is not None:
        result["validation_order_nll"] = components
    validation_split = "train_inner" if inner_fold is not None else "val"
    if inner_fold is not None:
        result.update(validation_scope=validation_split, internal_fold=inner_fold)
    if theraset_bridge is not None:
        result["theraset_bridge"] = theraset_bridge
    if xbridge or theraset_bridge is not None:
        from .xbridge_data import evidence_report

        key = "xbridge_diagnostics" if xbridge else "theraset_bridge_diagnostics"
        result[key] = evidence_report(y, probabilities, xbridge_outputs or {}, source.statistics)
    export_predictions = settings.get("export_validation_predictions", False)
    if export_predictions:
        result["validation_predictions"] = export_validation(
            run, "best", source, datasets[1], y, probabilities, threshold, best_epoch, best_weights, suite,
            predicted=predicted, count_predicted=count_predicted, view_outputs=xbridge_outputs or view_outputs,
            split=validation_split)
    if save_candidates and alternate_best is not None:
        candidate = alternate_best
        alternate_path = run/f"{alternate_stem}.pt"
        torch.save({**static, "model_state": candidate["model_state"], "best_epoch": candidate["epoch"],
                    "threshold": candidate["threshold"], "weight_source": candidate["weight_source"],
                    "selection_suite": alternate_suite, "selection_key": candidate["key"]}, alternate_path)
        alternate_y, alternate_p = candidate["targets"].numpy(), candidate["probabilities"].numpy()
        result["alternate_selection"] = {
            "suite": alternate_suite, "best_epoch": candidate["epoch"], "key": candidate["key"],
            "threshold": candidate["threshold"], "threshold_fit_split": protocol_description(
                alternate_suite, protocol["threshold_strategy"])["threshold_fit_split"],
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
                candidate["epoch"], candidate["weight_source"], alternate_suite, split=validation_split)
    if ema is not None:
        restore_model_state(model, online_best_state)
        raw_y, raw_p, raw_loss = predict_loader(model, loaders[1], device, precision, config["loss"])
        raw_threshold = final_threshold(raw_y, raw_p, suite, settings.get("threshold_strategy", "global"), fixed_threshold)
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
    if source.data.track in (*THERAPEP_TRACKS, "rich48"):
        from .therapep import published_metrics
        result["therapep_validation"] = published_metrics(y, probabilities, threshold)
    result["elapsed_seconds"] = elapsed_before+time.perf_counter()-started
    write_json(run/"results.json.tmp", result)
    (run/"results.json.tmp").replace(run/"results.json")
    if pb and model.stage == "C":
        from .models.privilegedbridge import export_deployment
        export_deployment(torch.load(run/"best.pt", map_location="cpu", weights_only=True), run/"deployment.pt")
    return result


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(checkpoint["model_name"], checkpoint["model_args"]).to(device)
    restore_model_state(model, checkpoint["model_state"])
    if checkpoint["model_name"] == HH_V17 and checkpoint["neighbor_reference"] is not None:
        model.neighbor_vectors, model.neighbor_labels = (v.to(device) for v in checkpoint["neighbor_reference"])
    return model.eval(), checkpoint


@torch.inference_mode()
def hh_validation_outputs(model, loader, device):
    items = {}
    for batch in loader:
        out = model(to_device(batch, device))
        unary, logits, psi = (out[key] for key in ("unary", "logits", "psi"))
        local = model.decode(unary, logits, psi)
        exact = model.decode(unary, logits, psi, exact=True)
        decisions = torch.stack((unary >= 0, logits >= 0, local, exact), dim=1)
        sets, unique = model.candidates(unary, logits)
        targets = batch["y"].to(device).bool()
        overlap = (sets & targets[:, None]).sum(-1)
        union = (sets | targets[:, None]).sum(-1)
        oracle = (overlap / union.clamp_min(1)).masked_fill(~unique, 0).amax(-1)
        energy = model.set_energy(decisions, unary, psi)
        batch_items = {"targets": batch["y"], "probabilities": logits.sigmoid(),
                       "unary_logits": unary, "conditional_logits": logits, "pair_potentials": psi,
                       "decisions": decisions, "raw_axes": decisions.float() @ model.projection > 0,
                       "effective_axes": model.axes(decisions), "energy": energy, "local_oracle_jaccard": oracle}
        for key, values in batch_items.items():
            items.setdefault(key, []).append(values.cpu().numpy())
    arrays = {key: np.concatenate(values) for key, values in items.items()}
    return arrays, {"pair_indices": list(zip(model.pair_left.tolist(), model.pair_right.tolist()))}


def hh_validation_report(y, probabilities, arrays, scores, statistics):
    actual = y.astype(bool)
    unary, conditional, local, exact = arrays["decisions"].transpose(1, 0, 2)
    unary_probabilities = torch.from_numpy(arrays["unary_logits"]).sigmoid().numpy()
    def report(p, prediction):
        return evaluate_metrics(y, p, .5, statistics["labels"]["support"],
                                statistics["function_projection"], predicted=prediction)
    added, deleted = conditional & ~unary, unary & ~conditional
    delta = np.abs(arrays["conditional_logits"]-arrays["unary_logits"])
    magnitude = np.abs(arrays["unary_logits"])
    quantiles = lambda values: np.quantile(values, [.5, .9, .99]).tolist() if values.size else None
    from .multimodal import independent_functions
    multifunction = independent_functions(y, statistics["function_projection"]).sum(-1) >= 2
    def counts(mask, prediction):
        truth, predicted = actual[mask], prediction[mask]
        return {"tp": int((truth & predicted).sum()), "fp": int((~truth & predicted).sum()),
                "fn": int((truth & ~predicted).sum())}
    return {**scores, "decision_order": ["unary", "conditional", "local", "exact"],
            "unary": report(unary_probabilities, unary), "conditional": report(probabilities, conditional),
            "local": report(probabilities, local), "exact": report(probabilities, exact),
            "mean_energy_regret": float(np.maximum(arrays["energy"][:, 3]-arrays["energy"][:, 2], 0).mean()),
            "mean_local_oracle_jaccard": float(arrays["local_oracle_jaccard"].mean()),
            "added": int(added.sum()), "added_recorded": int((added & actual).sum()),
            "deleted": int(deleted.sum()), "deleted_recorded": int((deleted & actual).sum()),
            "quantiles": [0.5, 0.9, 0.99], "unary_abs": quantiles(magnitude), "correction_abs": quantiles(delta),
            "relative_correction": quantiles(delta/(magnitude+1e-6)),
            "flipped_unary_abs": quantiles(magnitude[added | deleted]),
            "by_observed_function_count": {
                name: {"unary": counts(mask, unary), "conditional": counts(mask, conditional)}
                for name, mask in (("single_or_empty", ~multifunction), ("multiple", multifunction))}}


@torch.inference_mode()
def hh_v12_validation_outputs(model, loader, device):
    items = {key: [] for key in ("targets", "unary_logits", "unconditional_logits", "conditional_logits")}
    for batch in loader:
        output = model(to_device(batch, device), return_unconditional=True)
        for key, values in (("targets", batch["y"]), ("unary_logits", output["unary"]),
                            ("unconditional_logits", output["unconditional_logits"]),
                            ("conditional_logits", output["logits"])):
            items[key].append(values.cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in items.items()}


def hh_v12_validation_report(y, arrays, statistics):
    from .multimodal import independent_functions
    truth = y.astype(bool)
    projection = statistics["function_projection"]
    truth_axes = independent_functions(y, projection)
    reports, predictions, hits = {}, {}, {}
    for name in ("unary", "unconditional", "conditional"):
        probabilities = torch.from_numpy(arrays[f"{name}_logits"]).sigmoid().numpy()
        predictions[name] = probabilities >= .5
        hits[name] = (independent_functions(predictions[name], projection) & truth_axes).sum(1)
        reports[name] = metrics(y, probabilities, statistics)
    transitions = {}
    for start, end in (("unary", "unconditional"), ("unconditional", "conditional"), ("unary", "conditional")):
        before, after = predictions[start], predictions[end]
        added, deleted = after & ~before, before & ~after
        multi = truth_axes.sum(1) >= 2
        one_hit = multi & (hits[start] == 1)
        transitions[f"{start}_to_{end}"] = {
            "changed_rows": int((before != after).any(1).sum()),
            "added_recorded": int((added & truth).sum()),
            "added_unrecorded": int((added & ~truth).sum()),
            "deleted_recorded": int((deleted & truth).sum()),
            "deleted_unrecorded": int((deleted & ~truth).sum()),
            "one_correct_axis_rows": int(one_hit.sum()),
            "one_to_at_least_two_correct_axes": int((one_hit & (hits[end] >= 2)).sum()),
            "at_least_two_to_fewer_correct_axes": int(((hits[start] >= 2) & (hits[end] < 2)).sum()),
        }
    return {**reports, "transitions": transitions}


def evaluate(checkpoint_path, data_root, feature_root, split, device, output, predictions=None, batch_size=32, decision_path=None,
             without_private_evidence=False, peptide_feature_root=None, hh_diagnostics=False, v5_forward=None,
             v6_diagnostics=False):
    model, checkpoint = load_model(checkpoint_path, device)
    if v6_diagnostics and (checkpoint["model_name"] != PEPTIDE_V6 or model.mode != "interaction"
                           or split != "val" or decision_path):
        raise ValueError("V6 diagnostics require a V6 interaction checkpoint, Validation and the saved threshold")
    joint = checkpoint["model_name"] == HH_V16
    if joint and (decision_path or (hh_diagnostics and split != "val")):
        raise ValueError("HH V1.6 uses its joint expected-Jaccard decoder; auxiliary diagnostics use Validation")
    if checkpoint["model_name"] in ("hh_setreread_v13", "hh_setreread_v14", HH_V17):
        if decision_path or without_private_evidence or (hh_diagnostics and split != "val"):
            raise ValueError("HH uses the saved edit rule; background diagnostics use Validation")
        if checkpoint["model_name"] == HH_V17:
            from .hh_v17 import evaluate_editor
        elif checkpoint["model_name"] == "hh_setreread_v14":
            from .hh_v14 import evaluate_editor
        else:
            from .hh_v13 import evaluate_editor
        return evaluate_editor(model, checkpoint, data_root, feature_root, split, device, output, predictions,
                               batch_size, peptide_feature_root, hh_diagnostics)
    if without_private_evidence:
        if checkpoint["model_name"] != "trirelay" or not hasattr(model, "evidence_partition"):
            raise ValueError("Private evidence ablation requires a shared-private TriRelay checkpoint")
        model.private_evidence_enabled = False
    setreread = checkpoint["model_name"] == "setreread"
    hh_v20 = checkpoint["model_name"] == HH_V20
    sequence_only = setreread or checkpoint["model_name"] in SEQUENCE_ONLY_MODELS
    source = MultimodalData(data_root, checkpoint["track"], checkpoint["statistics"],
                           sequence_only=sequence_only, sequence_chemistry=setreread or hh_v20,
                           localize_terminals=checkpoint["statistics"].get("localized_terminal_chemistry", False))
    if checkpoint["model_name"] == PEPTIDE_V3:
        dataset = BackbonePeptides(source, split, checkpoint["encoder"]["max_length"])
    else:
        dataset = MultimodalPeptides(source, feature_root, split,
                                      peptide_feature_root=peptide_feature_root or checkpoint["config"]["data"].get("peptide_features"),
                                      sequence_only=sequence_only or (checkpoint["model_name"] == "privilegedbridge" and model.stage == "A"))
        validate_cache(dataset, checkpoint["config"], checkpoint["encoder"])
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_multimodal)
    decision, predicted = None, None
    residue_nll = {} if getattr(model, "order_evidence", None) is not None else None
    if joint:
        joint_outputs = predict_loader(model, loader, device, decode_sets=True, include_auxiliary=hh_diagnostics)
        y, probabilities, predicted, count_predicted, components = joint_outputs[:5]
        extra_outputs = {"predicted": predicted, "count_predicted": count_predicted,
                         "prediction_rule": "joint_expected_jaccard"}
        if hh_diagnostics:
            auxiliary = joint_outputs[-1]
            extra_outputs.update(auxiliary_probabilities=auxiliary, cluster_id=source.data.clusters[dataset.indices])
    elif hh_diagnostics:
        if checkpoint["model_name"] not in HH_MODELS or split != "val":
            raise ValueError("HH diagnostics require an HH-SetReread checkpoint and --split val")
        if checkpoint["model_name"] == "hh_setreread_v12":
            extra_outputs = hh_v12_validation_outputs(model, loader, device)
            y = extra_outputs.pop("targets")
            probabilities = torch.from_numpy(extra_outputs["conditional_logits"]).sigmoid().numpy()
        else:
            extra_outputs, scores = hh_validation_outputs(model, loader, device)
            y, probabilities = extra_outputs.pop("targets"), extra_outputs.pop("probabilities")
    else:
        extra_outputs = {}
        view_outputs = extra_outputs if predictions and getattr(model, "v5_readout", None) is not None else None
        v7_outputs = (extra_outputs if checkpoint["model_name"] == PEPTIDE_V7
                      and model.head == "structured" and split == "val" else None)
        y, probabilities = predict_loader(model, loader, device, residue_nll=residue_nll,
                                          view_outputs=view_outputs, v5_forward=v5_forward,
                                          v6_outputs=extra_outputs if v6_diagnostics else None,
                                          v7_outputs=v7_outputs,
                                          xbridge_outputs=extra_outputs if checkpoint["model_name"] == XBRIDGE else None)
    threshold = checkpoint["threshold"]
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
    if residue_nll is not None:
        result["order_nll"] = residue_nll
    if checkpoint["model_name"] in (XBRIDGE, THERASET_BRIDGE):
        from .xbridge_data import evidence_report

        key = "xbridge_diagnostics" if checkpoint["model_name"] == XBRIDGE else "theraset_bridge_diagnostics"
        result[key] = evidence_report(y, probabilities, extra_outputs, source.statistics)
        if checkpoint["model_name"] == THERASET_BRIDGE:
            result["theraset_bridge"] = checkpoint["config"]["loss"]["theraset_bridge"]
        extra_outputs["cluster_id"] = source.data.clusters[dataset.indices]
    if v5_forward is not None:
        result["v5_inference_ablation"] = v5_forward
    if v6_diagnostics:
        from .models.peptide_v6 import DIAGNOSTIC_NAMES
        support = source.statistics["labels"]["support"]
        projection = source.statistics["function_projection"]
        result["v6_diagnostics"] = {
            "mean_evidence_weights": dict(zip(extra_outputs["evidence_names"].tolist(),
                                              extra_outputs["evidence_weights"].mean(axis=(0, 2)).tolist())),
            **{name: evaluate_metrics(y, extra_outputs[f"{name}_probabilities"], threshold, support, projection)
               for name in DIAGNOSTIC_NAMES},
        }
    if checkpoint["model_name"] == PEPTIDE_V7:
        result["v7_head"] = model.head
        if split == "val":
            fallback = probabilities >= .5
            empty = np.flatnonzero(~fallback.any(axis=1))
            fallback[empty, probabilities[empty].argmax(axis=1)] = True
            diagnostics = {"top1_fallback_rows": len(empty),
                           "top1_fallback_metrics": metrics(y, probabilities, source.statistics, .5, fallback)}
            if model.head == "structured":
                from .models.peptide_v7 import axis_observations
                state, conflicts = axis_observations(torch.from_numpy(y), model.axes)
                code = 3 * state.numpy() + extra_outputs["state_probabilities"].argmax(axis=-1)
                diagnostics["state_names"] = extra_outputs["state_names"].tolist()
                diagnostics["axis_state_confusion"] = {
                    name: np.bincount(code[~conflicts[:, i].numpy(), i], minlength=9).reshape(3, 3).tolist()
                    for i, name in enumerate(extra_outputs["axis_names"].tolist())
                }
            result["v7_diagnostics"] = diagnostics
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
    if checkpoint["track"] in (*THERAPEP_TRACKS, "rich48"):
        from .therapep import published_metrics
        result["therapep"] = published_metrics(y, probabilities, threshold)
    settings = checkpoint["config"]["train"]
    suite = checkpoint.get("selection_suite", settings.get("metric_suite", "task_specific"))
    result["metric_protocol"] = protocol_description(
        suite, settings.get("threshold_strategy", "global"), settings.get("fixed_threshold", .5))
    if joint:
        result["metric_protocol"] = joint_set_protocol()
        result["joint_components"] = components
        result["marginal_fixed_threshold"] = metrics(y, probabilities, source.statistics, .5)
        result["marginal_top_count"] = metrics(y, probabilities, source.statistics, .5, count_predicted)
    result["primary_metrics"] = primary_metrics(result["metrics"], suite)
    if joint and hh_diagnostics:
        auxiliary_report = metrics(y, auxiliary, source.statistics, .5)
        result["hh_diagnostics"] = {
            "auxiliary_threshold": .5,
            "auxiliary": auxiliary_report,
            "joint_minus_auxiliary_jaccard_pp": 100. * (
                result["metrics"]["sample_jaccard"] - auxiliary_report["sample_jaccard"]),
        }
    elif hh_diagnostics:
        result["hh_diagnostics"] = (hh_v12_validation_report(y, extra_outputs, source.statistics)
                                    if checkpoint["model_name"] == "hh_setreread_v12" else
                                    hh_validation_report(y, probabilities, extra_outputs, scores, source.statistics))
        if checkpoint["model_name"] == "hh_setreread_v12":
            extra_outputs["cluster_id"] = source.data.clusters[dataset.indices]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    if predictions:
        Path(predictions).parent.mkdir(parents=True, exist_ok=True)
        with Path(predictions).open("xb") as f:
            np.savez_compressed(f, row_id=source.data.rows[dataset.indices], ID=source.data.ids[dataset.indices],
                                targets=y, probabilities=probabilities, label_names=np.array(source.data.labels), threshold=threshold,
                                split=split, task=checkpoint['track'], decision_json=json.dumps(decision),
                                inference_ablation_json=json.dumps(result.get('inference_ablation')), **extra_outputs)
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
    if checkpoint["model_name"] in ("hh_setreread_v13", "hh_setreread_v14", HH_V17):
        raw.pop("base_logits", None)
    if checkpoint["model_name"] == HH_V17:
        raw.pop("neighbor_features", None)
    if checkpoint["model_name"] == HH_V15_RANKER:
        inputs = {"esm": torch.from_numpy(raw["esm"]).float(),
                  "esm_peptide": torch.from_numpy(raw["esm_peptide"]).float(),
                  "mask": torch.from_numpy(raw["mask"]).bool()}
    elif checkpoint["model_name"] == PEPTIDE_V3:
        inputs = {"aa": torch.from_numpy(raw["aa"]).long(), "mask": torch.from_numpy(raw["mask"]).bool()}
    elif checkpoint["model_name"] in (PEPTIDE_V6, PEPTIDE_V7):
        inputs = {"aa": torch.from_numpy(raw["aa"]).long(), "mask": torch.from_numpy(raw["mask"]).bool(),
                  "esm": torch.from_numpy(raw["esm"]).float(),
                  "esm_peptide": torch.from_numpy(raw["esm_peptide"]).float()}
    elif checkpoint["model_name"] in ("setreread", HH_V20) and "chemical" not in raw:
        from .multimodal import prepare_sequence_inputs
        inputs = prepare_sequence_inputs(raw, checkpoint["statistics"])
    elif "phys_raw" in raw:
        from .multimodal import prepare_inputs
        inputs = prepare_inputs(raw, checkpoint["statistics"])
    else:
        inputs = {k: torch.from_numpy(v) for k, v in raw.items()}
    records = []
    scope = "sequence_only" if checkpoint["model_name"] in (*SEQUENCE_ONLY_MODELS, "privilegedbridge") else "provided_multimodal_fields"
    if checkpoint["model_name"] == "privilegedbridge" and model.stage == "B":
        scope = "provided_multimodal_fields"
    with torch.inference_mode():
        for start in range(0, len(inputs["mask"]), batch_size):
            batch = {k: v[start:start+batch_size].to(device) for k, v in inputs.items()}
            records.extend(model_prediction_records(model, checkpoint, batch, scope, decision_path))
    return records


def model_prediction_records(model, checkpoint, batch, scope, decision_path):
    output = model(batch)
    if checkpoint["model_name"] == HH_V16:
        if decision_path:
            raise ValueError("HH V1.6 uses its joint expected-Jaccard decoder")
        marginal, predicted, _ = model.decode(output)
        return prediction_records(checkpoint, marginal.cpu().tolist(), scope, predicted=predicted.cpu().numpy())
    if checkpoint["model_name"] == HH_V17:
        if decision_path:
            raise ValueError("HH V1.7 uses its saved reference-relative decision")
        from .models.hh_setreread_v17 import decisions
        base = output["base_logits"] >= 0
        final, _, added, removed = decisions(base, model.proposals(output), checkpoint["edit_threshold"],
                                               checkpoint["config"]["train"]["actions"])
        records = prediction_records(checkpoint, output["logits"].sigmoid().cpu().tolist(), scope,
                                     predicted=final.cpu().numpy())
        names = checkpoint["labels"]
        for record, old, plus, minus in zip(records, base.tolist(), added.tolist(), removed.tolist()):
            record.update(base_labels=[name for name, present in zip(names, old) if present],
                          added_label=names[plus] if plus >= 0 else None,
                          removed_label=names[minus] if minus >= 0 else None)
        return records
    probabilities = output["logits"].sigmoid().cpu().tolist()
    if checkpoint["model_name"] not in ("hh_setreread_v13", "hh_setreread_v14"):
        return prediction_records(checkpoint, probabilities, scope, decision_path)
    if decision_path:
        raise ValueError("HH predicts with its saved single-add rule")
    if checkpoint["model_name"] == "hh_setreread_v14":
        from .hh_v14 import prediction_records as second_axis_records
        return second_axis_records(checkpoint, output, probabilities, scope)
    from .models.hh_setreread_v13 import apply_edits
    final, added = apply_edits(output["logits"], output["edit_gain"], checkpoint["edit_threshold"])
    records = prediction_records(checkpoint, probabilities, scope)
    names = checkpoint["labels"]
    for item, base, chosen, label, gains in zip(records, output["logits"].cpu().tolist(),
                                                final.cpu().tolist(), added.cpu().tolist(),
                                                output["edit_gain"].cpu().tolist()):
        item["base_labels"] = [name for name, u in zip(names, base) if u >= 0]
        item["predicted_labels"] = [name for name, keep in zip(names, chosen) if keep]
        item["edit_gain"] = {name: gain for name, u, gain in zip(names, base, gains) if u < 0}
        item["added_label"] = names[label] if label >= 0 else None
    return records


def prediction_records(checkpoint, probabilities, scope, decision_path=None, predicted=None):
    if predicted is None:
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
    from .multimodal import prepare_inputs, prepare_sequence_inputs
    model, checkpoint = load_model(checkpoint_path, device)
    info = checkpoint["encoder"]
    if checkpoint["model_name"] == PEPTIDE_V3:
        if any(not s or len(s) > info["max_length"] for s in sequences):
            raise ValueError(f"V3 expects nonempty sequences of at most {info['max_length']} residues")
        result = []
        size = checkpoint["config"]["train"]["batch_size"]
        with torch.inference_mode():
            for start in range(0, len(sequences), size):
                selected = sequences[start:start+size]
                batch = collate_multimodal([backbone_inputs(s) for s in selected])
                records = model_prediction_records(model, checkpoint, to_device(batch, device), "sequence_only", decision_path)
                result.extend(dict(record, sequence=s) for record, s in zip(records, selected))
        return result
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
        prepare = (prepare_sequence_inputs if checkpoint["model_name"] in
                   ("setreread", HH_V15_RANKER, HH_V20, PEPTIDE_V6, PEPTIDE_V7) else prepare_inputs)
        inputs = prepare(raw, checkpoint["statistics"])
        scope = "sequence_only" if checkpoint["model_name"] in (*SEQUENCE_ONLY_MODELS, "privilegedbridge") else "sequence_and_stereochemistry_only"
        with torch.inference_mode():
            records = model_prediction_records(model, checkpoint, to_device(inputs, device), scope, decision_path)
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
