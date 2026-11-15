"""Train and evaluate the Fine39 reference-relative label editor."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .hh_v13 import ScoredPeptides, base_logits, datasets, evaluation_groups, load_oof, make_loader, save_predictions
from .metrics import evaluate_metrics
from .models import HH_V17
from .models.hh_setreread_v17 import ACTIONS, HHSetRereadV17, decisions
from .multimodal import independent_functions
from .neighbor_evidence import fold_neighbor_evidence
from .research import checkpoint_state, load_model, make_optimizer, to_device
from .trainer import seed_everything, write_json


def pack_neighbors(evidence):
    votes, positive, nearest = evidence
    return np.stack((votes, positive, np.broadcast_to(nearest[:, None], votes.shape)), -1).astype(np.float32)


@torch.inference_mode()
def collect(model, loader, device):
    model.eval()
    unary, probabilities, proposal, labels = [], [], [], []
    for batch in loader:
        labels.append(batch["y"])
        output = model(to_device(batch, device))
        unary.append(output["base_logits"].cpu())
        probabilities.append(output["logits"].sigmoid().cpu())
        proposal.append(tuple(value.cpu() for value in model.proposals(output)))
    return (torch.cat(unary), torch.cat(probabilities), tuple(torch.cat(values) for values in zip(*proposal)),
            torch.cat(labels).bool())


def jaccard(predicted, truth):
    return (predicted & truth).sum(-1) / (predicted | truth).sum(-1).clamp_min(1)


def fit_threshold(base, proposal, truth, actions):
    predicted, kind, _, _ = decisions(base, proposal, 0., actions)
    gain = proposal[0].gather(1, kind.clamp_min(0)[:, None]).squeeze(1).numpy()
    available = np.flatnonzero(kind.numpy() >= 0)
    if not len(available):
        return None
    order = available[np.argsort(-gain[available], kind="stable")]
    values = gain[order]
    last = np.r_[values[:-1] != values[1:], True]
    delta = (jaccard(predicted, truth) - jaccard(base, truth)).numpy()
    cumulative = np.cumsum(delta[order], dtype=np.float64)
    stops = np.flatnonzero(last)
    best = stops[cumulative[stops].argmax()]
    return (None if cumulative[best] <= 0 else
            float(values[best + 1]) if best + 1 < len(values) else 0.)


def score_outputs(outputs, statistics, threshold, actions):
    unary, probabilities, proposal, truth = outputs
    base = unary >= 0
    final, kind, added, removed = decisions(base, proposal, threshold, actions)
    y, base_np, final_np = truth.numpy(), base.numpy(), final.numpy()
    options = statistics["labels"]["support"], statistics["function_projection"]
    report = evaluate_metrics(y, probabilities.numpy(), .5, *options, predicted=final_np)
    old = evaluate_metrics(y, unary.sigmoid().numpy(), .5, *options, predicted=base_np)
    delta = (jaccard(final, truth) - jaccard(base, truth)).numpy()
    truth_axes = independent_functions(y, options[1])
    before, after = ((independent_functions(mask, options[1]) & truth_axes).sum(1) for mask in (base_np, final_np))
    matches = (base_np & y).sum(1)
    misses = base_np.any(1) & (matches == 0)
    high = misses & (unary.sigmoid().amax(1).numpy() >= .9)
    exact = (base_np == y).all(1)
    exact_now = (final_np == y).all(1)
    singleton = misses & (base_np.sum(1) == 1) & (y.sum(1) == 1)
    edits = {name: int((kind == i).sum()) for i, name in enumerate(ACTIONS)}
    edits.update(improved=int((delta > 0).sum()), harmed=int((delta < 0).sum()),
                 baseline_exact_harmed=int((exact & ~exact_now).sum()),
                 wrong_nonempty_hit=int((misses & (final_np & y).any(1)).sum()),
                 wrong_singleton_exact=int((singleton & exact_now).sum()),
                 high_confidence_wrong_exact=int((high & exact_now).sum()))
    for level in (2, 3):
        edits[f"correct_axis_{level}_recovered"] = int(((before == level - 1) & (after >= level)).sum())
        edits[f"correct_axis_{level}_lost"] = int(((before >= level) & (after < level)).sum())
    ranking = {}
    for name, probability in (("baseline", unary.sigmoid()), ("label_head", probabilities)):
        top = probability.argsort(dim=-1, descending=True, stable=True)[:, :5]
        hit = truth.gather(1, top)
        ranking[name] = {"high_confidence_wrong_top1": int((hit[:, 0].numpy() & high).sum()),
                         "high_confidence_wrong_top5": int((hit.any(1).numpy() & high).sum()),
                         "second_recorded_label_top5": int((hit.sum(1) >= 2).sum()),
                         "third_recorded_label_top5": int((hit.sum(1) >= 3).sum())}
    arrays = {"targets": y, "base_logits": unary.numpy(), "probabilities": probabilities.numpy(), "base_set": base_np,
              "final_set": final_np, "predicted": final_np, "action": kind.numpy(), "added_label": added.numpy(),
              "removed_label": removed.numpy(), "jaccard_gain": delta}
    return {"metrics": report, "baseline": old, "edits": edits, "ranking": ranking}, arrays


def threshold_for(outputs, actions):
    unary, _, proposal, truth = outputs
    return fit_threshold(unary >= 0, proposal, truth, actions)


def train_editor(config, name, files):
    settings, data = config["train"], config["data"]
    if (data["track"] != "fine39" or settings.get("precision") != "fp32"
            or settings.get("accumulation_steps", 1) != 1 or settings["metric_suite"] != "jaccard"):
        raise ValueError("HH V1.7 uses Fine39, FP32, accumulation_steps=1, and Jaccard selection")
    if not settings.get("initialize_from") or not files:
        raise ValueError("Provide --initialize-from and the three --hh-oof-files")
    started = time.perf_counter()
    seed_everything(settings["seed"])
    device = torch.device(settings["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    base, checkpoint = load_model(settings["initialize_from"], device)
    if (checkpoint["model_name"] != "theraset" or checkpoint["track"] != "fine39" or base.label_context
            or not base.evidence_decoder or not base.active.sum(1).eq(1).all() or checkpoint.get("threshold") != .5
            or checkpoint["config"]["train"].get("hh_oof_fold") is not None):
        raise ValueError("Use the full Train Fine39 TheraSet no_label_context_single_prototype at threshold 0.50")
    for key in ("encoder", "peptide_encoder"):
        config[key] = checkpoint["config"][key]
        if config[key]["max_length"] != 80:
            raise ValueError("HH V1.7 uses the length-80 ESM caches")
    limit = settings.get("limit_samples")
    diagnostic = bool(limit or settings.get("max_steps"))
    val = datasets(config, checkpoint, "val", limit=limit)
    rows, scores = load_oof(val.source.data, files, diagnostic)
    train = datasets(config, checkpoint, "train", indices=rows, limit=limit, sequence_only=True)
    scores = scores[np.searchsorted(rows, train.indices)]
    run = Path(settings["output"]) / name
    run.mkdir(parents=True, exist_ok=False)
    print(f"HH V1.7 | Train={len(train)} Validation={len(val)} | computing reference outputs", flush=True)
    val_scores = base_logits(base, make_loader(val, settings), device)
    del base
    val.sequence_only = True
    train_neighbors = val_neighbors = reference = None
    if config["model"]["neighbors"]:
        print("HH V1.7 | computing cluster-held-out Train nearest neighbors", flush=True)
        tr, va, reference = fold_neighbor_evidence(val.source.data, data["peptide_features"], train.indices,
                                                   val.indices, include_reference=True)
        train_neighbors, val_neighbors = pack_neighbors(tr), pack_neighbors(va)
    statistics = checkpoint["statistics"]
    args = {**config["model"], "function_projection": statistics["function_projection"]}
    model = HHSetRereadV17(**args).to(device)
    optimizer = make_optimizer(model, settings)
    train_loader = make_loader(ScoredPeptides(train, scores, train_neighbors), settings, shuffle=True,
                               generator=torch.Generator().manual_seed(settings["seed"]))
    val_loader = make_loader(ScoredPeptides(val, val_scores, val_neighbors), settings)
    settings["oof_files"] = list(files)
    actions = settings["actions"]
    write_json(run / "metadata.json", {"model": HH_V17, "task": "fine39", "seed": settings["seed"], "config": config,
               "labels": val.source.data.labels, "encoder": checkpoint["encoder"], "dataset": statistics["dataset"],
               "cohort": statistics["cohort"], "diagnostic_subset": diagnostic, "device": str(device),
               "torch": str(torch.__version__), "git_commit": subprocess.run(
                   ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()})
    best_key, best_state, stale, steps = None, None, 0, 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total, count = 0., 0
        for batch in train_loader:
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch, model(batch), config["loss"]["coexistence_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch["y"])
            count += len(batch["y"])
            steps += 1
            if settings.get("max_steps") and steps >= settings["max_steps"]:
                break
        outputs = collect(model, val_loader, device)
        threshold = threshold_for(outputs, actions)
        report, _ = score_outputs(outputs, statistics, threshold, actions)
        metrics = report["metrics"]
        key = metrics["sample_jaccard"], metrics["macro_ap"]
        if best_key is None or key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state, best_outputs, best_threshold = checkpoint_state(model, cpu=True), outputs, threshold
        else:
            stale += 1
        with (run / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "loss": total / count, "edit_threshold": threshold,
                                     "jaccard": key[0], "macro_ap": key[1], "edits": report["edits"]},
                                    allow_nan=False) + "\n")
        print(f"HH V1.7 | epoch {epoch}/{settings['epochs']} | loss={total/count:.5f} | "
              f"Val Jaccard={key[0]:.5f} MacroAP={key[1]:.5f} | best={best_epoch}", flush=True)
        if ((settings["patience"] and stale >= settings["patience"])
                or (settings.get("max_steps") and steps >= settings["max_steps"])):
            break
    state = {**best_state, **{f"backbone.{k}": v for k, v in checkpoint["model_state"].items()}}
    torch.save({**checkpoint, "model_name": HH_V17, "config": config, "model_state": state,
                "labels": val.source.data.labels, "selection_suite": "jaccard",
                "model_args": dict(args, backbone_args=checkpoint["model_args"]), "neighbor_reference": reference,
                "threshold": .5, "edit_threshold": best_threshold, "best_epoch": best_epoch,
                "diagnostic_subset": diagnostic}, run / "best.pt")
    report, arrays = score_outputs(best_outputs, statistics, best_threshold, actions)
    save_predictions(run / "best_validation.npz", val, arrays, best_threshold, "val")
    result = {"model": HH_V17, "task": "fine39", "seed": settings["seed"], "threshold": .5,
              "edit_threshold": best_threshold, "threshold_strategy": "validation_reference_delta_jaccard",
              "best_epoch": best_epoch, "epochs_completed": epoch, "global_step": steps,
              "validation": report.pop("metrics"), **report, "diagnostic_subset": diagnostic,
              "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "elapsed_seconds": time.perf_counter() - started, "checkpoint": str(run / "best.pt")}
    write_json(run / "results.json", result)
    print(f"Saved {run / 'best.pt'}", flush=True)
    return result


def evaluate_editor(model, checkpoint, data_root, feature_root, split, device, output, predictions,
                    batch_size, peptide_feature_root=None, diagnostics=False):
    config = checkpoint["config"]
    if split == "test" and checkpoint.get("diagnostic_subset"):
        raise ValueError("Diagnostic checkpoints only support Validation")
    config["data"].update(root=data_root, features=feature_root,
                          peptide_features=peptide_feature_root or config["data"]["peptide_features"])
    dataset = datasets(config, checkpoint, split, limit=config["train"].get("limit_samples"))
    outputs = collect(model, make_loader(dataset, {"batch_size": batch_size, "num_workers": 0}), device)
    threshold, actions = checkpoint["edit_threshold"], config["train"]["actions"]
    report, arrays = score_outputs(outputs, checkpoint["statistics"], threshold, actions)
    result = {"model": HH_V17, "task": "fine39", "split": split, "rows": len(dataset), "edit_threshold": threshold,
              **report, "groups": evaluation_groups(dataset.source, dataset.indices, arrays)}
    if diagnostics:
        result["actions_at_saved_threshold"] = {
            action: score_outputs(outputs, checkpoint["statistics"], threshold, action)[0]["metrics"]["sample_jaccard"]
            for action in (*ACTIONS, "all")}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    if predictions:
        save_predictions(Path(predictions), dataset, arrays, threshold, split)
    return result
