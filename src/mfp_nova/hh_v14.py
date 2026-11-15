"""Fit second-axis selection and label ranking on held-out-cluster scores."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F
from torch.utils.data import Subset

from .hh_v13 import ScoredPeptides, base_logits, datasets, evaluation_groups, load_oof, make_loader, save_predictions
from .metrics import evaluate_metrics
from .models.hh_setreread_v14 import HHSetRereadV14, apply_second_axis
from .multimodal import independent_functions
from .research import checkpoint_state, load_model, make_optimizer, to_device
from .trainer import seed_everything, write_json


NAME = "hh_setreread_v14"


def supervision(model, unary, y):
    keys = ("candidate_mask", "exact_target", "axis_target", "anchor_correct", "recorded_multifunction")
    device = model.membership.device
    chunks = [tuple(x.cpu() for x in model.supervision(u.to(device), target.to(device)))
              for u, target in zip(torch.from_numpy(unary).split(512), torch.from_numpy(y).split(512))]
    return {key: torch.cat(parts).numpy() for key, parts in zip(keys, zip(*chunks))}


@torch.inference_mode()
def read_scores(model, loader, device, branch=None, without_background=False):
    model.eval()
    values = {}
    for batch in loader:
        output = model(to_device(batch, device), branch=branch, without_background=without_background)
        for key, tensor in output.items():
            values.setdefault(key, []).append(tensor.cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in values.items()}


def ranking_stats(scores, target, branch):
    mask, exact = target["candidate_mask"], target["exact_target"]
    eligible = exact.any(-1)
    if branch == "row":
        kept = mask.any(-1)
        labels = eligible[kept]
        logits = scores[kept]
        pr = float(average_precision_score(labels, logits)) if labels.any() else None
        loss = (np.logaddexp(0., logits)-labels*logits).mean() if kept.any() else None
        p = np.exp(-np.logaddexp(0., -logits))
        return {"average_precision": pr, "loss": float(loss) if loss is not None else None,
                "brier": float(((p-labels)**2).mean()) if len(p) else None,
                "eligible_rows": int(kept.sum()), "positive_rows": int(labels.sum())}
    logits, labels, valid = scores[eligible], exact[eligible], mask[eligible]
    order = np.argsort(-np.where(valid, logits, -np.inf), axis=1, kind="stable")
    hits = np.take_along_axis(labels, order[:, :3], axis=1)
    loss = ((np.logaddexp(0., logits)-labels*logits)*valid).sum(1)/valid.sum(1)
    return {"top1": float(hits[:, 0].mean()) if len(hits) else None,
            "top3": float(hits.any(1).mean()) if len(hits) else None,
            "loss": float(loss.mean()) if len(hits) else None, "positive_rows": len(hits)}


def fit_head(model, branch, train, val_loader, targets, settings, device, run):
    head = model.row_head if branch == "row" else model.label_head
    optimizer = make_optimizer(head, settings)
    generator = torch.Generator().manual_seed(settings["seed"])
    loader = make_loader(train, settings, shuffle=True, generator=generator)
    best, best_state, stale, steps = None, None, 0, 0
    for epoch in range(1, settings["epochs"]+1):
        model.train()
        total, count = 0., 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.branch_loss(to_device(batch, device), branch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), settings["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch["y"])
            count += len(batch["y"])
            steps += 1
            if settings.get("max_steps") and steps >= settings["max_steps"]:
                break
        scores = read_scores(model, val_loader, device, branch)[f"{branch}_logit"]
        stats = ranking_stats(scores, targets, branch)
        score = stats["average_precision" if branch == "row" else "top1"]
        loss = stats["loss"]
        key = (score if score is not None else -1., -(loss if loss is not None else total/count))
        if best is None or key > best:
            best = key
            best_state = checkpoint_state(head, cpu=True)
            best_epoch, stale = epoch, 0
        else:
            stale += 1
        with (run/"history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"branch": branch, "epoch": epoch, "train_loss": total/count,
                                     "validation": stats}, allow_nan=False)+"\n")
        print(f"HH V1.4 {branch} | epoch {epoch} | loss={total/count:.6f} | "
              f"Val score={score} | best={best_epoch}", flush=True)
        if ((settings["patience"] and stale >= settings["patience"])
                or (settings.get("max_steps") and steps >= settings["max_steps"])):
            break
    head.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "epochs_completed": epoch, "steps": steps}


def base_candidate_scores(unary, candidates):
    return np.where(candidates, unary, -np.inf).max(-1)


def choose_threshold(unary, y, targets, row_scores, label_scores):
    """Scan tied score blocks using exact set changes, reusing the same predictions."""
    y = y.astype(bool)
    mask, exact = targets["candidate_mask"], targets["exact_target"]
    base = unary >= 0
    labels = np.where(mask, label_scores, -np.inf).argmax(-1)
    rows = np.flatnonzero(mask.any(-1))
    order = rows[np.argsort(-row_scores[rows], kind="stable")]
    ordered_scores = row_scores[order]
    ends = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1])+1, len(order)]
    hit = y[order, labels[order]].astype(bool)
    axis_hit = targets["axis_target"][order, labels[order]]
    exact_hit = exact[order, labels[order]]
    new_mfp_hit = targets["recorded_multifunction"][order]
    hits = (base & y).sum(-1)
    union = (base | y).sum(-1)
    before = hits/np.maximum(1, union)
    after = (hits[order]+hit)/np.maximum(1, union[order]+~hit)
    cum = lambda v: np.r_[0, np.cumsum(v)]
    fine_tp, fine_total = int(hits.sum()), int(y.sum()+base.sum())
    # All eligible rows start with one predicted axis; each addition makes two.
    base_mfp = targets["base_multifunction"]
    true_mfp = targets["recorded_multifunction"]
    mfp_tp, mfp_total = int((base_mfp & true_mfp).sum()), int(base_mfp.sum()+true_mfp.sum())
    counts = np.unique(ends)
    fine = cum(hit)[counts]
    mfp = cum(new_mfp_hit)[counts]
    strict = cum(exact_hit)[counts]
    axis = cum(axis_hit)[counts]
    useful = cum(exact.any(-1)[order])[counts]
    anchors = cum(targets["anchor_correct"][order])[counts]
    jaccard = before.mean() + cum(after-before[order])[counts]/len(y)
    f1 = 2*(fine_tp+fine)/np.maximum(1, fine_total+counts)
    mfp_f1 = 2*(mfp_tp+mfp)/np.maximum(1, mfp_total+counts)

    def point(i):
        n = counts[i]
        threshold = (None if n == 0 else float(ordered_scores[n]) if n < len(order)
                     else float(np.nextafter(np.float32(ordered_scores[-1]), np.float32(-np.inf))))
        return {"edit_threshold": threshold, "edits": int(n), "added_recorded": int(fine[i]),
                "added_unrecorded": int(n-fine[i]), "new_multifunction_false_positives": int(n-mfp[i]),
                "strict_second_axis_recoveries": int(strict[i]), "second_axis_recoveries": int(axis[i]),
                "selected_eligible_rows": int(useful[i]), "selected_correct_anchors": int(anchors[i]),
                "sample_jaccard": float(jaccard[i]), "micro_f1": float(f1[i]), "mfp_detection_f1": float(mfp_f1[i])}

    admissible = np.flatnonzero((f1 >= f1[0]) & (mfp_f1 >= mfp_f1[0]))
    key = lambda i: (strict[i], jaccard[i], -counts[i])
    best = max(admissible, key=key)
    edited = max(range(1, len(counts)), key=lambda i: (jaccard[i], strict[i], -counts[i]), default=None)
    budgets = {str(b): point(np.flatnonzero(counts-mfp <= b)[-1]) for b in (25, 50, 100)}
    return {"selected": point(best), "best_active": point(edited) if edited is not None else None,
            "at_false_positive_budget": budgets}


def report_predictions(arrays, labels, statistics, threshold):
    unary, candidates, row, label = (arrays[key] for key in ("logits", "candidate_mask", "row_logit", "label_logit"))
    tensors = tuple(torch.from_numpy(values) for values in (unary, candidates, row, label))
    final, added = apply_second_axis(*tensors, threshold)
    base = unary >= 0
    final, added = final.numpy(), added.numpy()
    y = labels.astype(bool)
    probabilities = torch.from_numpy(unary).sigmoid().numpy()
    projection, support = statistics["function_projection"], statistics["labels"]["support"]

    def metrics(predicted):
        return evaluate_metrics(y, probabilities, .5, support, projection, predicted=predicted)

    def row_jaccard(predicted):
        return (predicted & y).sum(-1)/np.maximum(1, (predicted | y).sum(-1))

    actual = independent_functions(y, projection)
    before, after = (independent_functions(p, projection) for p in (base, final))
    old_hits, new_hits = ((p & actual).sum(-1) for p in (before, after))
    added_mask = final & ~base
    edits = {"edits": int((added >= 0).sum()), "added_recorded": int((added_mask & y).sum()),
             "added_unrecorded": int((added_mask & ~y).sum()),
             "strict_second_axis_recoveries": int(((old_hits == 1) & (new_hits == 2) & (added_mask & y).any(-1)).sum()),
             "second_axis_recoveries": int(((old_hits == 1) & (new_hits == 2)).sum()),
             "new_multifunction_false_positives": int(((added >= 0) & (actual.sum(-1) < 2)).sum()),
             "original_axes_preserved_rows": int((~(before & ~after).any(-1)).sum())}
    empty_fill = base.copy()
    empty = np.flatnonzero(~base.any(-1))
    empty_fill[empty, unary[empty].argmax(-1)] = True
    return {"metrics": metrics(final), "baseline": metrics(base), "edits": edits,
            "empty_fill_control": metrics(empty_fill)}, {
                "targets": labels, "base_logits": unary, "probabilities": probabilities,
                "row_logit": row, "label_logit": label, "candidate_mask": candidates,
                "base_set": base, "final_set": final, "added_label": added,
                "jaccard_gain": row_jaccard(final)-row_jaccard(base)}


def controls(arrays, labels, targets, without_background=None):
    unary, row, label = (arrays[k] for k in ("logits", "row_logit", "label_logit"))
    raw = base_candidate_scores(unary, targets["candidate_mask"])
    modes = {"raw_row_raw_label": (raw, unary), "learned_row_raw_label": (row, unary),
             "raw_row_model_label": (raw, label), "learned_row_model_label": (row, label)}
    max_label = base_candidate_scores(label, targets["candidate_mask"])
    log_product = -np.logaddexp(0., -row)-np.logaddexp(0., -max_label)
    modes["combined_row_model_label"] = log_product, label
    if without_background is not None:
        modes["learned_row_without_background"] = row, without_background
    return {name: choose_threshold(unary, labels, targets, g, r) for name, (g, r) in modes.items()}


def cohort_summary(y, unary, target):
    scope = target["candidate_mask"].any(-1)
    return {"rows": len(y), "single_axis_candidate_rows": int(scope.sum()),
            "correct_anchor_rows": int((scope & target["anchor_correct"]).sum()),
            "strictly_recoverable_rows": int(target["exact_target"].any(-1).sum()),
            "raw_candidate_ranking": ranking_stats(unary, target, "label")}


def train_editor(config, name, files):
    settings = config["train"]
    if (config["data"]["track"] != "fine39" or settings.get("precision") != "fp32"
            or settings.get("accumulation_steps", 1) != 1):
        raise ValueError("HH V1.4 trains on Fine39 with FP32 and one batch per step")
    if not settings.get("initialize_from") or not files:
        raise ValueError("Provide --initialize-from and --hh-oof-files for HH V1.4")
    device = torch.device(settings["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    seed_everything(settings["seed"])
    base, checkpoint = load_model(settings["initialize_from"], device)
    if (checkpoint["model_name"] != "theraset" or checkpoint["track"] != "fine39" or base.label_context
            or not base.evidence_decoder or not base.active.sum(1).eq(1).all()
            or checkpoint["config"]["train"].get("hh_oof_fold") is not None):
        raise ValueError("Use the full Train Fine39 TheraSet no_label_context_single_prototype checkpoint")
    config["encoder"] = checkpoint["config"]["encoder"]
    config["peptide_encoder"] = checkpoint["config"]["peptide_encoder"]
    if any(config[key]["max_length"] != 80 for key in ("encoder", "peptide_encoder")):
        raise ValueError("HH V1.4 requires length-80 ESM caches")
    limit = settings.get("limit_samples")
    diagnostic = bool(limit or settings.get("max_steps"))
    val = datasets(config, checkpoint, "val", limit=limit)
    rows, scores = load_oof(val.source.data, files, diagnostic)
    train = datasets(config, checkpoint, "train", indices=rows, limit=limit, sequence_only=True)
    scores = scores[np.searchsorted(rows, train.indices)]
    val_scores = base_logits(base, make_loader(val, settings), device)
    del base
    val.sequence_only = True
    y_train, y_val = (ds.source.data.y[ds.indices] for ds in (train, val))
    args = {"statistics": checkpoint["statistics"]["labels"], **config["model"]}
    model = HHSetRereadV14(**args)
    train_targets, val_targets = (supervision(model, u, y) for u, y in ((scores, y_train), (val_scores, y_val)))
    row_rows = np.flatnonzero(train_targets["candidate_mask"].any(-1))
    label_rows = np.flatnonzero(train_targets["exact_target"].any(-1))
    if not len(label_rows) or len(label_rows) == len(row_rows):
        raise ValueError("The selected OOF training rows need both recoverable and other single-axis examples")
    model.to(device)
    run = Path(settings["output"])/name
    run.mkdir(parents=True, exist_ok=False)
    settings["oof_files"] = list(files)
    statistics = checkpoint["statistics"]
    cohorts = {"train_oof": cohort_summary(y_train, scores, train_targets),
               "val": cohort_summary(y_val, val_scores, val_targets)}
    write_json(run/"metadata.json", {"model": NAME, "task": "fine39", "seed": settings["seed"], "config": config,
               "labels": val.source.data.labels, "encoder": checkpoint["encoder"], "dataset": statistics["dataset"],
               "cohort": statistics["cohort"], "diagnostic_subset": diagnostic, "candidate_cohorts": cohorts,
               "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               "device": str(device), "torch": str(torch.__version__)})
    train_scored = ScoredPeptides(train, scores)
    val_loader = make_loader(ScoredPeptides(val, val_scores), settings)
    branches = {"row": fit_head(model, "row", Subset(train_scored, row_rows.tolist()), val_loader,
                                val_targets, settings, device, run)}
    if model.label_head is not None:
        branches["label"] = fit_head(model, "label", Subset(train_scored, label_rows.tolist()), val_loader,
                                     val_targets, settings, device, run)
    arrays = read_scores(model, val_loader, device)
    val_targets["base_multifunction"] = independent_functions(val_scores >= 0, statistics["function_projection"]).sum(-1) >= 2
    comparisons = controls(arrays, y_val, val_targets)
    threshold = comparisons["learned_row_model_label"]["selected"]["edit_threshold"]
    state = {**checkpoint_state(model, cpu=True), **{f"backbone.{k}": v for k, v in checkpoint["model_state"].items()}}
    torch.save({**checkpoint, "model_name": NAME, "config": config, "model_state": state,
                "model_args": dict(args, backbone_args=checkpoint["model_args"]), "labels": val.source.data.labels,
                "threshold": .5, "edit_threshold": threshold, "branch_epochs": branches,
                "diagnostic_subset": diagnostic, "selection_suite": "jaccard"}, run/"best.pt")
    report, predictions = report_predictions(arrays, y_val, statistics, threshold)
    save_predictions(run/"best_validation.npz", val, predictions, threshold, "val")
    result = {"model": NAME, "task": "fine39", "seed": settings["seed"], "threshold": .5,
              "edit_threshold": threshold, "threshold_strategy": "val_second_axis_f1_preserving",
              "branch_epochs": branches, "validation": report.pop("metrics"), **report, "candidate_cohorts": cohorts,
              "row_ranking": ranking_stats(arrays["row_logit"], val_targets, "row"),
              "label_ranking": ranking_stats(arrays["label_logit"], val_targets, "label"),
              "controls": comparisons, "validation_groups": evaluation_groups(val.source, val.indices, predictions),
              "trainable_parameters": sum(p.numel() for p in model.parameters()), "diagnostic_subset": diagnostic,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "elapsed_seconds": time.perf_counter()-started, "checkpoint": str(run/"best.pt")}
    write_json(run/"results.json", result)
    return result


def evaluate_editor(model, checkpoint, data_root, feature_root, split, device, output, predictions,
                    batch_size, peptide_feature_root=None, diagnostics=False):
    config = checkpoint["config"]
    if split == "test" and checkpoint.get("diagnostic_subset"):
        raise ValueError("Diagnostic editor checkpoints are restricted to Validation")
    config["data"].update(root=data_root, features=feature_root,
                          peptide_features=peptide_feature_root or config["data"]["peptide_features"])
    dataset = datasets(config, checkpoint, split, limit=config["train"].get("limit_samples"))
    settings = {"batch_size": batch_size, "num_workers": 0}
    arrays = read_scores(model, make_loader(dataset, settings), device)
    labels = dataset.source.data.y[dataset.indices]
    threshold = checkpoint["edit_threshold"]
    report, exported = report_predictions(arrays, labels, dataset.source.statistics, threshold)
    result = {"model": NAME, "task": "fine39", "split": split, "rows": len(dataset), "threshold": .5,
              "edit_threshold": threshold, **report, "groups": evaluation_groups(dataset.source, dataset.indices, exported)}
    if diagnostics:
        targets = supervision(model, arrays["logits"], labels)
        targets["base_multifunction"] = independent_functions(
            arrays["logits"] >= 0, dataset.source.statistics["function_projection"]).sum(-1) >= 2
        without_background = None
        if config["model"]["label_mode"] == "conditional":
            dataset.sequence_only = True
            loader = make_loader(ScoredPeptides(dataset, arrays["logits"]), settings)
            without_background = read_scores(model, loader, device, "label", without_background=True)["label_logit"]
            exported["without_background_label_logit"] = without_background
        result["controls"] = controls(arrays, labels, targets, without_background)
        result["row_ranking"] = ranking_stats(arrays["row_logit"], targets, "row")
        result["label_ranking"] = ranking_stats(arrays["label_logit"], targets, "label")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    if predictions:
        save_predictions(Path(predictions), dataset, exported, threshold, split)
    return result


def prediction_records(checkpoint, output, probabilities, scope):
    unary, candidates, row, label = (output[k] for k in ("logits", "candidate_mask", "row_logit", "label_logit"))
    final, added = apply_second_axis(unary, candidates, row, label, checkpoint["edit_threshold"])
    names = checkpoint["labels"]
    records = []
    for p, base, pred, chosen, score, mask, scores in zip(
            probabilities, (unary >= 0).cpu().tolist(), final.cpu().tolist(), added.cpu().tolist(),
            row.cpu().tolist(), candidates.cpu().tolist(), label.cpu().tolist()):
        records.append({"probabilities": dict(zip(names, p)), "evidence_scope": scope,
                        "base_labels": [label for label, keep in zip(names, base) if keep],
                        "predicted_labels": [label for label, keep in zip(names, pred) if keep],
                        "added_label": names[chosen] if chosen >= 0 else None,
                        "row_logit": score if any(mask) else None,
                        "label_logit": {label: value for label, keep, value in zip(names, mask, scores) if keep}})
    return records
