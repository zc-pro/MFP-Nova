"""Train and evaluate the single-add editor with cluster-held-out TheraSet scores."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import PeptideData
from .hypothesis_data import family_folds
from .metrics import evaluate_groups, evaluate_metrics
from .models.hh_setreread_v13 import HHSetRereadV13, apply_edits, edit_scores
from .multimodal import MultimodalData, MultimodalPeptides, collate_multimodal, independent_functions
from .research import checkpoint_state, load_model, make_optimizer, to_device, validate_cache
from .trainer import seed_everything, write_json


NAME = "hh_setreread_v13"
THRESHOLDS = (0., .005, .01, .02, .03, .05, .075, .10)


def fold_indices(root, fold, pool=None):
    data = PeptideData(root, "fine39")
    train = data.indices("train") if pool is None else pool
    held = family_folds(data.clusters[train], 3) == fold
    return train[~held], train[held]


@torch.inference_mode()
def base_logits(model, loader, device):
    model.eval()
    return np.concatenate([model(to_device(batch, device))["logits"].float().cpu().numpy()
                           for batch in loader])


@torch.inference_mode()
def export_fold(run, model, loader, device, fold, epoch, diagnostic, elapsed):
    dataset = loader.dataset
    data, rows = dataset.source.data, dataset.indices
    model.eval()
    scores = {"base_logits": [], "direct_logits": [], "correction": []}
    for batch in loader:
        output = model(to_device(batch, device))
        final, direct = output["logits"].float(), output["direct"].float()
        scores["base_logits"].append(final.cpu().numpy())
        scores["direct_logits"].append(direct.cpu().numpy())
        scores["correction"].append((final - direct).cpu().numpy())
    scores = {key: np.concatenate(parts) for key, parts in scores.items()}
    np.savez_compressed(run/"oof_logits.npz", row_id=data.rows[rows], ID=data.ids[rows],
                        cluster_id=data.clusters[rows], label_names=data.labels, targets=data.y[rows],
                        **scores, fold=fold, diagnostic=diagnostic)
    result = {"fold": fold, "epoch": epoch, "rows": len(rows), "diagnostic_subset": diagnostic,
              "elapsed_seconds": elapsed, "oof_logits": str(run/"oof_logits.npz")}
    write_json(run/"results.json", result)
    return result


def load_oof(data, files, diagnostic=False, pool=None):
    train = data.indices("train") if pool is None else pool
    folds = family_folds(data.clusters[train], 3)
    logits = np.empty((len(train), len(data.labels)), dtype=np.float32)
    filled = np.zeros(len(train), dtype=bool)
    seen = set()
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            fold = int(saved["fold"])
            expected = np.flatnonzero(folds == fold)
            lookup = {row: pos for row, pos in zip(data.rows[train[expected]], expected)}
            rows = saved["row_id"]
            pos = np.array([lookup.get(row, -1) for row in rows], dtype=int)
            indices = train[pos]
            scores = saved["base_logits"]
            if (fold not in range(3) or fold in seen or (pos < 0).any()
                    or len(np.unique(pos)) != len(pos)
                    or (not diagnostic and (bool(saved["diagnostic"]) or len(pos) != len(expected)))
                    or not np.array_equal(saved["label_names"], data.labels)
                    or not np.array_equal(saved["ID"], data.ids[indices])
                    or not np.array_equal(saved["cluster_id"], data.clusters[indices])
                    or not np.array_equal(saved["targets"], data.y[indices])
                    or scores.shape != (len(pos), len(data.labels)) or not np.isfinite(scores).all()):
                raise ValueError(f"OOF scores disagree with the original Train folds or label data: {path}")
            seen.add(fold)
            logits[pos], filled[pos] = scores, True
    if seen != {0, 1, 2} or not filled.any():
        raise ValueError("Provide one OOF export from each of folds 0, 1 and 2")
    return train[filled], logits[filled]


class ScoredPeptides(Dataset):
    def __init__(self, dataset, scores, neighbor_features=None):
        self.dataset = dataset
        self.scores = torch.from_numpy(scores)
        self.neighbor_features = (torch.from_numpy(neighbor_features) if neighbor_features is not None else None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = {**self.dataset[index], "base_logits": self.scores[index]}
        if self.neighbor_features is not None:
            sample["neighbor_features"] = self.neighbor_features[index]
        return sample


def make_loader(dataset, settings, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                      num_workers=settings["num_workers"], collate_fn=collate_multimodal, generator=generator)


@torch.inference_mode()
def read_gains(model, loader, device, without_background=False):
    model.eval()
    items = {"targets": [], "base_logits": [], "edit_gain": []}
    for batch in loader:
        out = model(to_device(batch, device), without_background=without_background)
        items["targets"].append(batch["y"].numpy())
        items["base_logits"].append(out["logits"].cpu().numpy())
        items["edit_gain"].append(out["edit_gain"].cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in items.items()}


def select_edits(arrays, thresholds=THRESHOLDS):
    y = arrays["targets"].astype(bool)
    base, value, label = edit_scores(torch.from_numpy(arrays["base_logits"]), torch.from_numpy(arrays["edit_gain"]))
    base, value, label = base.numpy(), value.numpy(), label.numpy()
    hits = (base & y).sum(-1)
    union = (base | y).sum(-1)
    baseline = hits / np.maximum(1, union)
    recorded = y[np.arange(len(y)), label]
    after = (hits + recorded) / np.maximum(1, union + ~recorded)
    curve = []
    for threshold in (*thresholds, None):
        edited = (value > threshold) if threshold is not None else np.zeros(len(y), dtype=bool)
        curve.append({"edit_threshold": threshold, "edits": int(edited.sum()),
                      "recorded_hits": int(recorded[edited].sum()),
                      "sample_jaccard": float(np.where(edited, after, baseline).mean())})
    key = lambda item: (item["sample_jaccard"], -item["edits"])
    best_finite = max(curve[:-1], key=key)
    best = max(reversed(curve), key=key)
    return best, best_finite, curve


def top_score_control(arrays):
    # Threshold in probability units, but preserve the raw-logit ordering.
    scores = arrays["base_logits"]
    control = dict(arrays, edit_gain=scores)
    cutoffs = np.array([.05, .10, .20, .30, .40, .45, .49])
    best, finite, curve = select_edits(control, thresholds=np.log(cutoffs/(1-cutoffs)))
    for item in curve:
        threshold = item.pop("edit_threshold")
        item["candidate_probability_threshold"] = float(1/(1+np.exp(-threshold))) if threshold is not None else None
    return best, finite, curve


def decisions(arrays, threshold, statistics):
    y, unary, gains = (arrays[key] for key in ("targets", "base_logits", "edit_gain"))
    final, labels = apply_edits(torch.from_numpy(unary), torch.from_numpy(gains), threshold)
    base, final, labels = unary >= 0, final.numpy(), labels.numpy()
    probabilities = torch.from_numpy(unary).sigmoid().numpy()
    projection, support = statistics["function_projection"], statistics["labels"]["support"]
    old = evaluate_metrics(y, probabilities, .5, support, projection, predicted=base)
    new = evaluate_metrics(y, probabilities, .5, support, projection, predicted=final)
    truth = y.astype(bool)
    truth_axes = independent_functions(y, projection)
    old_hits = (independent_functions(base, projection) & truth_axes).sum(-1)
    new_hits = (independent_functions(final, projection) & truth_axes).sum(-1)
    added = final & ~base
    union = (final | truth).sum(-1)
    per_row_gain = (final & truth).sum(-1)/np.maximum(1, union)
    per_row_gain -= (base & truth).sum(-1)/np.maximum(1, (base | truth).sum(-1))
    summary = {"edits": int((labels >= 0).sum()), "added_recorded": int((added & truth).sum()),
               "added_unrecorded": int((added & ~truth).sum()),
               "one_to_at_least_two_correct_axes": int(((old_hits == 1) & (new_hits >= 2)).sum()),
               "at_least_two_to_fewer_correct_axes": int(((old_hits >= 2) & (new_hits < 2)).sum()),
               "improved_rows": int((per_row_gain > 0).sum()), "declined_rows": int((per_row_gain < 0).sum()),
               "empty_base_contribution": float(np.where(~base.any(-1), per_row_gain, 0.).mean())}
    _, best_value, best_label = edit_scores(torch.from_numpy(unary), torch.from_numpy(gains))
    eligible = best_value.numpy() > -np.inf
    missed_rows = (~base & truth).any(-1)
    top_recorded = truth[np.arange(len(truth)), best_label.numpy()] & eligible
    summary["candidate_ranking"] = {"eligible_rows": int(eligible.sum()),
                                    "rows_with_missed_label": int(missed_rows.sum()),
                                    "top_edit_recorded": int(top_recorded.sum())}
    raw_axes = base.astype(int) @ np.asarray(projection) > 0
    candidate_current_axis = raw_axes @ np.asarray(projection).T > 0
    strata = {"empty_base": ~base.any(-1), "recorded_single_function": truth_axes.sum(-1) == 1,
              "recorded_multifunction": truth_axes.sum(-1) >= 2,
              "existing_axis_additions": (added & candidate_current_axis).any(-1),
              "new_axis_additions": (added & ~candidate_current_axis).any(-1),
              "deep_negative_additions": (added & (unary <= -4.)).any(-1),
              "other_negative_additions": (added & (unary > -4.)).any(-1)}
    summary["strata"] = {key: {"rows": int(mask.sum()), "edits": int((added & mask[:, None]).sum()),
                               "recorded_hits": int((added & truth & mask[:, None]).sum()),
                               "mean_jaccard_gain": float(per_row_gain[mask].mean()) if mask.any() else None}
                         for key, mask in strata.items()}
    return new, old, summary, dict(arrays, base_set=base, final_set=final, added_label=labels,
                                   probabilities=probabilities, jaccard_gain=per_row_gain)


def evaluation_groups(source, rows, arrays):
    statistics = source.statistics
    return evaluate_groups(arrays["targets"], arrays["probabilities"], source.groups(rows, arrays["targets"]),
                           .5, statistics["labels"]["support"], statistics["function_projection"],
                           predicted=arrays["final_set"])


def save_predictions(path, dataset, arrays, threshold, split):
    data, rows = dataset.source.data, dataset.indices
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays, row_id=data.rows[rows], ID=data.ids[rows],
                            cluster_id=data.clusters[rows], label_names=data.labels,
                            split=split, task="fine39", threshold=.5,
                            edit_threshold=np.inf if threshold is None else threshold)


def datasets(config, checkpoint, split, indices=None, limit=None, sequence_only=False):
    data = config["data"]
    source = MultimodalData(data["root"], "fine39", checkpoint["statistics"], sequence_only=sequence_only)
    dataset = MultimodalPeptides(source, data["features"], split, limit,
                                 data["peptide_features"], sequence_only=sequence_only, indices=indices)
    validate_cache(dataset, config, checkpoint["encoder"])
    return dataset


def train_editor(config, name, files):
    settings, data = config["train"], config["data"]
    if data["track"] != "fine39" or settings.get("precision") != "fp32" or settings.get("accumulation_steps", 1) != 1:
        raise ValueError("HH V1.3 editor uses Fine39, FP32 and accumulation_steps=1")
    if not settings.get("initialize_from") or not files:
        raise ValueError("Provide --initialize-from and three --hh-oof-files")
    device = torch.device(settings["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(settings["seed"])
    base, checkpoint = load_model(settings["initialize_from"], device)
    if (checkpoint["model_name"] != "theraset" or checkpoint["track"] != "fine39"
            or base.label_context or not base.evidence_decoder or not base.active.sum(1).eq(1).all()
            or checkpoint["config"]["train"].get("hh_oof_fold") is not None):
        raise ValueError("Use the full Train Fine39 TheraSet no_label_context_single_prototype checkpoint")
    config["encoder"] = checkpoint["config"]["encoder"]
    config["peptide_encoder"] = checkpoint["config"]["peptide_encoder"]
    if any(config[key]["max_length"] != 80 for key in ("encoder", "peptide_encoder")):
        raise ValueError("HH V1.3 uses max_length=80 for both encoders")
    limit, max_steps = settings.get("limit_samples"), settings.get("max_steps")
    diagnostic = bool(limit or max_steps)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    val = datasets(config, checkpoint, "val", limit=limit)
    train_rows, scores = load_oof(val.source.data, files, diagnostic)
    train = datasets(config, checkpoint, "train", indices=train_rows, limit=limit, sequence_only=True)
    scores = scores[np.searchsorted(train_rows, train.indices)]
    generator = torch.Generator().manual_seed(settings["seed"])
    train_loader = make_loader(ScoredPeptides(train, scores), settings, shuffle=True, generator=generator)
    val_scores = base_logits(base, make_loader(val, settings), device)
    val_loader = make_loader(ScoredPeptides(val, val_scores), settings)
    del base

    args = {"statistics": checkpoint["statistics"]["labels"], **config["model"]}
    model = HHSetRereadV13(**args).to(device)
    optimizer = make_optimizer(model, settings)
    run = Path(settings["output"])/name
    run.mkdir(parents=True, exist_ok=False)
    settings["oof_files"] = list(files)
    metadata = {"model": NAME, "task": "fine39", "seed": settings["seed"],
                "config": config, "labels": val.source.data.labels, "encoder": checkpoint["encoder"],
                "dataset": checkpoint["statistics"]["dataset"], "cohort": checkpoint["statistics"]["cohort"],
                "train_rows": len(train), "val_rows": len(val), "diagnostic_subset": diagnostic,
                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "torch": str(torch.__version__), "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    write_json(run/"metadata.json", metadata)
    best_key, best_finite_key = (-1., 0), (-1., 0)
    stale = steps = 0
    for epoch in range(1, settings["epochs"]+1):
        model.train()
        total, count = 0., 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(to_device(batch, device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.get("gradient_clip", 5), error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(batch["y"])
            count += len(batch["y"])
            steps += 1
            if max_steps and steps >= max_steps:
                break
        arrays = read_gains(model, val_loader, device)
        selected, finite, curve = select_edits(arrays)
        key = selected["sample_jaccard"], -selected["edits"]
        if key > best_key:
            best_key, best_epoch, best_arrays = key, epoch, arrays
            best_state, best_threshold, stale = checkpoint_state(model, cpu=True), selected["edit_threshold"], 0
        else:
            stale += 1
        finite_key = finite["sample_jaccard"], -finite["edits"]
        if finite_key > best_finite_key:
            best_finite_key = finite_key
            best_finite = {"epoch": epoch, **finite}
        with (run/"history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"epoch": epoch, "train_loss": total/count, "validation": selected,
                                     "best_finite": finite, "curve": curve}, allow_nan=False)+"\n")
        print(f"HH V1.3 {model.readout} | epoch {epoch} | loss={total/count:.6f} | "
              f"Val Jaccard={key[0]:.5f} edits={selected['edits']} | best={best_key[0]:.5f}@{best_epoch}", flush=True)
        if (settings["patience"] and stale >= settings["patience"]) or (max_steps and steps >= max_steps):
            break

    # Deploy the selected editor and the same full Train TheraSet in one checkpoint.
    state = {**best_state, **{f"backbone.{k}": v for k, v in checkpoint["model_state"].items()}}
    torch.save({**checkpoint, "model_name": NAME, "config": config, "model_state": state, "labels": val.source.data.labels,
                "model_args": dict(args, backbone_args=checkpoint["model_args"]), "threshold": .5,
                "edit_threshold": best_threshold, "best_epoch": best_epoch, "weight_source": "online",
                "selection_suite": "jaccard", "diagnostic_subset": diagnostic}, run/"best.pt")
    report, baseline, edits, arrays = decisions(best_arrays, best_threshold, checkpoint["statistics"])
    save_predictions(run/"best_validation.npz", val, arrays, best_threshold, "val")
    result = {"model": NAME, "task": "fine39", "seed": settings["seed"], "best_epoch": best_epoch,
              "epochs_completed": epoch, "global_step": steps, "threshold": .5, "edit_threshold": best_threshold,
              "threshold_strategy": "val_single_add_jaccard", "best_finite": best_finite,
              "elapsed_seconds": time.perf_counter()-started, "diagnostic_subset": diagnostic,
              "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "validation": report, "baseline": baseline, "edits": edits,
              "val_top_score_control": top_score_control(best_arrays)[0],
              "validation_groups": evaluation_groups(val.source, val.indices, arrays),
              "checkpoint": str(run/"best.pt")}
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
    loader = make_loader(dataset, {"batch_size": batch_size, "num_workers": 0})
    threshold = checkpoint["edit_threshold"]
    arrays = read_gains(model, loader, device)
    report, baseline, edits, arrays = decisions(arrays, threshold, checkpoint["statistics"])
    result = {"model": NAME, "task": "fine39", "split": split, "rows": len(dataset), "threshold": .5,
              "edit_threshold": threshold, "metrics": report, "baseline": baseline, "edits": edits,
              "groups": evaluation_groups(dataset.source, dataset.indices, arrays)}
    if diagnostics:
        scored = ScoredPeptides(dataset, arrays["base_logits"])
        unconditioned = read_gains(model, make_loader(scored, {"batch_size": batch_size, "num_workers": 0}),
                                  device, without_background=True)
        other, _, other_edits, other_arrays = decisions(unconditioned, threshold, checkpoint["statistics"])
        result["without_background"] = {"metrics": other, "edits": other_edits}
        result["curve"] = select_edits(arrays)[2]
        best, finite, curve = top_score_control(arrays)
        result["top_score_control"] = {"best": best, "best_finite": finite, "curve": curve}
        arrays["unconditional_edit_gain"] = unconditioned["edit_gain"]
        arrays["unconditional_final_set"] = other_arrays["final_set"]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    if predictions:
        save_predictions(Path(predictions), dataset, arrays, threshold, split)
    return result
