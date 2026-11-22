"""Compare an original-head update and a signed local update using frozen Fine39 features."""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset

from compare_validation_predictions import paired_cluster_interval
from mfp_nova.multimodal import MultimodalPeptides, independent_functions
from mfp_nova.research import load_model, to_device, validate_cache
from mfp_nova.research_layers import observed_label_loss
from mfp_nova.trainer import seed_everything, write_json
from run_trievidence import batches, open_data, row_jaccard


class HeadUpdate(nn.Module):
    def __init__(self, dim, labels, local=False, window_width=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(labels, 3 * dim))
        self.bias = nn.Parameter(torch.zeros(labels))
        if local:
            width = window_width or dim
            self.windows = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU())
            self.detectors = nn.Linear(32, labels, bias=False)
            self.context = nn.Linear(3 * dim, 1)
            self.correction = nn.Linear(3, 1)
            nn.init.zeros_(self.correction.weight)
            nn.init.zeros_(self.correction.bias)

    def forward(self, features, windows, mask, base, threshold_logit):
        features = features.float()
        scores = base + (features * self.weight).sum(-1) + self.bias
        if hasattr(self, "windows"):
            width = self.windows[0].normalized_shape[0]
            if windows.size(-1) != width:
                windows = F.pad(windows, (0, width - windows.size(-1)))
            evidence = self.detectors(self.windows(windows.float()))
            top = evidence.masked_fill(~mask[..., None], -torch.inf).topk(min(3, mask.size(1)), dim=1).values
            present = top.isfinite()
            peak = top.masked_fill(~present, 0.).sum(1) / present.sum(1).clamp_min(1)
            state = self.context(features).squeeze(-1)
            update = self.correction(torch.stack([peak, state, peak * torch.tanh(state)], dim=-1)).squeeze(-1)
            scores = scores + update
        return scores - threshold_logit


@torch.no_grad()
def extract(model, dataset, args, teacher=None):
    captured = {}

    def capture_windows(_module, _inputs, output):
        captured["windows"], captured["mask"], _ = output

    def capture_features(_module, _inputs, output):
        captured["features"] = output

    hooks = [model.local_windows.register_forward_hook(capture_windows),
             model.output_norm.register_forward_hook(capture_features)]
    cache = []
    try:
        loader = batches(dataset, args.batch_size)
        for step, batch in enumerate(loader, 1):
            inputs = to_device(batch, args.device)
            logits = model(inputs)["logits"].float().cpu()
            entry = (captured["features"].to(device="cpu", dtype=torch.float16),
                     captured["windows"].to(device="cpu", dtype=torch.float16),
                     captured["mask"].cpu(), logits, batch["y"])
            if teacher is not None:
                maps = torch.cat(teacher.cnn.residue_maps(inputs), dim=-1)
                entry += (maps.to(device="cpu", dtype=torch.float16), batch["mask"])
            cache.append(entry)
            if step == 1 or step % 128 == 0 or step == len(loader):
                print(f"Frozen forward {step}/{len(loader)}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    return cache


def align(path, source, val):
    with np.load(path) as file:
        if str(file["split"]) != "val":
            raise ValueError("Expected original Validation predictions")
        order = np.argsort(file["row_id"])
        rows = val.indices
        for key, expected in (("row_id", source.data.rows[rows]), ("ID", source.data.ids[rows]),
                              ("cluster_id", source.data.clusters[rows]), ("targets", source.data.y[rows])):
            np.testing.assert_array_equal(file[key][order], expected)
        np.testing.assert_array_equal(file["label_names"], source.data.labels)
        return {key: file[key][order] for key in file.files if file[key].ndim == 2}


@torch.no_grad()
def predict(head, cache, device, offset, use_cnn):
    return torch.cat([head(*head_inputs(batch, device, use_cnn), offset).cpu() for batch in cache])


def head_inputs(batch, device, use_cnn):
    features, windows, mask, logits = batch[:4]
    if use_cnn:
        windows, mask = batch[5:]
    return tuple(value.to(device) for value in (features, windows, mask, logits))


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    model, checkpoint = load_model(args.checkpoint, args.device)
    if checkpoint["model_name"] != "therahelix" or checkpoint["track"] != "fine39":
        raise ValueError("Use the selected Fine39 TheraHelix checkpoint")
    teacher = None
    if args.theraset_checkpoint:
        teacher, teacher_checkpoint = load_model(args.theraset_checkpoint, args.device)
        if (teacher_checkpoint["model_name"] != "theraset" or teacher_checkpoint["track"] != "fine39"
                or teacher_checkpoint.get("best_epoch") != 53 or not teacher.cnn.chemical_evidence):
            raise ValueError("Use the original epoch-53 Fine39 TheraSet baseline")
        for key in ("dataset", "chemical_vocab", "localized_terminal_chemistry"):
            if teacher_checkpoint["statistics"].get(key) != checkpoint["statistics"].get(key):
                raise ValueError(f"Chemistry inputs differ between checkpoints: {key}")
    source, val = open_data(args, "val", checkpoint)
    rows = val.indices
    reference = align(args.checkpoint.with_name("best_validation.npz"), source, val)
    old = align(args.old_predictions, source, val)
    threshold = float(checkpoint["threshold"])
    offset = float(torch.tensor(threshold).logit())
    print("Extract Validation", flush=True)
    val_cache = extract(model, val, args, teacher)
    base = torch.cat([entry[3] for entry in val_cache]) - offset
    np.testing.assert_array_equal(base.numpy() >= 0, reference["probabilities"] >= threshold)
    y = source.data.y[rows].astype(bool)
    t = old["theraset_logits"]
    h = old["latent_logits"] - np.float32(np.log(.64 / .36))
    old_two = np.clip(t, -13.81551, 13.81551) + np.clip(old["latent_logits"], -13.81551, 13.81551) - np.float32(np.log(.64 / .36)) >= 0
    original = base.numpy() >= 0
    tracked = {"lost_theraset_only": y & old_two & ~original & (t >= 0) & (h < 0),
               "kept_helixset_only": y & original & (t < 0) & (h >= 0),
               "extra_unrecorded": ~y & original & ~old_two}
    groups = {"all": np.ones(len(y), bool),
              "multifunction": independent_functions(y, source.statistics["function_projection"]).sum(1) >= 2}
    if args.nearest:
        with np.load(args.nearest) as nearest:
            order = np.argsort(nearest["row_id"])
            np.testing.assert_array_equal(nearest["row_id"][order], source.data.rows[rows])
            groups["similarity_lt70"] = nearest["similarity"][order] < .7
    train = MultimodalPeptides(source, args.features, "train", peptide_feature_root=args.peptide_features)
    validate_cache(train, checkpoint["config"], checkpoint["encoder"])
    order = torch.randperm(len(train)).tolist()
    print("Extract Train", flush=True)
    train_cache = extract(model, Subset(train, order), args, teacher)
    del model, teacher
    dim = train_cache[0][1].size(-1)
    if args.theraset_checkpoint:
        width = max(dim, train_cache[0][5].size(-1))
        initial = HeadUpdate(dim, y.shape[1], local=True, window_width=width)
        control, candidate = "therahelix_windows", "theraset_cnn"
        heads = {control: initial.to(args.device), candidate: deepcopy(initial).to(args.device)}
    else:
        control, candidate = "original_head", "local_residual"
        heads = {control: HeadUpdate(dim, y.shape[1]).to(args.device),
                 candidate: HeadUpdate(dim, y.shape[1], local=True).to(args.device)}
    optimizers = {key: torch.optim.Adam(head.parameters(), lr=args.lr) for key, head in heads.items()}
    history = []
    for epoch in range(1, args.epochs + 1):
        losses = dict.fromkeys(heads, 0.)
        for idx in torch.randperm(len(train_cache)).tolist():
            batch = train_cache[idx]
            labels = batch[4].to(args.device)
            for key, head in heads.items():
                scores = head(*head_inputs(batch, args.device, key == "theraset_cnn"), 0.)
                loss = observed_label_loss(scores, labels, source.statistics["labels"])
                optimizers[key].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[key].step()
                losses[key] += float(loss.detach()) * len(labels)
        outputs = {key: predict(head, val_cache, args.device, offset, key == "theraset_cnn")
                   for key, head in heads.items()}
        fixed_count = {}
        for key, scores in outputs.items():
            keep = np.zeros_like(original)
            ranks = np.argsort(-scores.numpy(), axis=1, kind="stable")
            np.put_along_axis(keep, ranks, np.arange(y.shape[1]) < original.sum(1, keepdims=True), axis=1)
            fixed_count[key] = keep
        cutoffs = {key: max(np.arange(-.5, .501, .05), key=lambda c: row_jaccard(
            torch.from_numpy(y), scores >= c).mean().item()) for key, scores in outputs.items()}
        metrics = {}
        for group, keep in groups.items():
            truth = torch.from_numpy(y[keep])
            masks = {"base": original[keep], **{key: scores.numpy()[keep] >= 0 for key, scores in outputs.items()}}
            jac = {key: row_jaccard(truth, torch.from_numpy(mask)).numpy() for key, mask in masks.items()}
            calibration = {}
            for key, scores in outputs.items():
                cutoff = cutoffs[key]
                calibration[key] = {"val_fitted_logit_offset": float(cutoff),
                                    "acc_pct": 100 * row_jaccard(truth, scores[keep] >= cutoff).mean().item()}
            metrics[group] = {"rows": int(keep.sum()), "fixed_base_count_acc_pct": {
                key: 100 * row_jaccard(truth, torch.from_numpy(mask[keep])).mean().item()
                for key, mask in fixed_count.items()}, "arms": {
                key: {"acc_pct": 100 * float(jac[key].mean()),
                      "zero_hit": int((~(mask & y[keep]).any(1)).sum()),
                      "added_unrecorded": int((mask & ~original[keep] & ~y[keep]).sum()),
                      "lost_observed": int((~mask & original[keep] & y[keep]).sum()),
                      **{name: int((mask & tracked_mask[keep]).sum()) for name, tracked_mask in tracked.items()}}
                for key, mask in masks.items()},
                f"{candidate}_vs_{control}_ci95_pp": paired_cluster_interval(
                    jac[candidate] - jac[control], source.data.clusters[rows][keep]),
                "exploratory_calibrated": calibration}
        history.append({"epoch": epoch, "loss": {k: v/len(train) for k, v in losses.items()}, "groups": metrics})
        write_json(args.output / "results.json", {"checkpoint": str(args.checkpoint), "seed": args.seed,
                   "learning_rate": args.lr, "batch_size": args.batch_size, "planned_final_epoch": args.epochs,
                   "theraset_checkpoint": str(args.theraset_checkpoint) if args.theraset_checkpoint else None,
                   "head_parameters": {k: sum(p.numel() for p in head.parameters()) for k, head in heads.items()},
                   "history": history})
        print(f"Epoch {epoch}/{args.epochs} " + str({k: v["acc_pct"] for k, v in metrics["all"]["arms"].items()}), flush=True)
    torch.save({key: head.cpu().state_dict() for key, head in heads.items()}, args.output / "heads.pt")
    np.savez_compressed(args.output / "val_scores.npz", row_id=source.data.rows[rows], base=base.numpy(),
                        **{key: value.numpy() for key, value in outputs.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("old_predictions", type=Path, help="Original packed TriEvidence val_predictions.npz")
    parser.add_argument("--theraset-checkpoint", type=Path,
                        help="Compare equal heads reading frozen TheraHelix windows and original TheraSet CNN maps")
    parser.add_argument("--nearest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs") / datetime.now().strftime("therahelix_local_read_%Y%m%d_%H%M%S"))
    main(parser.parse_args())
