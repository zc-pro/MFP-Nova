"""Fit a shared state or pair readout, retaining frozen TheraSet and AA34 weights."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from mfp_nova.models.trievidence import empty_top10, fuse_evidence
from mfp_nova.models.trievidence_shared import SharedTriEvidence
from mfp_nova.multimodal import MultimodalPeptides
from mfp_nova.research import to_device, validate_cache
from mfp_nova.research_layers import observed_label_loss
from mfp_nova.trainer import seed_everything, write_json
from run_trievidence import batches, collect, diagnostics, open_data, row_jaccard


def train(args, output):
    seed_everything(42)
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if parent["model_name"] != "trievidence" or "helixset_args" not in parent["model_args"]:
        raise ValueError("Use the original packed six-layer TriEvidence checkpoint")
    settings = {k: v for k, v in parent["model_args"].items() if k != "helixset_args"}
    settings["shared_mode"] = args.mode
    model = SharedTriEvidence(**settings)
    inherited = {k: v for k, v in parent["model_state"].items() if k.startswith(("theraset.", "aa34."))}
    model.load_state_dict({**model.state_dict(), **inherited, "offsets": parent["model_state"]["offsets"]})
    parent.pop("model_state")
    source, val = open_data(args, "val", parent)
    train_data = MultimodalPeptides(source, args.features, "train", peptide_feature_root=args.peptide_features)
    validate_cache(train_data, parent["config"], parent["encoder"])
    y = torch.from_numpy(source.data.y[val.indices].astype(bool))
    with np.load(args.checkpoint.with_name("val_predictions.npz")) as baseline:
        if str(baseline["split"]) != "val":
            raise ValueError("Use the Validation export from the packed baseline")
        for key, expected in (("row_id", source.data.rows[val.indices]), ("targets", y.numpy()),
                              ("label_names", source.data.labels), ("cluster_id", source.data.clusters[val.indices])):
            np.testing.assert_array_equal(baseline[key], expected)
        ref = tuple(float(row_jaccard(y, torch.from_numpy(baseline[k])).mean())
                    for k in ("predicted_raw", "predicted"))
        a, c = (torch.from_numpy(baseline[k]) for k in ("theraset_logits", "aa34_logits"))
    model.to(args.device)
    offsets = model.offsets.detach().cpu().clone()
    val_loader = batches(val, args.batch_size)
    train_loader = batches(train_data, args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW((p for p in model.readout.parameters() if p.requires_grad), lr=args.learning_rate)
    cutoffs = torch.logit(torch.arange(4, 37, dtype=torch.float32) / 40.)
    print(f"Shared readout {args.mode} | Train={len(train_data)} Validation={len(val)} | output={output}", flush=True)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        total = 0.
        for step, batch in enumerate(train_loader, 1):
            batch = to_device(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = observed_label_loss(model.head_logits(batch), batch["y"], source.statistics["labels"])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch["y"])
            if epoch == 1 and step % 256 == 0:
                print(f"Epoch 1 train {step}/{len(train_loader)}", flush=True)
        model.eval()
        h, = collect(lambda batch: (model.head_logits(batch),), val_loader, args.device)
        best_epoch = None
        for cutoff in cutoffs:
            offsets[2] = cutoff
            score = fuse_evidence(a, c, h, offsets)
            acc = (float(row_jaccard(y, score >= 0).mean()), float(row_jaccard(y, empty_top10(score, a)).mean()))
            key = (min(v - r for v, r in zip(acc, ref)), *acc)
            if best_epoch is None or key > best_epoch[0]:
                best_epoch = key, offsets.clone()
        key, chosen = best_epoch
        record = {"epoch": epoch, "raw_acc": key[1], "fallback_acc": key[2],
                  "shared_threshold": float(chosen[2].sigmoid()), "train_loss": total / len(train_data),
                  "seconds": perf_counter() - started}
        history.append(record)
        print(json.dumps(record), flush=True)
        if best is None or key > best[0]:
            best = key, chosen, h, epoch
            model.offsets.copy_(chosen)
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(dict(parent, model_args=settings, model_state=state, epoch=epoch),
                       output / "best.pt")
    _, chosen, h, epoch = best
    report = diagnostics(source, val, a, c, h, chosen, "val", output, shared=True)
    report.update(shared_mode=args.mode, best_epoch=epoch, train_rows=len(train_data), history=history,
                  reference_checkpoint=str(args.checkpoint), parameters=sum(p.numel() for p in model.parameters()))
    write_json(output / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--mode", choices=("states", "pairs"), required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    output = args.output or Path("runs") / datetime.now().strftime(f"trievidence_{args.mode}_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    train(args, output)
