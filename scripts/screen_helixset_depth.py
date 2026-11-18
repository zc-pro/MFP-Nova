"""Compare HelixSet's supervised exits on Fine39 Validation without retraining."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mfp_nova.metrics import evaluate_metrics
from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from mfp_nova.research import load_model, to_device, validate_cache


@torch.inference_mode()
def collect(model, loader, device, show_progress=False):
    targets = []
    columns = {}
    for step, batch in enumerate(loader, 1):
        output = model(to_device(batch, device))
        targets.append(batch["y"].cpu().numpy())
        exits = {"direct": output["direct"]}
        for depth, logits in enumerate([*output["decoder_logits"], output["logits"]], 1):
            exits[f"decoder_{depth}"] = logits
        for name, logits in exits.items():
            columns.setdefault(name, []).append(logits.float().sigmoid().cpu().numpy())
        if show_progress and (step == 1 or step % 32 == 0 or step == len(loader)):
            print(f"Validation batch {step}/{len(loader)}", flush=True)
    return np.concatenate(targets), {name: np.concatenate(parts) for name, parts in columns.items()}


def summarize(targets, columns, threshold, train_support):
    scores = {}
    for name, probabilities in columns.items():
        metrics = evaluate_metrics(targets, probabilities, threshold, train_support)
        top5 = np.argsort(-probabilities, axis=1, kind="stable")[:, :5]
        hits = np.take_along_axis(targets, top5, axis=1).any(axis=1)
        scores[name] = {key: metrics[key] for key in
                        ("sample_jaccard", "sample_precision", "sample_recall", "macro_ap", "tail_macro_ap")}
        scores[name]["top5_hit_rows"] = int(hits.sum())
        observed_hit = ((probabilities >= threshold) & targets.astype(bool)).any(axis=1)
        scores[name]["zero_hit_rows"] = int((~observed_hit).sum())
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Original Fine39 HelixSet best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rows", type=int, default=0, help="Seed-42 random subset; 0 uses all Validation")
    parser.add_argument("--latent-depth", type=int, help="Keep only the first N latent blocks before reading all exits")
    parser.add_argument("--output-dir", type=Path, help="New directory for this arm")
    parser.add_argument("--data-root", type=Path, help="Override the dataset path saved in the checkpoint")
    parser.add_argument("--features", type=Path, help="Override the primary encoder cache path")
    parser.add_argument("--peptide-features", type=Path, help="Override the peptide encoder cache path")
    args = parser.parse_args()
    model, checkpoint = load_model(args.checkpoint, args.device)
    if (checkpoint["model_name"] != "helixset" or checkpoint["track"] != "fine39"
            or hasattr(model, "early_evidence")):
        parser.error("Use the standard Fine39 HelixSet checkpoint with supervised decoder exits")
    if args.latent_depth is not None:
        if not 1 <= args.latent_depth <= len(model.latent_blocks):
            parser.error("latent-depth must be within checkpoint depth")
        model.latent_blocks = model.latent_blocks[:args.latent_depth]
    data = checkpoint["config"]["data"]
    stats = checkpoint["statistics"]
    source = MultimodalData(args.data_root or data["root"], "fine39", stats,
                            localize_terminals=stats.get("localized_terminal_chemistry", False))
    dataset = MultimodalPeptides(source, args.features or data["features"], "val",
                                 peptide_feature_root=args.peptide_features or data.get("peptide_features"))
    validate_cache(dataset, checkpoint["config"], checkpoint["encoder"])
    if not 0 <= args.rows <= len(dataset):
        parser.error("rows must be within Validation size")
    rows = np.arange(len(dataset))
    if args.rows:
        rows = np.sort(np.random.default_rng(42).choice(rows, size=args.rows, replace=False))
    loader = DataLoader(Subset(dataset, rows), batch_size=args.batch_size, collate_fn=collate_multimodal)
    run = args.output_dir or Path("runs") / datetime.now().strftime("helixset_depth_screen_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True)
    print(f"Validation | rows={len(rows)} latent_depth={len(model.latent_blocks)} output={run}", flush=True)
    targets, columns = collect(model, loader, args.device, show_progress=True)
    threshold = np.asarray(checkpoint["threshold"])
    support = stats["labels"]["support"]
    report = {"checkpoint": str(args.checkpoint), "split": "val", "rows": len(rows), "sample_seed": 42,
              "batch_size": args.batch_size, "latent_depth": len(model.latent_blocks), "threshold": threshold.tolist(),
              "exits": summarize(targets, columns, threshold, support)}
    indices = dataset.indices[rows]
    np.savez_compressed(run / "exits.npz", row_id=source.data.rows[indices], cluster_id=source.data.clusters[indices],
                        targets=targets, label_names=source.data.labels, **columns)
    result = json.dumps(report, indent=2)
    (run / "summary.json").write_text(result, encoding="utf-8")
    print(result, run, sep="\n")


if __name__ == "__main__":
    main()
