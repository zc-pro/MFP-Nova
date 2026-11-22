"""Compare Fine39 AA34 fine-tuning with and without a competing-label loss."""

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from mfp_nova.models.trievidence import AA34Reader, empty_top10, fuse_evidence
from mfp_nova.multimodal import BackbonePeptides, MultimodalData, independent_functions
from mfp_nova.research import to_device
from mfp_nova.research_layers import observed_label_loss
from mfp_nova.trainer import seed_everything, write_json
from refine_trievidence_depth import candidate_ranking_loss
from run_trievidence import batches, collect, diagnostics, row_jaccard, tune_cnn_offset


def train_epoch(reader, loader, optimizer, device, stats, ranking_weight):
    reader.train()
    totals = torch.zeros(2, device=device)
    for batch in loader:
        batch = to_device(batch, device)
        logits = reader(batch)
        classification = observed_label_loss(logits, batch["y"], stats)
        ranking = (candidate_ranking_loss(logits, batch["y"], stats)
                   if ranking_weight else classification.new_zeros(()))
        optimizer.zero_grad(set_to_none=True)
        (classification + ranking_weight * ranking).backward()
        optimizer.step()
        totals += torch.stack([classification.detach(), ranking.detach()]) * len(logits)
    return (totals.cpu() / len(loader.dataset)).tolist()


def first_positive_rank(logits, y):
    order = logits.argsort(dim=1, descending=True, stable=True)
    ranked = y.gather(1, order)
    positions = torch.arange(1, y.size(1) + 1)
    return torch.where(ranked, positions, y.size(1) + 1).amin(1)


def run(args):
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != "trievidence" or checkpoint["track"] != "fine39":
        raise ValueError("Use the original packed Fine39 TriEvidence model")
    settings = checkpoint["model_args"]
    stats = checkpoint["statistics"]
    source = MultimodalData(args.data_root, "fine39", stats,
                            localize_terminals=stats.get("localized_terminal_chemistry", False))
    length = checkpoint["encoder"]["max_length"]
    train = BackbonePeptides(source, "train", length)
    val = BackbonePeptides(source, "val", length)
    rows = val.indices
    y = torch.from_numpy(source.data.y[rows].astype(bool))
    reader = AA34Reader(stats["labels"], settings["cnn_channels"], settings["cnn_dropout"],
                        settings.get("reader_head", "global"))
    state = checkpoint["model_state"]
    offsets = state["offsets"]
    reader.load_state_dict({key.removeprefix("aa34."): value for key, value in state.items()
                            if key.startswith("aa34.")})
    del checkpoint, state
    reader.requires_grad_(False)
    for module in (reader.motifs.convs[1], reader.motifs.norms[1], reader.classifier):
        module.requires_grad_(True)
    with np.load(args.checkpoint.with_name("val_predictions.npz")) as ref:
        for key, expected in (("row_id", source.data.rows[rows]), ("ID", source.data.ids[rows]),
                              ("cluster_id", source.data.clusters[rows]), ("targets", y.numpy()),
                              ("label_names", source.data.labels)):
            np.testing.assert_array_equal(ref[key], expected)
        t, original, h = (torch.from_numpy(ref[k]) for k in ("theraset_logits", "aa34_logits", "latent_logits"))
        original_raw, original_final = (torch.from_numpy(ref[k]) for k in ("predicted_raw", "predicted"))
    loader = batches(val, args.batch_size)
    reader.to(args.device).eval()
    visible, = collect(lambda batch: (reader(batch),), loader, args.device)
    torch.testing.assert_close(visible, original, rtol=1e-4, atol=1e-4)
    fusion = fuse_evidence(t, original, h, offsets)
    np.testing.assert_array_equal(fusion >= 0, original_raw)
    np.testing.assert_array_equal(empty_top10(fusion, t), original_final)

    args.output.mkdir(parents=True, exist_ok=False)
    weights = {"classification": 0., "ranking": args.ranking_weight}
    arms = {name: copy.deepcopy(reader) for name in weights}
    optimizers = {name: torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
                  for name, model in arms.items()}
    rank0 = first_positive_rank(original, y)
    hard = original_raw.any(1) & ~(original_final & y).any(1)
    groups = {"all": torch.ones(len(y), dtype=torch.bool), "nonempty_zero": hard,
              "multifunction": torch.from_numpy(independent_functions(
                  y.numpy(), stats["function_projection"]).sum(1) >= 2)}
    if args.nearest:
        with np.load(args.nearest) as near:
            order = np.argsort(near["row_id"])
            np.testing.assert_array_equal(near["row_id"][order], source.data.rows[rows])
            low = torch.from_numpy(near["similarity"][order] < .7)
            groups.update(low_similarity=low, low_nonempty_zero=low & hard)
    history = {"seed": args.seed, "batch_size": args.batch_size, "lr": args.lr, "epochs": args.epochs,
               "ranking_weight": args.ranking_weight,
               "base_raw_acc": float(row_jaccard(y, original_raw).mean()),
               "base_fallback_acc": float(row_jaccard(y, original_final).mean()), "history": []}
    for epoch in range(1, args.epochs + 1):
        for name, model in arms.items():
            # Reset shuffle and dropout to the same random stream for each arm.
            seed_everything(args.seed + epoch)
            losses = train_epoch(model, batches(train, args.batch_size, shuffle=True), optimizers[name],
                                 args.device, stats["labels"], weights[name])
            model.eval()
            aa, = collect(lambda batch: (model(batch),), loader, args.device)
            fused = fuse_evidence(t, aa, h, offsets)
            raw = fused >= 0
            rank = first_positive_rank(aa, y)
            out = args.output / name / f"epoch_{epoch}"
            out.mkdir(parents=True)
            report = diagnostics(source, val, t, aa, h, offsets, "val", out)
            key, calibrated, _ = tune_cnn_offset(y, t, aa, h, offsets)
            report["exploratory_calibrated"] = {"raw_acc": key[0], "aa34_threshold": float(calibrated[1].sigmoid())}
            report["groups"] = {
                group: {"rows": int(mask.sum()),
                        "raw_acc": float(row_jaccard(y, raw)[mask].mean()),
                        "rank_2_5_to_1": int((mask & (rank0 >= 2) & (rank0 <= 5) & (rank == 1)).sum()),
                        "rank_1_lost": int((mask & (rank0 == 1) & (rank > 1)).sum()),
                        "zero_rescued": int((mask & ~(original_raw & y).any(1) & (raw & y).any(1)).sum()),
                        "new_zero": int((mask & (original_raw & y).any(1) & ~(raw & y).any(1)).sum()),
                        "gained_observed": int((y & raw & ~original_raw)[mask].sum()),
                        "lost_observed": int((y & original_raw & ~raw)[mask].sum())}
                for group, mask in groups.items() if mask.any()}
            record = {"epoch": epoch, "arm": name, "classification_loss": losses[0], "ranking_loss": losses[1],
                      "raw_acc": report["arms"]["fusion"]["sample_jaccard"],
                      "fallback_acc": report["arms"]["fusion_empty_top10"]["sample_jaccard"]}
            report.update(record)
            write_json(out / "results.json", report)
            if epoch == args.epochs:
                torch.save({"model_state": model.state_dict(), "offsets": offsets, "labels": source.data.labels,
                            "dataset": stats["dataset"], "channels": settings["cnn_channels"],
                            "reader_head": settings.get("reader_head", "global")},
                           out / "reader.pt")
            history["history"].append(record)
            write_json(args.output / "history.json", history)
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--nearest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranking-weight", type=float, default=.1)
    parser.add_argument("--output", type=Path,
                        default=Path("runs") / datetime.now().strftime("trievidence_aa34_rank_%Y%m%d_%H%M%S"))
    run(parser.parse_args())
