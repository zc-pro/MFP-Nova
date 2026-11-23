"""Compare matched AA34 updates with and without fused candidate supervision."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from mfp_nova.models.trievidence import center_evidence, empty_top10
from mfp_nova.multimodal import BackbonePeptides
from mfp_nova.research import to_device
from mfp_nova.research_layers import observed_label_loss
from mfp_nova.trainer import seed_everything, write_json
from run_duoevidence import DuoEvidence
from run_trievidence import batches, collect, open_data, row_jaccard


def candidate_loss(thera, aa, targets, offsets):
    fused = center_evidence(torch.stack((thera, aa)), offsets).mean(0)
    # Revisit recorded positives missed by fusion. Unrecorded labels are weak competitors.
    missed = (targets > 0) & (fused.detach() < 0)
    strongest_other = fused.masked_fill(targets > 0, -1e4).amax(1, keepdim=True)
    pair_loss = F.softplus(strongest_other - fused)
    return (pair_loss * missed).sum() / missed.sum().clamp_min(1)


def run(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != "duoevidence" or checkpoint["track"] != "fine39":
        raise ValueError("Use a packed Fine39 DuoEvidence checkpoint")
    model = DuoEvidence(checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    offsets = model.offsets.to(args.device)
    model.theraset.to(args.device).eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    cached = {}
    for split in ("train", "val"):
        source, full = open_data(args, split, checkpoint)
        backbone = BackbonePeptides(source, split, checkpoint["encoder"]["max_length"])
        np.testing.assert_array_equal(backbone.indices, full.indices)
        print(f"Cache fixed TheraSet | {split}", flush=True)
        logits, = collect(lambda batch: (model.theraset(batch)["logits"],),
                          batches(full, args.feature_batch), args.device, progress=True)
        cached[split] = backbone, logits
    del model.theraset
    train, train_logits = cached["train"]
    val, thera = cached["val"]
    targets = torch.from_numpy(source.data.y[val.indices]).bool()
    # Backbone samples carry their source row ID. Map it to the cached Train order.
    train_ids = torch.from_numpy(source.data.rows[train.indices]).long()
    ordered_ids, positions = train_ids.sort()
    train_loader = batches(train, args.batch_size, shuffle=True)
    val_loader = batches(val, args.batch_size)

    def score(reader):
        reader.eval()
        aa, = collect(lambda batch: (reader(batch),), val_loader, args.device)
        scores = center_evidence(torch.stack((thera, aa)), offsets.cpu()).mean(0)
        return scores >= 0, empty_top10(scores, thera)

    def summarize(raw, final):
        hits = (targets & final).sum(1)
        label_count = targets.sum(1)
        recall = hits / label_count.clamp_min(1)
        groups = {"two": label_count == 2, "three_plus": label_count >= 3}
        return {"raw_acc": float(row_jaccard(targets, raw).mean()),
                "sample_jaccard": float(row_jaccard(targets, final).mean()),
                "zero_hit": int((hits == 0).sum()),
                "nonempty_zero_hit": int((raw.any(1) & (hits == 0)).sum()),
                "group_recall": {key: float(recall[group].mean()) for key, group in groups.items() if group.any()}}

    base_raw, base = score(model.aa34.to(args.device))
    base_hit = (base & targets).any(1)
    history = {"baseline": summarize(base_raw, base), "seed": args.seed, "lr": args.lr,
               "ranking_weight": args.ranking_weight, "epochs": args.epochs, "history": []}
    for arm, weight in (("classification", 0.), ("fused_ranking", args.ranking_weight)):
        reader = copy.deepcopy(model.aa34)
        optimizer = torch.optim.AdamW(reader.parameters(), lr=args.lr, weight_decay=.01)
        out = args.output / arm
        out.mkdir()
        for epoch in range(1, args.epochs + 1):
            seed_everything(args.seed + epoch)
            reader.train()
            losses = torch.zeros(2, device=args.device)
            for batch in train_loader:
                indices = positions[torch.searchsorted(ordered_ids, batch["row_id"])]
                fixed = train_logits[indices].to(args.device)
                batch = to_device(batch, args.device)
                logits = reader(batch)
                labels = batch["y"]
                classification = observed_label_loss(logits, labels, source.statistics["labels"])
                ranking = (candidate_loss(fixed, logits, labels, offsets) if weight
                           else classification.new_zeros(()))
                optimizer.zero_grad(set_to_none=True)
                (classification + weight * ranking).backward()
                optimizer.step()
                losses += torch.stack((classification.detach(), ranking.detach())) * len(labels)
            raw, final = score(reader)
            hit = (final & targets).any(1)
            classification_loss, ranking_loss = (losses.cpu() / len(train)).tolist()
            record = dict(arm=arm, epoch=epoch, **summarize(raw, final),
                          classification_loss=classification_loss, ranking_loss=ranking_loss,
                          zero_rescued=int((~base_hit & hit).sum()), zero_created=int((base_hit & ~hit).sum()))
            history["history"].append(record)
            write_json(args.output / "results.json", history)
            print(json.dumps(record), flush=True)
            state = dict(checkpoint["model_state"])
            state.update({f"aa34.{key}": value.cpu() for key, value in reader.state_dict().items()})
            torch.save(dict(checkpoint, model_state=state, aa34_refinement=record), out / f"epoch_{epoch}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-batch", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--ranking-weight", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())
