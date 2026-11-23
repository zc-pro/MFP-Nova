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
from run_duoevidence import AA34BackboneInteraction, AA34FamilyReadout, AA34PooledSupplement, AA34PositionReadout, DuoEvidence
from run_trievidence import batches, collect, open_data, row_jaccard


def candidate_loss(thera, aa, targets, offsets):
    fused = center_evidence(torch.stack((thera, aa)), offsets).mean(0)
    # Revisit recorded positives missed by fusion. Unrecorded labels are weak competitors.
    missed = (targets > 0) & (fused.detach() < 0)
    strongest_other = fused.masked_fill(targets > 0, -1e4).amax(1, keepdim=True)
    pair_loss = F.softplus(strongest_other - fused)
    return (pair_loss * missed).sum() / missed.sum().clamp_min(1)


def preserve_hits(original, updated, targets):
    known = (targets > 0) & (original >= 0)
    already_hit = known.any(1, keepdim=True)
    new_unrecorded = already_hit & (targets == 0) & (original < 0)
    lost = (F.relu(-updated).square() * known).sum() / known.sum().clamp_min(1)
    added = (F.relu(updated).square() * new_unrecorded).sum() / already_hit.sum().clamp_min(1)
    return lost + .3 * added


def run(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (checkpoint["model_name"] != "duoevidence" or checkpoint["track"] != "fine39"
            or any(checkpoint.get(f"aa34_{kind}") for kind in ("readout_position", "family_mode", "context_mode"))):
        raise ValueError("Use the original packed Fine39 DuoEvidence checkpoint")
    supplement_test = args.experiment == "pooled_supplement"
    if supplement_test:
        context = torch.load(args.context_checkpoint, map_location="cpu", weights_only=True)
        if (context.get("aa34_context_mode") != "pooled" or context["labels"] != checkpoint["labels"]
                or any(not torch.equal(value, context["model_state"][key])
                       for key, value in checkpoint["model_state"].items() if key.startswith("theraset."))):
            raise ValueError("The pooled checkpoint must use the same TheraSet and label order")
        pooled_state = {}
        for key, value in context["model_state"].items():
            if key.startswith("aa34.") and not key.startswith("aa34.classifier.1."):
                pooled_state[key.removeprefix("aa34.").replace("classifier.0.", "classifier.")] = value
        del context
    model = DuoEvidence(checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    offsets = model.offsets.to(args.device)
    model.theraset.to(args.device).eval().requires_grad_(False)
    args.output.mkdir(parents=True, exist_ok=False)
    cached = {}
    validation_only = args.experiment in ("family", "relative", "backbone_relative")
    for split in (("val",) if validation_only else ("train", "val")):
        source, full = open_data(args, split, checkpoint)
        backbone = BackbonePeptides(source, split, checkpoint["encoder"]["max_length"])
        np.testing.assert_array_equal(backbone.indices, full.indices)
        print(f"Cache fixed TheraSet | {split}", flush=True)
        logits, = collect(lambda batch: (model.theraset(batch)["logits"],),
                          batches(full, args.feature_batch), args.device, progress=True)
        cached[split] = (full if supplement_test else backbone), logits
    del model.theraset
    val, thera = cached["val"]
    if validation_only:
        train = BackbonePeptides(source, "train", checkpoint["encoder"]["max_length"])
    else:
        train, train_logits = cached["train"]
        train_ids = torch.from_numpy(source.data.rows[train.indices]).long()
        ordered_ids, positions = train_ids.sort()
    targets = torch.from_numpy(source.data.y[val.indices]).bool()
    train_loader = batches(train, args.batch_size, shuffle=True)
    val_loader = batches(val, args.batch_size)

    def score(reader):
        reader.eval()
        aa, = collect(lambda batch: (reader(batch),), val_loader, args.device)
        scores = center_evidence(torch.stack((thera, aa)), offsets.cpu()).mean(0)
        return scores >= 0, empty_top10(scores, thera), scores.argmax(1)

    names = source.data.labels
    pairs = [(actual, wrong, names.index(actual), names.index(wrong)) for actual, wrong in (
        ("antibacterial__unknown", "anti_gram_"),
        ("antimicrobial__unknown", "anti_gram_"),
        ("anti_gram+", "antimicrobial__unknown"),
        ("anti_gram_", "antimicrobial__unknown"),
        ("antifungal__unknown", "antimicrobial__unknown"),
        ("antimicrobial__unknown", "anti_gram+"),
    ) if actual in names and wrong in names]

    def summarize(raw, final, top):
        hits = (targets & final).sum(1)
        label_count = targets.sum(1)
        recall = hits / label_count.clamp_min(1)
        groups = {"two": label_count == 2, "three_plus": label_count >= 3}
        report = {"raw_acc": float(row_jaccard(targets, raw).mean()),
                "sample_jaccard": float(row_jaccard(targets, final).mean()),
                "zero_hit": int((hits == 0).sum()),
                "nonempty_zero_hit": int((raw.any(1) & (hits == 0)).sum()),
                "group_recall": {key: float(recall[group].mean()) for key, group in groups.items() if group.any()}}
        if args.experiment in ("family", "relative", "backbone_relative", "pooled_supplement"):
            hard = raw.any(1) & (hits == 0)
            report["confusions"] = {
                f"{actual} -> {wrong}": int((hard & targets[:, i] & (top == j)).sum())
                for actual, wrong, i, j in pairs}
        return report

    base_raw, base, base_top = score(model.aa34.to(args.device))
    base_hit = (base & targets).any(1)
    original_hard = base_raw.any(1) & ~base_hit
    history = {"baseline": summarize(base_raw, base, base_top), "seed": args.seed, "lr": args.lr,
               "experiment": args.experiment, "epochs": args.epochs, "history": []}
    if args.experiment == "readout":
        arms = (("global", 0.), ("local", 0.))
    elif args.experiment == "relative":
        arms = (("context", 0.), ("relative", 0.))
    elif args.experiment == "backbone_relative":
        arms = (("classification", 0.), ("backbone_relative", 0.))
    elif args.experiment == "family":
        arms = (("classification", 0.), ("auxiliary", 0.), ("decomposed", 0.))
        history["family_weight"] = args.family_weight
    elif supplement_test:
        arms = (("pooled_free", 0.), ("pooled_protected", 0.))
        history.update(protect_weight=args.protect_weight, context_checkpoint=str(args.context_checkpoint))
    else:
        arms = (("classification", 0.), ("fused_ranking", args.ranking_weight))
        history["ranking_weight"] = args.ranking_weight
    for arm, weight in arms:
        readout_test = args.experiment in ("readout", "relative")
        family_test = arm in ("auxiliary", "decomposed")
        if supplement_test:
            reader = AA34PooledSupplement(checkpoint["model_args"]).to(args.device).requires_grad_(False)
            reader.motifs.load_state_dict(model.aa34.motifs.state_dict())
            reader.classifier.load_state_dict(model.aa34.classifier.state_dict())
            reader.pooled.load_state_dict(pooled_state)
            reader.supplement.requires_grad_(True)
            initial_raw, initial_final, _ = score(reader)
            if not torch.equal(initial_raw, base_raw) or not torch.equal(initial_final, base):
                raise ValueError("Pooled supplement changed predictions before training")
        elif readout_test:
            seed_everything(args.seed)
            reader = AA34PositionReadout(source.statistics["labels"], arm).to(args.device)
            reader.motifs.load_state_dict(model.aa34.motifs.state_dict())
            reader.classifier.load_state_dict(model.aa34.classifier.state_dict())
            reader.motifs.requires_grad_(False)
            reader.classifier.requires_grad_(False)
            initial_raw, initial_final, _ = score(reader)
            if not torch.equal(initial_raw, base_raw) or not torch.equal(initial_final, base):
                raise ValueError(f"{arm} readout changed the checkpoint predictions before training")
        elif family_test:
            seed_everything(args.seed)
            reader = AA34FamilyReadout(source.statistics["labels"], arm).to(args.device)
            reader.motifs.load_state_dict(model.aa34.motifs.state_dict())
            reader.classifier.load_state_dict(model.aa34.classifier.state_dict())
            reader.initialize_family()
        elif arm == "backbone_relative":
            reader = AA34BackboneInteraction(source.statistics["labels"]).to(args.device)
            reader.motifs.load_state_dict(model.aa34.motifs.state_dict())
            reader.classifier.load_state_dict(model.aa34.classifier.state_dict())
            initial_raw, initial_final, _ = score(reader)
            if not torch.equal(initial_raw, base_raw) or not torch.equal(initial_final, base):
                raise ValueError("Backbone interaction changed predictions before training")
        else:
            reader = copy.deepcopy(model.aa34)
        if args.experiment == "backbone_relative":
            reader.motifs.requires_grad_(False)
            reader.motifs.convs[-1].requires_grad_(True)
            reader.motifs.norms[-1].requires_grad_(True)
        trainable = [p for p in reader.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=.01)
        out = args.output / arm
        out.mkdir()
        for epoch in range(1, args.epochs + 1):
            seed_everything(args.seed + epoch)
            reader.train()
            if readout_test or supplement_test:
                reader.motifs.eval()
                reader.classifier.eval()
            if supplement_test:
                reader.pooled.eval()
            losses = torch.zeros(4, device=args.device)
            for batch in train_loader:
                if not validation_only and (readout_test or weight or supplement_test):
                    indices = positions[torch.searchsorted(ordered_ids, batch["row_id"])]
                    fixed = train_logits[indices].to(args.device)
                batch = to_device(batch, args.device)
                labels = batch["y"]
                protection = offsets.new_zeros(())
                if supplement_test:
                    logits, original = reader(batch, return_base=True)
                    family_loss = logits.new_zeros(())
                    if arm == "pooled_protected":
                        original_fused = center_evidence(torch.stack((fixed, original)), offsets).mean(0)
                        updated = center_evidence(torch.stack((fixed, logits)), offsets).mean(0)
                        protection = preserve_hits(original_fused, updated, labels)
                elif family_test:
                    logits, family = reader(batch, return_family=True)
                    observed = (labels @ reader.family_projection > 0).to(family.dtype)
                    weak = torch.where(observed > 0, 1., .3)
                    family_loss = F.binary_cross_entropy_with_logits(family, observed, weight=weak)
                else:
                    logits = reader(batch)
                    family_loss = logits.new_zeros(())
                supervised = (center_evidence(torch.stack((fixed, logits)), offsets).mean(0)
                              if args.experiment in ("readout", "pooled_supplement") else logits)
                classification = observed_label_loss(supervised, labels, source.statistics["labels"])
                ranking = (candidate_loss(fixed, logits, labels, offsets) if weight
                           else classification.new_zeros(()))
                optimizer.zero_grad(set_to_none=True)
                (classification + weight * ranking + args.family_weight * family_loss
                 + args.protect_weight * protection).backward()
                optimizer.step()
                losses += torch.stack((classification.detach(), ranking.detach(), family_loss.detach(),
                                       protection.detach())) * len(labels)
            raw, final, top = score(reader)
            hit = (final & targets).any(1)
            classification_loss, ranking_loss, family_loss, preservation_loss = (losses.cpu() / len(train)).tolist()
            record = dict(arm=arm, epoch=epoch, **summarize(raw, final, top),
                          classification_loss=classification_loss, ranking_loss=ranking_loss,
                          zero_rescued=int((~base_hit & hit).sum()), zero_created=int((base_hit & ~hit).sum()),
                          original_nonempty_zero_rescued=int((original_hard & hit).sum()),
                          original_nonempty_zero_delta_pp=float(
                              100 * ((row_jaccard(targets, final) - row_jaccard(targets, base)) * original_hard).mean()),
                          original_nonempty_hit_delta_pp=float(
                              100 * ((row_jaccard(targets, final) - row_jaccard(targets, base))
                                     * (base_raw.any(1) & base_hit)).mean()),
                          changed_predictions=int((final != base).any(1).sum()),
                          trainable_parameters=sum(p.numel() for p in trainable))
            if args.experiment == "family":
                record["family_loss"] = family_loss
            if supplement_test:
                record["preservation_loss"] = preservation_loss
            history["history"].append(record)
            write_json(args.output / "results.json", history)
            print(json.dumps(record), flush=True)
            state = dict(checkpoint["model_state"])
            state.update({f"aa34.{key}": value.cpu() for key, value in reader.state_dict().items()})
            refined = dict(checkpoint, model_state=state, aa34_refinement=record)
            if readout_test:
                refined["aa34_readout_position"] = arm
            if arm == "backbone_relative":
                refined["aa34_readout_position"] = arm
            if supplement_test:
                refined["aa34_readout_position"] = "pooled_supplement"
            if family_test:
                refined["aa34_family_mode"] = arm
            torch.save(refined, out / f"epoch_{epoch}.pt")


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
    parser.add_argument("--experiment", choices=("ranking", "readout", "family", "relative", "backbone_relative",
                                                  "pooled_supplement"), default="ranking")
    parser.add_argument("--context-checkpoint", type=Path)
    parser.add_argument("--protect-weight", type=float, default=.1)
    parser.add_argument("--family-weight", type=float, default=.02)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--ranking-weight", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (args.experiment == "pooled_supplement") != (args.context_checkpoint is not None):
        parser.error("--context-checkpoint is required for pooled_supplement")
    if args.lr is None:
        args.lr = 1e-3 if args.experiment in ("readout", "relative", "pooled_supplement") else 2e-5
    run(args)
