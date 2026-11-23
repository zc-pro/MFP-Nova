"""Fine39 TheraSet + AA34. Train both readers from random initialization."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mfp_nova.cli import main as cli
from mfp_nova.metrics import evaluate_metrics
from mfp_nova.models.theraset import TheraSet
from mfp_nova.models.trievidence import AA34Reader, LocalLabelReadout, center_evidence, empty_top10
from mfp_nova.multimodal import BackbonePeptides, function_projection
from mfp_nova.trainer import seed_everything, write_json
from run_trievidence import batches, collect, open_data, pass_train, read_anchor, row_jaccard


def decision_scores(theraset, aa34, weight, cutoff):
    scores = center_evidence(torch.stack((theraset, aa34)), theraset.new_zeros(2))
    return (1 - weight) * scores[0] + weight * scores[1] - cutoff


class AA34PositionReadout(AA34Reader):
    def __init__(self, statistics, position):
        super().__init__(statistics)
        num_labels = len(statistics["labels"])
        width = 1536 if position in ("context", "relative") else 512
        self.readout = LocalLabelReadout(width, num_labels, subtract_background=False)
        self.readout_gain = nn.Parameter(torch.zeros(num_labels))
        self.position = position

    def forward(self, batch):
        logits, maps = super().forward(batch, return_maps=True)
        mask = batch["mask"]
        if self.position == "global":
            maps = maps.masked_fill(~mask[..., None], -torch.inf).amax(1, keepdim=True)
            mask = mask.any(1, keepdim=True)
        elif self.position in ("context", "relative"):
            valid = mask[..., None]
            if self.position == "relative":
                left, right = (neighbor_context(maps, mask, offsets) for offsets in ((-2, -4), (2, 4)))
            else:
                mean = maps.masked_fill(~valid, 0).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
                peak = maps.masked_fill(~valid, -torch.inf).amax(1, keepdim=True)
                left, right = mean.expand_as(maps), peak.expand_as(maps)
            maps = torch.cat((maps, left, right), dim=-1)
        return logits + self.readout_gain * self.readout(maps, mask)


def neighbor_context(maps, mask, offsets):
    """Average existing neighbors at signed offsets, ignoring boundaries and padding."""
    positions = torch.arange(maps.shape[1], device=maps.device)
    neighbors = positions[:, None] + positions.new_tensor(offsets)
    safe = neighbors.clamp(0, maps.shape[1] - 1)
    valid = (neighbors == safe)[None] & mask[:, safe]
    total = maps[:, safe].masked_fill(~valid[..., None], 0).sum(2)
    return total / valid.sum(2).clamp_min(1)[..., None]


class AA34BackboneInteraction(AA34Reader):
    def __init__(self, statistics):
        super().__init__(statistics)
        self.context = nn.Sequential(nn.Linear(1024, 32, bias=False), nn.GELU(),
                                     nn.Linear(32, 512, bias=False))
        nn.init.zeros_(self.context[-1].weight)

    def forward(self, batch):
        mask = batch["mask"]
        maps = self.motifs(batch)
        left = neighbor_context(maps, mask, (-2, -4))
        right = neighbor_context(maps, mask, (2, 4))
        maps = maps + maps * self.context(torch.cat((left, right), dim=-1)).tanh()
        return self.classifier(maps.masked_fill(~mask[..., None], -torch.inf).amax(1))


class AA34FamilyReadout(AA34Reader):
    def __init__(self, statistics, mode):
        super().__init__(statistics)
        projection = torch.tensor(function_projection(statistics["labels"], "fine39"))
        # Only groups with multiple labels get a shared score. Singleton logits stay independent.
        projection = projection[:, projection.sum(0) > 1]
        self.register_buffer("family_projection", projection)
        self.family = nn.Linear(self.classifier[-1].in_features, projection.shape[1])
        self.mode = mode

    def initialize_family(self):
        mean = self.family_projection.T / self.family_projection.sum(0)[:, None]
        with torch.no_grad():
            self.family.weight.copy_(mean @ self.classifier[-1].weight)
            self.family.bias.copy_(mean @ self.classifier[-1].bias)

    def forward(self, batch, return_family=False):
        maps = self.motifs(batch)
        pooled = maps.masked_fill(~batch["mask"][..., None], -torch.inf).amax(1)
        hidden = self.classifier[:-1](pooled)
        fine = self.classifier[-1](hidden)
        family = self.family(hidden)
        if self.mode == "decomposed":
            mean = (fine @ self.family_projection) / self.family_projection.sum(0)
            fine = fine + (family - mean) @ self.family_projection.T
        return (fine, family) if return_family else fine


class DuoEvidence(nn.Module):
    def __init__(self, thera_args, readout_position=None, family_mode=None):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.theraset = TheraSet(**thera_args)
        statistics = thera_args["statistics"]
        if readout_position == "backbone_relative":
            self.aa34 = AA34BackboneInteraction(statistics)
        elif readout_position:
            self.aa34 = AA34PositionReadout(statistics, readout_position)
        elif family_mode:
            self.aa34 = AA34FamilyReadout(statistics, family_mode)
        else:
            self.aa34 = AA34Reader(statistics)
        self.register_buffer("offsets", torch.zeros(2))

    def forward(self, batch, decision=None):
        a = self.theraset(batch)["logits"]
        b = self.aa34(batch)
        if decision is None:
            logits = center_evidence(torch.stack((a, b)), self.offsets).mean(0)
        else:
            logits = decision_scores(a, b, decision["aa34_weight"], decision["cutoff"])
        return {"logits": logits, "predicted": empty_top10(logits, a),
                "theraset_logits": a, "aa34_logits": b}


def fit_reader(args, output, ckpt):
    source, validation = open_data(args, "val", ckpt)
    seed_everything(args.seed)
    model = DuoEvidence(ckpt["model_args"]).to(args.device)
    model.theraset.load_state_dict(ckpt["model_state"])
    model.eval()
    a, = collect(lambda batch: (model.theraset(batch)["logits"],),
                 batches(validation, args.batch_size), args.device, progress=True)
    longest = ckpt["encoder"]["max_length"]
    train = batches(BackbonePeptides(source, "train", longest), 128, shuffle=True)
    val = BackbonePeptides(source, "val", longest)
    np.testing.assert_array_equal(val.indices, validation.indices)
    y = torch.from_numpy(source.data.y[val.indices].astype(bool))
    optimizer = torch.optim.AdamW(model.aa34.parameters(), lr=3e-4, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.reader_epochs)
    cuts = torch.arange(4, 37) / 40
    best = None
    for epoch in range(1, args.reader_epochs + 1):
        loss = pass_train(model.aa34, train, args.device, optimizer, source.statistics["labels"], .3)
        schedule.step()
        model.eval()
        b, = collect(lambda batch: (model.aa34(batch),), batches(val, 128), args.device)
        stacked = torch.stack((a, b))
        epoch_best = None
        for cut in cuts:
            offsets = torch.tensor([ckpt["threshold"], float(cut)]).logit()
            scores = center_evidence(stacked, offsets).mean(0)
            raw = scores >= 0
            pred = empty_top10(scores, a)
            key = (float(row_jaccard(y, pred).mean()), float(row_jaccard(y, raw).mean()), -abs(float(cut)-.5))
            if epoch_best is None or key > epoch_best[0]:
                epoch_best = key, epoch, float(cut)
        if best is None or epoch_best[0] > best[0]:
            best = epoch_best
            weights = {k: v.detach().cpu().clone() for k, v in model.aa34.state_dict().items()}
        print(f"AA34 {epoch}/{args.reader_epochs} loss={loss:.6f} best_val={best[0][0]:.4%}@{best[1]}", flush=True)
    model.aa34.load_state_dict(weights)
    model.offsets.copy_(torch.tensor([ckpt["threshold"], best[2]]).logit())
    packed = {key: ckpt[key] for key in ("config", "encoder", "statistics", "labels", "model_args", "track")}
    packed.update(model_name="duoevidence", model_state=model.cpu().state_dict(), seed=args.seed,
                  theraset_epoch=ckpt["best_epoch"], aa34_epoch=best[1], thresholds=model.offsets.sigmoid().tolist())
    torch.save(packed, output/"best.pt")
    return best[0][0]


def evaluate(args, checkpoint, split, output):
    packed = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if packed["model_name"] != "duoevidence":
        raise ValueError("Expected a DuoEvidence checkpoint")
    model = DuoEvidence(packed["model_args"], packed.get("aa34_readout_position"), packed.get("aa34_family_mode"))
    model.load_state_dict(packed["model_state"])
    model.to(args.device).eval()
    source, ds = open_data(args, split, packed)
    decision = json.loads(args.decision.read_text(encoding="utf-8")) if args.decision else None

    def read(batch):
        result = model(batch, decision)
        return tuple(result[key] for key in ("logits", "predicted", "theraset_logits", "aa34_logits"))

    logits, pred, theraset, aa34 = collect(read, batches(ds, args.batch_size), args.device, progress=True)
    pred = pred.bool().numpy()
    y = source.data.y[ds.indices].astype(bool)
    prob = logits.sigmoid().numpy()
    result = evaluate_metrics(y, prob, train_support=source.statistics["labels"]["support"],
                              function_projection=source.statistics["function_projection"], predicted=pred)
    result.update(split=split, rows=len(y), parameters=sum(p.numel() for p in model.parameters()),
                  raw_acc=float(row_jaccard(torch.from_numpy(y), logits>=0).mean()),
                  zero_hit=int((~(pred & y).any(1)).sum()), thresholds=packed["thresholds"],
                  seed=packed["seed"], theraset_epoch=packed["theraset_epoch"], aa34_epoch=packed["aa34_epoch"])
    if decision is not None:
        result["decision"] = decision
    if "aa34_readout_position" in packed:
        result["aa34_readout_position"] = packed["aa34_readout_position"]
    if "aa34_family_mode" in packed:
        result["aa34_family_mode"] = packed["aa34_family_mode"]
    write_json(output/"results.json", result)
    np.savez_compressed(output/f"{split}_predictions.npz", row_id=source.data.rows[ds.indices],
                        cluster_id=source.data.clusters[ds.indices], label_names=source.data.labels,
                        targets=y, probabilities=prob, predicted=pred, predicted_raw=(logits>=0).numpy(),
                        split=split, offsets=model.offsets.cpu().numpy(), theraset_logits=theraset.numpy(),
                        aa34_logits=aa34.numpy())
    print(f"{split} raw={result['raw_acc']:.4%} final={result['sample_jaccard']:.4%}", flush=True)
    return result


def calibrate(predictions, output, seed):
    torch.set_num_threads(1)
    with np.load(predictions) as saved:
        if str(saved["split"]) != "val":
            raise ValueError("Calibration requires original Validation predictions")
        thera, aa34, targets, offsets, baseline_raw, baseline_final = (
            torch.from_numpy(saved[key]) for key in ("theraset_logits", "aa34_logits", "targets",
                                                     "offsets", "predicted_raw", "predicted"))
        _, cluster = np.unique(saved["cluster_id"], return_inverse=True)
    scores = center_evidence(torch.stack((thera, aa34)), offsets).mean(0)
    raw, final = scores >= 0, empty_top10(scores, thera)
    if not torch.equal(raw, baseline_raw) or not torch.equal(final, baseline_final):
        raise ValueError("Export must come from an uncalibrated DuoEvidence evaluation")
    # Keep all rows from each cluster on the same side of each calibration fit.
    clusters = cluster.max() + 1
    if clusters < 5:
        raise ValueError("Five-fold calibration requires at least five clusters")
    fold = np.random.default_rng(seed).permutation(clusters)[cluster] % 5
    target_count = targets.sum(1)

    def summarize(raw, final):
        hits = (targets & final).sum(1)
        recall = hits / target_count.clamp_min(1)
        return {"raw_acc": float(row_jaccard(targets, raw).mean()),
                "sample_jaccard": float(row_jaccard(targets, final).mean()),
                "zero_hit": int((hits == 0).sum()),
                "empty_before_fallback": int((~raw.any(1)).sum()),
                "recall_exactly_two_labels": float(recall[target_count == 2].mean()) if (target_count == 2).any() else None,
                "recall_at_least_three_labels": float(recall[target_count >= 3].mean()) if (target_count >= 3).any() else None}

    base_rule = (.5, float(offsets.mean()))
    candidates = [base_rule] + [
        (weight, tick / 10) for weight in (.25, .375, .5, .625, .75) for tick in range(-5, 14)]
    # Cache scores for selection and recompute decisions only for the selected rules.
    def predict(index):
        if index == 0:
            return raw, final
        weight, cutoff = candidates[index]
        logits = decision_scores(thera, aa34, weight, cutoff)
        return logits >= 0, empty_top10(logits, thera)

    values = np.stack([row_jaccard(targets, predict(i)[1]).numpy() for i in range(len(candidates))])
    report = {"split": "val", "rows": len(targets), "clusters": int(clusters), "seed": seed,
              "baseline": summarize(raw, final)}
    for name, indices in (("threshold_only", [i for i, (weight, _) in enumerate(candidates) if weight == .5]),
                          ("weight_and_threshold", list(range(len(candidates))))):
        held_raw, held_final = torch.empty_like(raw), torch.empty_like(final)
        choices = []
        for k in range(5):
            held = fold == k
            winner = indices[values[indices][:, ~held].mean(1).argmax()]
            raw_k, final_k = predict(winner)
            held_raw[held], held_final[held] = raw_k[held], final_k[held]
            choices.append(candidates[winner])
        metrics = summarize(held_raw, held_final)
        metrics["delta_pp"] = 100 * (metrics["sample_jaccard"] - report["baseline"]["sample_jaccard"])
        metrics["fold_weight_cutoff"] = choices
        report[name] = metrics
        print(f"{name}: {metrics['sample_jaccard']:.4%} ({metrics['delta_pp']:+.4f} pp)", flush=True)
        winner = indices[values[indices].mean(1).argmax()]
        # A baseline winner uses the checkpoint's exact threshold and tie behavior.
        if winner == 0:
            continue
        weight, cutoff = candidates[winner]
        write_json(output/f"{name}_decision.json", {"aa34_weight": weight, "cutoff": cutoff})
    write_json(output/"results.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "calibrate"))
    parser.add_argument("--checkpoint", type=Path, help="Packed model, for evaluate only")
    parser.add_argument("--decision", type=Path, help="Validation-fitted fusion weights and cutoff, for evaluate")
    parser.add_argument("--predictions", type=Path, help="Uncalibrated Validation predictions, for calibrate")
    parser.add_argument("--config", type=Path, default=Path("configs/theraset.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--reader-epochs", type=int, default=25)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    training = args.command == "train"
    if training and (args.split != "val" or args.checkpoint or args.decision or args.predictions):
        parser.error("train starts fresh on Train/Validation")
    if args.command == "evaluate" and (not args.checkpoint or args.predictions):
        parser.error("evaluate requires --checkpoint")
    if args.command == "calibrate" and (not args.predictions or args.split != "val" or args.decision or args.checkpoint):
        parser.error("calibrate requires --predictions from an uncalibrated Validation evaluation")
    output = args.output or Path("runs") / f"duoevidence_{args.command}_s{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    if training:
        cli(["train", "--model", "theraset", "--config", str(args.config),
             "--v5-mode", "baseline", "--task", "fine39",
             "--metric-suite", "jaccard", "--seed", str(args.seed), "--device", args.device,
             "--batch-size", str(args.batch_size), "--accumulation-steps", str(args.accumulation_steps),
             "--precision", "fp32", "--epochs", str(args.epochs), "--data-root", str(args.data_root),
             "--feature-root", str(args.features), "--peptide-feature-root", str(args.peptide_features),
             "--name", str((output/"theraset").resolve())])
        ckpt = read_anchor(output/"theraset/best.pt", "theraset")
        expected = fit_reader(args, output, ckpt)
        result = evaluate(args, output/"best.pt", "val", output)
        print(f"Packed replay delta={(result['sample_jaccard']-expected)*100:+.5f}pp")
    elif args.command == "evaluate":
        evaluate(args, args.checkpoint, args.split, output)
    else:
        calibrate(args.predictions, output, args.seed)


if __name__ == "__main__":
    main()
