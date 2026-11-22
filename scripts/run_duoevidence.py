"""Fine39 TheraSet + AA34. Train both readers from random initialization."""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mfp_nova.cli import main as cli
from mfp_nova.metrics import evaluate_metrics
from mfp_nova.models.theraset import TheraSet
from mfp_nova.models.trievidence import AA34Reader, center_evidence, empty_top10
from mfp_nova.multimodal import BackbonePeptides
from mfp_nova.trainer import seed_everything, write_json
from run_trievidence import batches, collect, open_data, pass_train, read_anchor, row_jaccard


class DuoEvidence(nn.Module):
    def __init__(self, thera_args):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.theraset = TheraSet(**thera_args)
        self.aa34 = AA34Reader(thera_args["statistics"])
        self.register_buffer("offsets", torch.zeros(2))

    def forward(self, batch):
        a = self.theraset(batch)["logits"]
        b = self.aa34(batch)
        logits = center_evidence(torch.stack((a, b)), self.offsets).mean(0)
        return {"logits": logits, "predicted": empty_top10(logits, a)}


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
    model = DuoEvidence(packed["model_args"])
    model.load_state_dict(packed["model_state"])
    model.to(args.device).eval()
    source, ds = open_data(args, split, packed)

    def read(batch):
        result = model(batch)
        return result["logits"], result["predicted"]

    logits, pred = collect(read, batches(ds, args.batch_size), args.device, progress=True)
    pred = pred.bool().numpy()
    y = source.data.y[ds.indices].astype(bool)
    prob = logits.sigmoid().numpy()
    result = evaluate_metrics(y, prob, train_support=source.statistics["labels"]["support"],
                              function_projection=source.statistics["function_projection"], predicted=pred)
    result.update(split=split, rows=len(y), parameters=sum(p.numel() for p in model.parameters()),
                  raw_acc=float(row_jaccard(torch.from_numpy(y), logits>=0).mean()),
                  zero_hit=int((~(pred & y).any(1)).sum()), thresholds=packed["thresholds"],
                  seed=packed["seed"], theraset_epoch=packed["theraset_epoch"], aa34_epoch=packed["aa34_epoch"])
    write_json(output/"results.json", result)
    np.savez_compressed(output/f"{split}_predictions.npz", row_id=source.data.rows[ds.indices],
                        cluster_id=source.data.clusters[ds.indices], label_names=source.data.labels,
                        targets=y, probabilities=prob, predicted=pred, predicted_raw=(logits>=0).numpy())
    print(f"{split} raw={result['raw_acc']:.4%} final={result['sample_jaccard']:.4%}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("--checkpoint", type=Path, help="Packed model, for evaluate only")
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
    if training == (args.checkpoint is not None) or (training and args.split != "val"):
        parser.error("train starts fresh on Train/Validation; evaluate requires --checkpoint")
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
    else:
        evaluate(args, args.checkpoint, args.split, output)


if __name__ == "__main__":
    main()
