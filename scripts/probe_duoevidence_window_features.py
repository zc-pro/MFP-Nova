"""Train a matched window read on frozen AA34 or HelixSet residue features."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from mfp_nova.models.helixset import HelixSet
from mfp_nova.models.trievidence import AA34Reader, center_evidence, empty_top10
from mfp_nova.multimodal import BackbonePeptides
from mfp_nova.research import to_device
from mfp_nova.research_layers import observed_label_loss
from mfp_nova.trainer import seed_everything, write_json
from run_trievidence import batches, open_data, read_anchor, row_jaccard


class WindowProbe(nn.Module):
    def __init__(self, width, labels):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.project = nn.Linear(width, 96, bias=False)
        self.score = nn.Linear(32, labels, bias=False)
        self.gain = nn.Parameter(torch.zeros(labels))

    def forward(self, features, mask):
        valid = mask[:, None]
        tokens = self.project(self.norm(features)).transpose(1, 2).masked_fill(~valid, 0)
        evidence = []
        for width, view in zip((3, 7, 15), tokens.chunk(3, dim=1)):
            count = F.avg_pool1d(valid.float(), width, stride=1, padding=width // 2).clamp_min(1 / width)
            mean = F.avg_pool1d(view, width, stride=1, padding=width // 2) / count
            response = self.score(F.gelu(mean.transpose(1, 2)))
            attention = response.masked_fill(~mask[..., None], -torch.inf).softmax(1)
            evidence.append((response * attention).sum(1))
        return self.gain * torch.stack(evidence).mean(0)


@torch.no_grad()
def extract(dataset, aa34, helix, args, longest):
    saved, captured = [], []
    hook = None
    if helix is not None:
        hook = helix.pyramid.register_forward_pre_hook(lambda _, inputs: captured.append(inputs[0]))
    try:
        for step, batch in enumerate(batches(dataset, args.feature_batch), 1):
            batch = to_device(batch, args.device)
            base, maps = aa34(batch, return_maps=True)
            if helix is not None:
                helix(batch)
                maps = captured.pop()
            pad = longest - maps.size(1)
            saved.append((F.pad(maps, (0, 0, 0, pad)).to(device="cpu", dtype=torch.float16),
                          F.pad(batch["mask"], (0, pad)).cpu(), base.cpu(), batch["y"].cpu()))
            if step == 1 or step % 128 == 0:
                print(f"Extract batch {step}", flush=True)
    finally:
        if hook is not None:
            hook.remove()
    return TensorDataset(*(torch.cat(parts) for parts in zip(*saved)))


def run(args):
    checkpoint = read_anchor(args.duo, "duoevidence")
    source, full_val = open_data(args, "val", checkpoint)
    stats = source.statistics["labels"]
    longest = checkpoint["encoder"]["max_length"]
    aa34 = AA34Reader(stats).to(args.device).eval()
    aa34.load_state_dict({k.removeprefix("aa34."): v for k, v in checkpoint["model_state"].items()
                          if k.startswith("aa34.")})
    helix = None
    if args.source == "helix":
        packed = read_anchor(args.helix, "trievidence")
        np.testing.assert_array_equal(packed["labels"], checkpoint["labels"])
        _, train = open_data(args, "train", packed)
        _, val = open_data(args, "val", packed)
        helix = HelixSet(**packed["model_args"]["helixset_args"], direct_only=True).to(args.device).eval()
        helix.load_state_dict({k.removeprefix("helixset."): v for k, v in packed["model_state"].items()
                              if k.startswith("helixset.")})
        np.testing.assert_array_equal(train.indices, source.data.indices("train"))
    else:
        train, val = (BackbonePeptides(source, split, longest) for split in ("train", "val"))
    np.testing.assert_array_equal(val.indices, full_val.indices)
    y = source.data.y[val.indices].astype(bool)
    offsets = checkpoint["model_state"]["offsets"]
    with np.load(args.predictions) as original:
        for key, expected in (("split", "val"), ("row_id", source.data.rows[val.indices]),
                              ("targets", y), ("cluster_id", source.data.clusters[val.indices]),
                              ("label_names", source.data.labels), ("offsets", offsets.numpy())):
            np.testing.assert_array_equal(original[key], expected)
        thera = torch.from_numpy(original["theraset_logits"])
        original_aa = original["aa34_logits"]
        initial_pred = original["predicted"]
    args.output.mkdir(parents=True, exist_ok=False)
    val_features = extract(val, aa34, helix, args, longest)
    np.testing.assert_allclose(val_features.tensors[2].numpy(), original_aa, rtol=1e-4, atol=1e-3)
    train_features = extract(train, aa34, helix, args, longest)
    del aa34, helix
    seed_everything(args.seed)
    head = WindowProbe(train_features.tensors[0].shape[-1], len(stats["labels"])).to(args.device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    valid = DataLoader(val_features, batch_size=args.batch_size)
    fit = DataLoader(train_features, batch_size=args.batch_size, shuffle=True)
    history = []
    for epoch in range(args.epochs + 1):
        if epoch:
            head.train()
            for features, mask, base, targets in fit:
                features, mask, base, targets = (v.to(args.device) for v in (features.float(), mask, base, targets))
                opt.zero_grad(set_to_none=True)
                loss = observed_label_loss(base + head(features, mask), targets, stats, .3)
                loss.backward()
                opt.step()
        head.eval()
        with torch.no_grad():
            correction = torch.cat([head(f.float().to(args.device), m.to(args.device)).cpu()
                                    for f, m, _, _ in valid])
        reader = torch.from_numpy(original_aa) + correction
        fused = center_evidence(torch.stack((thera, reader)), offsets).mean(0)
        pred = empty_top10(fused, thera).numpy()
        if epoch == 0:
            np.testing.assert_array_equal(pred, initial_pred)
        record = {"epoch": epoch, "sample_jaccard": float(row_jaccard(torch.from_numpy(y), torch.from_numpy(pred)).mean()),
                  "zero_hit": int((~(y & pred).any(1)).sum())}
        print(record, flush=True)
        history.append(record)
    write_json(args.output / "results.json", {"source": args.source, "history": history,
                                               "trainable_parameters": sum(p.numel() for p in head.parameters())})
    torch.save(head.state_dict(), args.output / "reader.pt")
    np.savez_compressed(args.output / "val_predictions.npz", split="val", row_id=source.data.rows[val.indices],
                        cluster_id=source.data.clusters[val.indices], label_names=source.data.labels, targets=y,
                        predicted=pred, predicted_raw=(fused >= 0).numpy(), probabilities=fused.sigmoid().numpy(),
                        offsets=offsets.numpy(), theraset_logits=thera.numpy(), aa34_logits=reader.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duo", type=Path, required=True)
    parser.add_argument("--helix", type=Path)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source", choices=("aa34", "helix"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-batch", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.source == "helix" and args.helix is None:
        parser.error("--helix is required for HelixSet feature extraction")
    run(args)
