"""Isolate the value of local span evidence for TheraSet swap decisions."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate_fine39_oof_fallback import load_branch_oof
from train_fine39_action_gain import actions, assess, ranks
from train_fine39_pair_ranker import read_val, row_jaccard
from mfp_nova.data import CachedPeptides, PeptideData, collate_features
from mfp_nova.hypothesis_data import family_folds
from mfp_nova.models.local_candidate import LocalCandidateHead
from mfp_nova.trainer import seed_everything


class SwapBatch(Dataset):
    def __init__(self, pack, cache=None):
        _, _, gain, rows, _, _, add, incumbent = pack
        self.indices = np.flatnonzero(~add)
        self.indices = self.indices[np.argsort(rows[self.indices], kind="stable")]
        self.count = np.bincount(rows[self.indices], minlength=len(incumbent))
        self.offset = np.concatenate(([0], np.cumsum(self.count)))
        self.rows = np.flatnonzero(self.count)
        self.cache = cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        selected = self.indices[self.offset[row]:self.offset[row + 1]]
        if self.cache is None:
            return selected, None
        return selected, self.cache[row]


class BatchInputs:
    def __init__(self, pack, mean, scale):
        features, _, gain, _, candidate, remove, _, _ = pack
        features = ((features - mean) / scale).astype(np.float32)
        # Add actions carry missing removal features. The loader evaluates swaps only.
        self.features = torch.from_numpy(features)
        self.gain = torch.tensor(gain, dtype=torch.float32)
        self.candidate = torch.from_numpy(candidate)
        self.remove = torch.from_numpy(remove)

    def __call__(self, samples):
        actions, cached = zip(*samples)
        count = np.array([len(items) for items in actions])
        groups = torch.from_numpy(np.repeat(np.arange(len(count)), count))
        weights = torch.as_tensor(np.repeat(1 / count, count), dtype=torch.float32)
        index = np.concatenate(actions)
        inputs = (self.features[index], self.candidate[index], self.remove[index], groups)
        target = self.gain[index]
        if cached[0] is None:
            return index, inputs, target, weights, None, None
        residues, mask, _ = collate_features(cached)
        return index, inputs, target, weights, residues, mask


class SpanUtility(nn.Module):
    def __init__(self, width, input_dim, num_labels, mode):
        super().__init__()
        # Keep the scoring MLP and its initialization the same in both arms.
        self.predict_gain = nn.Sequential(nn.Linear(width + 2, 48), nn.GELU(),
                                          nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 1))
        self.span = LocalCandidateHead(input_dim, num_labels, mode="span") if mode == "span" else None

    def forward(self, features, candidate, remove, group, residues, mask):
        if self.span is None:
            local = features.new_zeros((len(features), 2))
        else:
            logits = self.span(residues, mask)
            local = torch.stack((logits[group, candidate], logits[group, remove]), dim=1)
        return self.predict_gain(torch.cat((features, local), dim=1)).squeeze(-1)


def run_epoch(model, loader, device, optimizer=None, predictions=None):
    model.train(optimizer is not None)
    for idx, inputs, target, weights, residues, mask in loader:
        inputs = tuple(value.to(device) if value is not None else None for value in (*inputs, residues, mask))
        if optimizer is None:
            with torch.inference_mode():
                predictions[idx] = model(*inputs).cpu().numpy()
        else:
            error = model(*inputs) - target.to(device)
            w = weights.to(device)
            loss = (w * error.square()).sum() / w.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scores", "span"), required=True)
    parser.add_argument("--oof", nargs=3, type=Path, required=True)
    parser.add_argument("--evidence-val", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    seed_everything(42)
    data = PeptideData(args.data_root, "fine39")
    train, base, direct = load_branch_oof(data, args.oof)
    support = np.log1p(data.y[train].sum(0))
    train_pack = actions(base, direct, base >= 0, data.y[train].astype(bool), support)
    val = data.indices("val")
    base, direct, base_prob, direct_prob = read_val(data, args.evidence_val, val)
    y = data.y[val].astype(bool)
    primary = base_prob >= .5
    if not np.isclose(row_jaccard(y, primary).mean(), .6127036236779466):
        raise ValueError("Expected the historical strong TheraSet Validation export")
    val_pack = actions(base, direct, primary, y, support)
    clusters = data.clusters[val]
    folds = family_folds(clusters, 5)
    hard = ~((y & (ranks(base_prob) <= 5)).any(1) | (y & (ranks(direct_prob) <= 5)).any(1))
    train_cache = val_cache = None
    width = 0
    if args.mode == "span":
        train_cache = CachedPeptides(data, args.peptide_features, "train")
        val_cache = CachedPeptides(data, args.peptide_features, "val")
        width = train_cache.meta["input_dim"]
        for cache, indices in ((train_cache, train), (val_cache, val)):
            np.testing.assert_array_equal(cache.indices, indices)
        keys = ("name", "resolved_revision", "input_dim", "dataset")
        if any(train_cache.meta[key] != val_cache.meta[key] for key in keys):
            raise ValueError("Train and Validation caches use different peptide encoders or datasets")
    swap_features = train_pack[0][~train_pack[-2]]
    mean, scale = swap_features.mean(0), swap_features.std(0).clip(.1)
    model_args = {"width": len(mean), "input_dim": width, "num_labels": len(data.labels), "mode": args.mode}
    model = SpanUtility(**model_args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    loaders = [
        DataLoader(SwapBatch(pack, cache), batch_size=args.batch_size, shuffle=shuffle,
                   generator=torch.Generator().manual_seed(42), collate_fn=BatchInputs(pack, mean, scale))
        for pack, cache, shuffle in ((train_pack, train_cache, True), (val_pack, val_cache, False))
    ]
    out = args.output_dir or Path("runs") / datetime.now().strftime(
        f"fine39_span_action_gain_{args.mode}_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        run_epoch(model, loaders[0], args.device, optimizer=optimizer)
        predictions = np.full(len(val_pack[2]), -np.inf)
        run_epoch(model, loaders[1], args.device, predictions=predictions)
        result = assess(predictions, val_pack, y, folds, clusters, hard, ~val_pack[-2])
        history.append({"epoch": epoch, "seconds": perf_counter() - start, **result})
        print(json.dumps(history[-1]), flush=True)
    report = {"mode": args.mode, "train_rows": len(train), "validation_rows": len(val),
              "train_swaps": len(swap_features), "history": history}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_args": model_args, "model_state": model.cpu().state_dict(),
                "mean": torch.tensor(mean), "scale": torch.tensor(scale)}, out / "utility.pt")
    print(out.resolve())


if __name__ == "__main__":
    main()
