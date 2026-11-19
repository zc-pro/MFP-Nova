"""Fit a local candidate ranker using Fine39 Train OOF scores and observed labels."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from evaluate_fine39_oof_fallback import load_branch_oof
from mfp_nova.data import CachedPeptides, collate_features
from mfp_nova.hypothesis_data import family_folds
from mfp_nova.models.local_candidate import CandidatePairRanker
from mfp_nova.multimodal import MultimodalData
from mfp_nova.research_layers import observed_label_loss
from mfp_nova.trainer import seed_everything


class Samples(Dataset):
    def __init__(self, data, indices, base, direct, cache=None):
        self.y = data.y[indices]
        self.base, self.direct, self.cache = base, direct, cache
        if cache is not None:
            np.testing.assert_array_equal(cache.indices, indices)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.base[index], self.direct[index], self.y[index], self.cache[index][0] if self.cache else None


def collate(samples):
    base, direct, y, residues = zip(*samples)
    base, direct, y = [torch.from_numpy(np.stack(values).astype(np.float32)) for values in (base, direct, y)]
    if residues[0] is None:
        return base, direct, y, None, None
    residues, mask, _ = collate_features(list(zip(residues, y)))
    return base, direct, y, residues, mask


def top5_mask(logits):
    order = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :5]
    return torch.zeros_like(logits, dtype=torch.bool).scatter(1, order, True)


def rank_loss(scores, base, direct, y, statistics):
    observed = y.bool()
    initial_top = top5_mask(base)
    unrecorded_competitors = initial_top & ~observed
    candidates = observed & ~initial_top
    pairs = candidates[:, :, None] & unrecorded_competitors[:, None, :]
    harder = 1 + (~top5_mask(direct)).float()
    difference = scores[:, :, None] - scores[:, None, :]
    pair = (F.softplus(-difference) * pairs * harder[:, :, None]).sum((1, 2))
    pair = (pair / pairs.sum((1, 2)).clamp_min(1)).mean()
    classification = observed_label_loss(scores, y, statistics)
    return classification + .15 * pair + .005 * (scores - base).square().mean()


def row_jaccard(y, predicted):
    return (y & predicted).sum(1) / (y | predicted).sum(1)


def compare(y, base, base_probabilities, direct_probabilities, scores, clusters):
    rows = np.arange(len(y))
    original = base_probabilities >= .5
    empty = ~original.any(1)
    # Starting from the stronger single-model empty fallback, compare at most one swap.
    incumbent = original.copy()
    incumbent[rows[empty], base[empty].argmax(1)] = True
    available = ~incumbent
    add = np.where(available, scores, -np.inf).argmax(1)
    remove = np.where(incumbent, scores, np.inf).argmin(1)
    difference = scores[rows, add] - scores[rows, remove]
    folds = family_folds(clusters, 5)
    thera_top = np.argsort(-base_probabilities, axis=1, kind="stable")[:, :5]
    direct_top = np.argsort(-direct_probabilities, axis=1, kind="stable")[:, :5]
    hard = ~(np.take_along_axis(y, thera_top, axis=1).any(1)
             | np.take_along_axis(y, direct_top, axis=1).any(1))
    candidate_top = np.argsort(-scores, axis=1, kind="stable")[:, :5]
    rank_rescues = np.take_along_axis(y, candidate_top, axis=1).any(1) & hard
    incumbent_jaccard = row_jaccard(y, incumbent)
    old_hit = (y & incumbent).any(1)

    def summarize(predicted):
        jac = row_jaccard(y, predicted)
        delta = jac - incumbent_jaccard
        hit = (y & predicted).any(1)
        return {"acc_pct": float(100 * jac.mean()), "delta_from_empty_top1_pp": float(100 * delta.mean()),
                "actual_hard_hits": int((hard & hit).sum()), "new_hit_rows": int((hit & ~old_hit).sum()),
                "lost_hit_rows": int((~hit & old_hit).sum()), "better_rows": int((delta > 0).sum()),
                "worse_rows": int((delta < 0).sum()), "zero_hit_rows": int((~hit).sum())}

    report = {"hard_rows": int(hard.sum()), "candidate_top5_recovers_on_both_miss": int(rank_rescues.sum()),
              "original": summarize(original), "empty_top1": summarize(incumbent)}
    for name, allowed in (("all_rows", available.any(1)), ("originally_empty", empty)):
        options = (float("inf"), 4., 2., 1., .5, .25, 0.)
        predictions = {}
        for threshold in options:
            edits = rows[allowed & (difference > threshold)]
            prediction = incumbent.copy()
            prediction[edits, add[edits]] = True
            prediction[edits, remove[edits]] = False
            predictions[threshold] = prediction
        result = incumbent.copy()
        thresholds = []
        gain = {key: row_jaccard(y, value) for key, value in predictions.items()}
        for fold in range(5):
            held = folds == fold
            threshold = max(options, key=lambda key: gain[key][~held].mean())
            result[held] = predictions[threshold][held]
            thresholds.append(None if np.isinf(threshold) else threshold)
        report[f"crossfit_{name}"] = {**summarize(result), "thresholds": thresholds,
                                    "edited": int((result != incumbent).any(1).sum())}
    return report


def read_val(data, path, indices):
    with np.load(path) as evidence:
        order = np.argsort(evidence["row_id"])
        row_ids = data.rows[indices]
        pos = order[np.searchsorted(evidence["row_id"][order], row_ids)]
        for expected, key in ((row_ids, "row_id"), (data.y[indices], "targets"), (data.labels, "label_names")):
            np.testing.assert_array_equal(expected, evidence[key] if key == "label_names" else evidence[key][pos])
        return tuple(evidence[key][pos] for key in ("logits", "direct_logits", "probabilities", "direct_probabilities"))


def move(batch, device):
    return tuple(value.to(device) if value is not None else None for value in batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scores", "span"), required=True)
    parser.add_argument("--oof", nargs=3, type=Path, required=True)
    parser.add_argument("--evidence-val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diagnostic-rows", type=int)
    args = parser.parse_args()
    seed_everything(42)
    source = MultimodalData(args.data_root, "fine39")
    data = source.data
    train, base_oof, direct_oof = load_branch_oof(data, args.oof)
    val = data.indices("val")
    base, direct, base_prob, direct_prob = read_val(data, args.evidence_val, val)
    y = data.y[val].astype(bool)
    if not np.isclose(row_jaccard(y, base_prob >= .5).mean(), .6127036236779466):
        raise ValueError("Expected the original 61.27% TheraSet Validation export")
    train_cache = val_cache = None
    if args.mode == "span":
        train_cache = CachedPeptides(data, args.peptide_features, "train")
        val_cache = CachedPeptides(data, args.peptide_features, "val")
        keys = ("name", "resolved_revision", "input_dim", "max_length", "dataset")
        if any(train_cache.meta[key] != val_cache.meta[key] for key in keys):
            raise ValueError("Train and Validation peptide caches use different encoders or datasets")
    train_set = Samples(data, train, base_oof, direct_oof, train_cache)
    val_set = Samples(data, val, base, direct, val_cache)
    if args.diagnostic_rows:
        train_set = Subset(train_set, range(min(len(train_set), args.diagnostic_rows)))
        val_set = Subset(val_set, range(min(len(val_set), args.diagnostic_rows)))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
    width = train_cache.meta["input_dim"] if train_cache is not None else 0
    model_args = {"input_dim": width, "num_labels": len(data.labels), "mode": args.mode}
    model = CandidatePairRanker(**model_args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    args.output_dir.mkdir(parents=True)
    history = []
    scope = slice(len(val_set))
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        model.train()
        for batch in train_loader:
            before, raw, target, feats, mask = move(batch, args.device)
            output = model(before, raw, feats, mask)
            loss = rank_loss(output, before, raw, target, source.statistics["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            outputs = []
            for batch in val_loader:
                before, raw, _, feats, mask = move(batch, args.device)
                outputs.append(model(before, raw, feats, mask).cpu().numpy())
        scores = np.concatenate(outputs)
        result = compare(y[scope], base[scope], base_prob[scope], direct_prob[scope],
                         scores, data.clusters[val][scope])
        history.append({"epoch": epoch, "seconds": perf_counter() - start, **result})
        print(json.dumps(history[-1]), flush=True)
    path = args.output_dir
    report = {"mode": args.mode, "train_rows": len(train_set), "validation_rows": len(val_set),
              "diagnostic_rows": args.diagnostic_rows, "history": history}
    (path / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_args": model_args, "model_state": model.cpu().state_dict(),
                "labels": data.labels}, path / "candidate_pair_ranker.pt")
    np.savez_compressed(path / "val_logits.npz", row_id=data.rows[val][scope], logits=scores,
                        targets=y[scope], label_names=data.labels)


if __name__ == "__main__":
    main()
