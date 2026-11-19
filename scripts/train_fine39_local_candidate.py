"""Compare a pooled and a multiscale label-query candidate head on Fine39."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Subset

from evaluate_fine39_oof_fallback import load_branch_oof
from mfp_nova.data import CachedPeptides, collate_features
from mfp_nova.models.local_candidate import LocalCandidateHead
from mfp_nova.multimodal import MultimodalData
from mfp_nova.research_layers import observed_label_loss
from mfp_nova.trainer import seed_everything


def ranks(scores):
    return np.argsort(np.argsort(-scores, axis=1, kind="stable"), axis=1) + 1


class WeightedCache(CachedPeptides):
    def __getitem__(self, index):
        residues, y = super().__getitem__(index)
        return residues, y, self.multipliers[index]


def collate_train(samples):
    residues, y, weight = zip(*samples)
    return (*collate_features(list(zip(residues, y))), torch.as_tensor(np.stack(weight)))


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    return np.concatenate([model(x.to(device), mask.to(device)).float().cpu().numpy() for x, mask, _ in loader])


def summarize(base, direct, candidate, y):
    scores = ranks(base)
    head = ranks(candidate)
    base_top5 = (y & (scores <= 5)).any(1)
    direct_top5 = (y & (ranks(direct) <= 5)).any(1)
    head_top5 = (y & (head <= 5)).any(1)
    harder = ~base_top5 & ~direct_top5
    original = base >= .5
    rows = np.arange(len(y))
    empty = ~original.any(1)
    # Fixed, parameter-free policies. The first comparison preserves every row's label count.
    combined = 1 / (scores + 3) + 1 / (head + 3)
    reordered = ranks(combined) <= original.sum(1, keepdims=True)
    fill_base, fill_head = original.copy(), original.copy()
    fill_base[rows[empty], base[empty].argmax(1)] = True
    fill_head[rows[empty], candidate[empty].argmax(1)] = True

    def metrics(pred):
        hits = (pred & y).sum(1)
        return {"acc_pct": float(100 * (hits / (pred | y).sum(1)).mean()),
                "zero_hit_rows": int((hits == 0).sum()),
                "harder_actual_hits": int((harder & (hits > 0)).sum()),
                "base_hit_lost": int(((hits == 0) & (original & y).any(1)).sum())}

    active = y.any(0)
    return {
        "rows": len(y), "both_theraset_heads_top5_miss": int(harder.sum()),
        "head_top5_hits": int(head_top5.sum()),
        "head_top5_recovers_on_both_miss": int((head_top5 & harder).sum()),
        "head_macro_ap": float(average_precision_score(y[:, active], candidate[:, active])),
        "policies": {"original": metrics(original), "fixed_count_rank_fusion": metrics(reordered),
                     "fill_empty_base_top1": metrics(fill_base), "fill_empty_candidate_top1": metrics(fill_head)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("span", "pool"), required=True)
    parser.add_argument("--oof", nargs=3, required=True, type=Path)
    parser.add_argument("--evidence-val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diagnostic-rows", type=int, help="Limit both splits for a plumbing check")
    args = parser.parse_args()
    seed_everything(42)
    source = MultimodalData(args.data_root, "fine39")
    data = source.data
    train = WeightedCache(data, args.peptide_features, "train")
    val = CachedPeptides(data, args.peptide_features, "val")
    encoder_keys = ("name", "resolved_revision", "input_dim", "max_length", "dataset")
    if any(train.meta[key] != val.meta[key] for key in encoder_keys):
        raise ValueError("Train and Validation peptide feature caches use different encoders or datasets")
    rows, base_oof, direct_oof = load_branch_oof(data, args.oof)
    np.testing.assert_array_equal(rows, train.indices)
    difficult = (ranks(base_oof) > 5) & (ranks(direct_oof) > 5)
    train.multipliers = 1 + difficult.astype(np.float32)
    with np.load(args.evidence_val) as evidence:
        row_order = np.argsort(evidence["row_id"])
        target_ids = data.rows[val.indices]
        index = row_order[np.searchsorted(evidence["row_id"][row_order], target_ids)]
        np.testing.assert_array_equal(evidence["row_id"][index], target_ids)
        np.testing.assert_array_equal(evidence["targets"][index], val.targets)
        np.testing.assert_array_equal(evidence["label_names"], data.labels)
        base, direct = (evidence[key][index] for key in ("probabilities", "direct_probabilities"))
    y = val.targets.astype(bool)
    base_prediction = base >= .5
    base_acc = (base_prediction & y).sum(1) / (base_prediction | y).sum(1)
    if not np.isclose(base_acc.mean(), .6127036236779466):
        raise ValueError("Expected the full original 61.27% TheraSet Validation export")
    train_set, val_set = train, val
    if args.diagnostic_rows:
        train_set = Subset(train, range(min(len(train), args.diagnostic_rows)))
        val_set = Subset(val, range(min(len(val), args.diagnostic_rows)))
        length = len(val_set)
        base, direct, y = base[:length], direct[:length], y[:length]
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_train)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_features)
    model_args = {"input_dim": train.meta["input_dim"], "num_labels": len(data.labels), "mode": args.mode}
    model = LocalCandidateHead(**model_args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    args.output_dir.mkdir(parents=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        started = perf_counter()
        model.train()
        for residues, mask, targets, weight in train_loader:
            logits = model(residues.to(args.device), mask.to(args.device))
            loss = observed_label_loss(logits, targets.to(args.device), source.statistics["labels"],
                                       positive_multiplier=weight.to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        logits = predict(model, val_loader, args.device)
        metrics = summarize(base, direct, logits, y)
        history.append({"epoch": epoch, "seconds": perf_counter() - started, **metrics})
        print(json.dumps(history[-1]), flush=True)
    result = {"mode": args.mode, "seed": 42, "train_rows": len(train_set),
              "diagnostic_rows": args.diagnostic_rows, "parameters": sum(p.numel() for p in model.parameters()),
              "history": history}
    path = args.output_dir
    (path / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_args": model_args, "model_state": model.cpu().state_dict(),
                "labels": data.labels}, path / "candidate_head.pt")
    np.savez_compressed(path / "val_logits.npz", logits=logits, row_id=data.rows[val.indices[:len(y)]],
                        targets=y, label_names=data.labels)


if __name__ == "__main__":
    main()
