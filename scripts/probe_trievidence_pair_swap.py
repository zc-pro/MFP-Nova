"""Compare frozen TriEvidence evidence for one count-preserving label swap."""

import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mfp_nova.models.trievidence import TriEvidence, center_evidence, empty_top10
from mfp_nova.multimodal import independent_functions
from mfp_nova.research import to_device
from mfp_nova.trainer import seed_everything, write_json
from run_trievidence import batches, open_data, row_jaccard


MODES = ("scores", "summary", "windows")
BUDGETS = (50, 100, 200)


def select_actions(centered):
    fused = centered.mean(0)
    raw = fused >= 0
    first = centered.argmax(2)
    disagreement = (first != first[0]).any(0)
    order = fused.argsort(dim=1, descending=True, stable=True)
    incoming = order[:, 1:5]
    available = ~raw.gather(1, incoming)
    eligible = disagreement & raw.any(1) & available.any(1)
    outgoing = fused.masked_fill(~raw, torch.inf).argmin(1)
    return raw, first, eligible, torch.cat((outgoing[:, None], incoming), dim=1), available


def local_evidence(reader, maps, mask, labels):
    # Local linear approximation to each frozen AA34 score around its pooled input.
    with torch.enable_grad():
        pooled = maps.masked_fill(~mask[..., None], -torch.inf).amax(1).detach().requires_grad_()
        selected = reader.classifier(pooled).gather(1, labels)
        directions = [
            torch.autograd.grad(score.sum(), pooled, retain_graph=i < selected.shape[1] - 1)[0]
            for i, score in enumerate(selected.unbind(1))
        ]
    response = torch.bmm(F.normalize(torch.stack(directions, 1), dim=-1), maps.transpose(1, 2))
    peak, positions = response.masked_fill(~mask[:, None], -torch.inf).topk(
        min(3, maps.shape[1]), dim=2)
    valid = peak.isfinite()
    length = mask.sum(1)
    mean = response.masked_fill(~mask[:, None], 0).sum(2) / length[:, None]
    summary = torch.stack((peak[..., 0], peak[..., 0] - mean), dim=-1)
    rows = torch.arange(len(maps), device=maps.device)[:, None, None]
    features = maps[rows, positions]
    relative = positions / (length[:, None, None] - 1).clamp_min(1)
    windows = torch.cat((features, relative[..., None], valid[..., None]), dim=-1)
    windows = windows.masked_fill(~valid[..., None], 0)
    windows = F.pad(windows, (0, 0, 0, 3 - windows.shape[2]))
    return summary, windows.half()


def extract(args, model, checkpoint, split):
    source, dataset = open_data(args, split, checkpoint)
    parts = []
    truth, baseline, eligible_rows, distinct_firsts = [], [], [], []
    offset = 0
    for step, batch in enumerate(batches(dataset, args.batch_size), 1):
        batch = to_device(batch, args.device)
        with torch.no_grad():
            thera, helix = model.anchor_logits(batch)
            aa34, maps = model.aa34(batch, return_maps=True)
            centered = center_evidence(torch.stack((thera, aa34, helix)), model.offsets)
            raw, first, eligible, labels, available = select_actions(centered)
            y = batch["y"].bool()
            final = empty_top10(centered.mean(0), thera)
            if eligible.any():
                rows = eligible.nonzero(as_tuple=True)[0]
                labels = labels[rows]
                scores = centered[:, rows].permute(1, 2, 0).gather(
                    1, labels[..., None].expand(-1, -1, 3))
                present = y[rows].gather(1, labels).long()
                gain = present[:, 1:] - present[:, :1]
                hits = (y[rows] & raw[rows]).sum(1, keepdim=True)
                union = (y[rows] | raw[rows]).sum(1, keepdim=True)
                target = (hits + gain) / (union - gain) - hits / union
                summary, windows = local_evidence(model.aa34, maps[rows], batch["mask"][rows], labels)
                parts.append(tuple(t.cpu() for t in (labels, scores, summary, windows, available[rows], target)))
                eligible_rows.append((rows + offset).cpu())
            truth.append(y.cpu())
            baseline.append(final.cpu())
            distinct_firsts.append((1 + (first[1] != first[0]).long()
                                    + ((first[2] != first[0]) & (first[2] != first[1])).long()).cpu())
            offset += len(y)
        if step == 1 or step % 64 == 0:
            print(f"{split} evidence batch {step}", flush=True)
    print(f"{split} collected {len(dataset)} rows and {sum(len(r) for r in eligible_rows)} eligible swaps", flush=True)
    return (tuple(torch.cat(columns) for columns in zip(*parts)), torch.cat(eligible_rows),
            torch.cat(truth), torch.cat(baseline), torch.cat(distinct_firsts), source, dataset.indices)


class SwapReadout(nn.Module):
    def __init__(self, labels, window_dim, mode):
        super().__init__()
        self.mode = mode
        self.labels = nn.Embedding(labels, 16)
        width = 16 + 3
        if mode != "scores":
            width += 2
        if mode == "windows":
            self.window_features = nn.Linear(window_dim - 2, 16)
            width += 3 * 18
        self.pair = nn.Sequential(nn.Linear(2 * width, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, labels, scores, summary, windows, shuffle_positions=False):
        parts = [self.labels(labels), scores]
        if self.mode != "scores":
            parts.append(summary)
        if self.mode == "windows":
            features, pos, valid = windows.float().split((windows.shape[-1] - 2, 1, 1), dim=-1)
            if shuffle_positions:
                values = pos.reshape(len(pos), -1)
                mask = valid.reshape_as(values).bool()
                permutation = torch.rand(values.shape, device=pos.device).masked_fill(~mask, 2).argsort(1)
                ranks = mask.cumsum(1) - 1
                pos = values.gather(1, permutation.gather(1, ranks.clamp_min(0)))
                pos = (pos * mask).reshape_as(valid)
            projected = F.gelu(self.window_features(features)) * valid
            parts.append(torch.cat((projected, pos, valid), -1).flatten(2))
        evidence = torch.cat(parts, dim=-1)
        removed = evidence[:, :1].expand(-1, evidence.shape[1] - 1, -1)
        return self.pair(torch.cat((removed, evidence[:, 1:]), dim=-1)).squeeze(-1)


def score_rows(train, val, args, labels, mode):
    seed_everything(args.seed)
    head = SwapReadout(labels, train[3].shape[-1], mode).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(torch.utils.data.TensorDataset(*train), batch_size=args.head_batch,
                        shuffle=True, generator=generator)
    for epoch in range(args.epochs):
        loss_sum, terms = 0., 0
        for batch in loader:
            labels, scores, summary, windows, available, target = [v.to(args.device) for v in batch]
            prediction = head(labels, scores, summary, windows)
            loss = F.mse_loss(prediction[available], target[available])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count = int(available.sum())
            terms += count
            loss_sum += float(loss.detach()) * count
        print(f"{mode} epoch {epoch+1}/{args.epochs} MSE={loss_sum/terms:.6f}", flush=True)
    with torch.no_grad():
        loader = DataLoader(torch.utils.data.TensorDataset(*val), batch_size=args.head_batch)
        torch.manual_seed(args.seed)
        for shuffled in ((False, True) if mode == "windows" else (False,)):
            estimates = []
            for labels, scores, summary, windows, available, _ in loader:
                values = head(*(v.to(args.device) for v in (labels, scores, summary, windows)),
                              shuffle_positions=shuffled)
                estimates.append(values.cpu().masked_fill(~available, -torch.inf))
            yield ("windows_shuffled_positions" if shuffled else mode), torch.cat(estimates)


def measure(y, baseline, pred, edited, groups):
    delta = row_jaccard(y, pred) - row_jaccard(y, baseline)
    old_hit = (baseline & y).any(1)
    new_hit = (pred & y).any(1)
    return {name: {"rows": int(mask.sum()), "edited": int((edited & mask).sum()),
                   "acc": float(row_jaccard(y[mask], pred[mask]).mean()),
                   "delta_pp": float(100 * delta[mask].mean()),
                   "improved": int(((delta > 0) & mask).sum()),
                   "regressed": int(((delta < 0) & mask).sum()),
                   "tied_edits": int(((delta == 0) & edited & mask).sum()),
                   "recovered_zero": int((~old_hit & new_hit & mask).sum()),
                   "new_zero": int((old_hit & ~new_hit & mask).sum())}
            for name, mask in groups.items() if mask.any()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--nearest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_args = checkpoint["model_args"]
    if (checkpoint["model_name"] != "trievidence" or checkpoint["track"] != "fine39"
            or model_args.get("candidate_mode") or model_args.get("reader_head", "global") != "global"):
        raise ValueError("Use the original packed Fine39 TriEvidence with the global AA34 reader")
    model = TriEvidence(**model_args)
    model.load_state_dict(checkpoint.pop("model_state"))
    model.to(args.device).eval().requires_grad_(False)
    model.reuse_input_compute = True
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now()
    train, *_ = extract(args, model, checkpoint, "train")
    val, rows, y, baseline, distinct_firsts, source, indices = extract(args, model, checkpoint, "val")
    groups = {"all": torch.ones(len(y), dtype=torch.bool)}
    groups["two_agree"] = distinct_firsts == 2
    groups["all_disagree"] = distinct_firsts == 3
    axes = independent_functions(y.numpy(), source.statistics["function_projection"])
    groups["multifunction"] = torch.from_numpy(axes.sum(1) >= 2)
    row_id = source.data.rows[indices]
    if args.nearest is not None:
        with np.load(args.nearest) as near:
            order = np.argsort(near["row_id"])
            np.testing.assert_array_equal(near["row_id"][order], np.sort(row_id))
            match = np.searchsorted(near["row_id"][order], row_id)
            groups["similarity_lt70"] = torch.from_numpy(near["similarity"][order[match]] < .70)
    oracle = val[-1].masked_fill(~val[-2], -torch.inf).amax(1).clamp_min(0)
    results = {"split": "val", "seed": args.seed, "epochs": args.epochs,
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "baseline_acc": float(row_jaccard(y, baseline).mean()), "eligible_rows": len(rows),
               "oracle_acc": float(row_jaccard(y, baseline).mean() + oracle.sum() / len(y)), "arms": {}}
    actions = {}
    for mode in MODES:
        for name, estimates in score_rows(train, val, args, len(source.data.labels), mode):
            score, winner = estimates.max(1)
            outgoing = val[0][:, 0]
            incoming = val[0].gather(1, winner[:, None] + 1).squeeze(1)
            actions.update({name + "_score": score.numpy(), name + "_in": incoming.numpy()})
            ranked = score.argsort(descending=True, stable=True)
            selections = {str(budget): ranked[:budget] for budget in BUDGETS}
            selections["positive_gain"] = ranked[score[ranked] > 0]
            for budget, chosen in selections.items():
                edited = torch.zeros(len(y), dtype=torch.bool)
                edited[rows[chosen]] = True
                pred = baseline.clone()
                pred[rows[chosen], outgoing[chosen]] = False
                pred[rows[chosen], incoming[chosen]] = True
                results["arms"].setdefault(name, {})[budget] = measure(y, baseline, pred, edited, groups)
                print(name, budget, results["arms"][name][budget]["all"], flush=True)
    results.update(seconds=(datetime.now() - started).total_seconds(), checkpoint=str(args.checkpoint))
    np.savez_compressed(args.output / "val_actions.npz", row_id=row_id[rows.numpy()],
                        outgoing=val[0][:, 0].numpy(), **actions)
    write_json(args.output / "results.json", results)


if __name__ == "__main__":
    main()
