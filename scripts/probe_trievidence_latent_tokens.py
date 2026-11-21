"""Compare pooled and label-query reads over frozen HelixSet tokens by cluster."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F

from mfp_nova.models.trievidence import empty_top10, fuse_evidence
from probe_trievidence_latent import top


class Reader(nn.Module):
    def __init__(self, input_dim, labels, width=48):
        super().__init__()
        self.scale = width ** -.5
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU())
        self.query = nn.Parameter(torch.randn(labels, width) * .1)

    def forward(self, tokens):
        interactions = self.project(tokens.float()) @ self.query.T * self.scale
        read = (interactions.softmax(dim=1) * interactions).sum(1)
        return torch.tanh(read)


def probe(predictions, features, nearest):
    cfg = json.loads(predictions.with_name("results.json").read_text(encoding="utf-8"))
    with np.load(predictions) as p, np.load(features) as f, np.load(nearest) as n:
        if str(p["split"]) != "val":
            raise ValueError("Expected Validation exports")
        rows = p["row_id"]
        np.testing.assert_array_equal(rows, f["row_id"])
        order = np.argsort(n["row_id"])
        np.testing.assert_array_equal(rows, n["row_id"][order])
        low = n["similarity"][order] < .7
        y, clusters = p["targets"].astype(bool), p["cluster_id"]
        a, b, h = [torch.from_numpy(p[key + "_logits"]) for key in ("theraset", "aa34", "latent")]
        tokens = torch.from_numpy(f["latent_tokens"])
        raw_reference, final_reference = p["predicted_raw"], p["predicted"]
    if tokens.ndim != 3 or tokens.shape[:2] != (len(rows), 32):
        raise ValueError(f"Expected 32 latent tokens per row, got {tuple(tokens.shape)}")
    offsets = torch.tensor(cfg["branch_thresholds"]).logit()
    pooled = tokens.float().mean(1, keepdim=True)
    targets = torch.from_numpy(y.astype(np.float32))
    branches = {"original": h, "mean_pool": h.clone(), "label_query": h.clone()}
    for fold, (train, held) in enumerate(GroupKFold(5).split(y, groups=clusters), 1):
        for name, states in (("mean_pool", pooled), ("label_query", tokens)):
            torch.manual_seed(42 + fold)
            net = Reader(states.shape[-1], y.shape[1])
            optimizer = torch.optim.AdamW(net.parameters(), lr=.002)
            for _ in range(12):
                for indices in torch.randperm(len(train)).split(128):
                    ix = train[indices.numpy()]
                    optimizer.zero_grad(set_to_none=True)
                    loss = F.binary_cross_entropy_with_logits(h[ix] + net(states[ix]), targets[ix])
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                for ix in np.array_split(held, (len(held) + 127) // 128):
                    branches[name][ix] += net(states[ix])
        print(f"Finished fold {fold}/5", flush=True)

    def jac(pred):
        return (pred & y).sum(1) / np.maximum(1, (pred | y).sum(1))

    grouped = {"all": np.ones(len(y), bool), "similarity_lt70": low, "multilabel": y.sum(1) >= 2}
    report = {"scope": "Exploratory cluster cross-fit on original Validation", "groups": {}}
    results = {}
    for name, values in branches.items():
        fused = fuse_evidence(a, b, values, offsets)
        raw, final = (fused >= 0).numpy(), empty_top10(fused, a).numpy()
        if name == "original":
            np.testing.assert_array_equal(raw, raw_reference)
            np.testing.assert_array_equal(final, final_reference)
        pools = {k: top(a.numpy(), k) | top(b.numpy(), k) | top(values.numpy(), k) for k in (5, 10)}
        results[name] = (raw, final, pools)
    for group, mask in grouped.items():
        table = {"rows": int(mask.sum())}
        for name, (raw, final, pools) in results.items():
            table[name] = {
                "raw_acc_pct": float(100 * jac(raw)[mask].mean()),
                "fallback_acc_pct": float(100 * jac(final)[mask].mean()),
                "fallback_zero_hits": int((mask & ~(final & y).any(1)).sum()),
            }
            for k, pool in pools.items():
                ref = results["original"][2][k]
                hit, ref_hit = (y & pool).any(1), (y & ref).any(1)
                table[name][f"top{k}"] = {
                    "pool_misses": int((mask & ~hit).sum()),
                    "rescued": int((mask & hit & ~ref_hit).sum()),
                    "lost": int((mask & ~hit & ref_hit).sum()),
                    "net_recorded_labels": int((y & pool)[mask].sum() - (y & ref)[mask].sum()),
                }
        _, cluster = np.unique(clusters[mask], return_inverse=True)
        sizes = np.bincount(cluster)
        draws = np.random.default_rng(42).integers(len(sizes), size=(2000, len(sizes)))
        denominator = sizes[draws].sum(1)
        for reference in ("original", "mean_pool"):
            comparisons = {}
            for i, metric in enumerate(("raw", "fallback")):
                delta = (jac(results["label_query"][i]) - jac(results[reference][i]))[mask]
                totals = np.bincount(cluster, weights=delta)
                comparisons[metric] = {"delta_pp": float(100 * delta.mean()),
                                       "paired_cluster_ci95_pp": np.quantile(
                                           100 * totals[draws].sum(1) / denominator, [.025, .975]).tolist()}
            table[f"label_query_vs_{reference}"] = comparisons
        report["groups"][group] = table
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("nearest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = probe(args.predictions, args.tokens, args.nearest)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(args.output.resolve())
