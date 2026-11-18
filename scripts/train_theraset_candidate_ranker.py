"""Cross-fit a small candidate ranker on a fixed-epoch Fine39 TheraSet checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from analyze_xbridge_validation import jaccard_rows, nearest_similarity, paired_interval
from mfp_nova.hypothesis_data import family_folds
from mfp_nova.metrics import evaluate_metrics
from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from mfp_nova.research import make_model, restore_model_state, to_device, validate_cache
from mfp_nova.xbridge_data import recovery_diagnostics


class CandidateRanker(nn.Module):
    def __init__(self, labels, state_dim):
        super().__init__()
        self.label = nn.Embedding(labels, 8)
        self.head = nn.Sequential(nn.Linear(11 + state_dim, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, candidates):
        inputs = torch.cat((features, self.label(candidates)), dim=-1)
        return 2 * self.head(inputs).squeeze(-1).tanh()


def pair_loss(base, residual, observed, unrecorded_weight):
    scores = base + residual
    pairs = observed.unsqueeze(-1) & ~observed.unsqueeze(-2)
    difference = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    per_row = (F.softplus(-difference) * pairs).sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)
    return unrecorded_weight * per_row.mean() + .02 * residual.square().mean()


@torch.no_grad()
def collect(model, loader, device):
    logits, features, states = [], [], []
    for batch in loader:
        out = model(to_device(batch, device), return_states=True)
        final, direct = out["logits"], out["direct"]
        logits.append(final.cpu())
        features.append(torch.stack((final, direct, final - direct), dim=-1).cpu())
        states.append(out["evidence_state"].cpu())
    return tuple(torch.cat(parts) for parts in (logits, features, states))


def fit_ranker(features, candidates, base, observed, fit, query, epochs, weight, seed):
    torch.manual_seed(seed)
    model = CandidateRanker(39, features.shape[-1] - 3).to(features.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    for _ in range(epochs):
        for batch in fit[torch.randperm(len(fit), device=fit.device)].split(128):
            residual = model(features[batch], candidates[batch])
            loss = pair_loss(base[batch], residual, observed[batch], weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return model(features[query], candidates[query]), model.cpu().state_dict()


def permute_candidates(logits, candidates, base, residual):
    # Permute the original Top-5 score slots. The 0.50 prediction count and
    # the rest of the label scores stay identical for every peptide.
    order = torch.argsort(base + residual, dim=1, descending=True, stable=True)
    return logits.scatter(1, candidates.gather(1, order), base)


def summarize(logits, targets, baseline, mask, clusters, statistics):
    scores = logits[mask].sigmoid().numpy()
    truth = targets[mask]
    decision = scores >= .5
    projection = statistics["function_projection"]
    metrics = evaluate_metrics(truth, scores, .5, statistics["labels"]["support"], projection)
    before = (truth & baseline[mask]).any(1)
    after = (truth & decision).any(1)
    delta = jaccard_rows(truth, decision) - jaccard_rows(truth, baseline[mask])
    base_top = baseline[mask]
    order = np.argsort(-scores, axis=1, kind="stable")
    evidence = np.asarray(projection, dtype=bool)[order] & np.take_along_axis(truth, order, axis=1)[..., None]
    rank = np.sort(np.where(evidence.any(1), evidence.argmax(1) + 1, scores.shape[1] + 1), axis=1)
    return {
        **{key: metrics[key] for key in ("sample_jaccard", "sample_precision", "coverage", "macro_ap",
                                        "micro_f1", "mfp_function_prediction")},
        **recovery_diagnostics(truth, scores, projection),
        "jaccard_delta": float(delta.mean()), "conditional_cluster_interval": paired_interval(delta, clusters[mask]),
        "zero_hit_rescued": int((~before & after).sum()), "new_zero_hit": int((before & ~after).sum()),
        "changed_sets": int((decision != base_top).any(1).sum()),
        "second_axis_mean_rank": float(rank[rank[:, 1] <= scores.shape[1], 1].mean()) if (rank[:, 1] <= scores.shape[1]).any() else None,
        "third_axis_mean_rank": float(rank[rank[:, 2] <= scores.shape[1], 2].mean()) if (rank[:, 2] <= scores.shape[1]).any() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Fixed-epoch bridge-off last.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head-epochs", type=int, default=12)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config, statistics = ckpt["config"], ckpt["statistics"]
    settings, paths = config["train"], config["data"]
    fold = settings.get("xbridge_inner_fold")
    if (ckpt["model_name"] != "theraset_bridge" or ckpt["track"] != "fine39" or fold is None
            or config["loss"]["theraset_bridge"] != "off" or ckpt["epoch"] != settings["epochs"]):
        raise ValueError("Use the final last.pt from a completed, fixed-epoch Fine39 bridge-off inner fold")
    model = make_model(ckpt["model_name"], ckpt["model_args"]).to(args.device).eval()
    ema = ckpt["ema_state"]
    if int(ema["n_averaged"]) == 0:
        raise ValueError("The fixed-epoch checkpoint must contain EMA updates")
    restore_model_state(model, {key.removeprefix("module."): value for key, value in ema.items()
                                if key.startswith("module.")})
    checkpoint_epoch = ckpt["epoch"]
    encoder = ckpt["encoder"]
    del ckpt, ema

    source = MultimodalData(paths["root"], "fine39", statistics,
                           localize_terminals=statistics.get("localized_terminal_chemistry", False))
    data = source.data
    pool = data.indices("train")
    mask = family_folds(data.clusters[pool], 5) == fold
    fit, held = pool[~mask], pool[mask]
    dataset = MultimodalPeptides(source, paths["features"], "train",
                                 peptide_feature_root=paths.get("peptide_features"), indices=held)
    validate_cache(dataset, config, encoder)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_multimodal)
    logits, signals, evidence = collect(model, loader, args.device)
    del model
    targets, clusters = data.y[held].astype(bool), data.clusters[held]
    similarity = nearest_similarity(data, held, fit)
    candidates = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :5]
    base = logits.gather(1, candidates)
    observed = torch.from_numpy(targets).gather(1, candidates)
    numeric = signals.gather(1, candidates[..., None].expand(-1, -1, signals.size(-1))) / 4
    states = evidence.gather(1, candidates[..., None].expand(-1, -1, evidence.size(-1)))
    parts = family_folds(clusters, 2)
    baseline = logits.sigmoid().numpy() >= .5
    strata = {"all": np.ones(len(held), bool), "below_50": similarity < .5,
              "below_70": similarity < .7, "at_least_70": similarity >= .7}
    report = {"fold": fold, "checkpoint_epoch": checkpoint_epoch, "seed": settings["seed"],
              "head_epochs": args.head_epochs, "fit_backbone_rows": len(fit), "held_rows": len(held),
              "results": {}}
    variants = {"off": logits}
    weights = {}
    for name, values in (("scores", numeric), ("evidence", torch.cat((numeric, states), dim=-1))):
        residuals = torch.empty_like(base)
        weights[name] = []
        inputs = [value.to(args.device) for value in (values, candidates, base, observed)]
        for part in (0, 1):
            query = np.flatnonzero(parts == part)
            fit_head = torch.as_tensor(np.flatnonzero(parts != part), device=args.device)
            correction, weights_part = fit_ranker(*inputs, fit_head, torch.as_tensor(query, device=args.device),
                                                  args.head_epochs, config["loss"]["unlabeled_weight"],
                                                  settings["seed"] + part)
            residuals[query] = correction.cpu()
            weights[name].append(weights_part)
        variants[name] = permute_candidates(logits, candidates, base, residuals)

    for name, ranked in variants.items():
        report["results"][name] = {key: summarize(ranked, targets, baseline, subset, clusters, statistics)
                                   for key, subset in strata.items() if subset.any()}
    args.output.mkdir(parents=True)
    torch.save(weights, args.output / "rankers.pt")
    np.savez_compressed(args.output / "predictions.npz", row_id=data.rows[held], cluster_id=clusters,
                        head_query_half=parts, targets=targets, similarity=similarity,
                        label_names=data.labels, **{name: tensor.numpy() for name, tensor in variants.items()})
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: result["all"] for name, result in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
