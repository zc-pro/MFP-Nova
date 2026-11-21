"""Train matched fixed and TheraSet-conditioned reads over frozen HelixSet tokens."""

import argparse
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from compare_validation_predictions import paired_cluster_interval
from mfp_nova.models.trievidence import TriEvidence, center_evidence, empty_top10, fuse_evidence
from mfp_nova.multimodal import MultimodalPeptides, independent_functions
from mfp_nova.research import to_device, validate_cache
from mfp_nova.research_layers import observed_label_loss
from mfp_nova.trainer import seed_everything, write_json
from run_trievidence import batches, open_data, row_jaccard


class CrossRead(nn.Module):
    def __init__(self, label_count, state_dim, token_dim, width=32):
        super().__init__()
        self.labels = nn.Parameter(torch.randn(label_count, width) * .02)
        self.query = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, width))
        self.keys = nn.Linear(token_dim, width, bias=False)
        self.values = nn.Linear(token_dim, width, bias=False)
        self.output = nn.Parameter(torch.zeros(label_count, width))

    def forward(self, state, tokens):
        q = self.labels + self.query(state.float())
        tokens = tokens.float()
        attention = (q @ self.keys(tokens).transpose(1, 2) * q.size(-1)**-.5).softmax(-1)
        values = self.values(tokens)
        read = attention @ values - values.mean(1, keepdim=True)
        return (read * self.output).sum(-1)


@torch.no_grad()
def features(model, dataset, args):
    # Preallocate the two large CPU arrays and keep the reference scores in FP32.
    cache = {}
    last_tokens = []
    def capture(_module, _inputs, output):
        last_tokens.append(output)
    hook = model.helixset.latent_blocks[-1].register_forward_hook(capture)
    start = 0
    try:
        loader = batches(dataset, args.batch_size)
        for step, batch in enumerate(loader, 1):
            batch = to_device(batch, args.device)
            t = model.theraset(batch, return_states=True)
            h = model.helixset(batch)["logits"]
            a = model.aa34(batch)
            logits = [t["logits"], a, h]
            mixed = fuse_evidence(*logits, model.offsets)
            ct, ca, ch = center_evidence(torch.stack(logits), model.offsets)
            possible = ((2 * ct + ca) >= 0) | ((2 * ch + ca) >= 0)
            part = dict(state=t["evidence_state"].half(), tokens=last_tokens.pop().half(),
                        base=mixed, anchor=t["logits"], possible=possible, y=batch["y"])
            end = start + len(h)
            for name, tensor in part.items():
                if step == 1:
                    cache[name] = torch.empty((len(dataset), *tensor.shape[1:]), dtype=tensor.dtype)
                cache[name][start:end].copy_(tensor)
            start = end
            if step == 1 or step % 128 == 0 or step == len(loader):
                print(f"Frozen forward {step}/{len(loader)}", flush=True)
    finally:
        hook.remove()
    return cache


@torch.no_grad()
def predict(head, cache, mean, conditioned, device):
    out = []
    for idx in torch.arange(len(cache["y"])).split(128):
        state = cache["state"][idx].to(device) if conditioned else mean
        delta = head(state, cache["tokens"][idx].to(device)).cpu()
        out.append(cache["base"][idx] + delta)
    return torch.cat(out)


def compare(outputs, val, groups, clusters):
    truth = val["y"].bool()
    scores, decisions = {}, {}
    for arm, logits in outputs.items():
        decisions[arm] = {"raw": logits >= 0, "fallback": empty_top10(logits, val["anchor"])}
        scores[arm] = {k: row_jaccard(truth, p).numpy() for k, p in decisions[arm].items()}
    report = {}
    for group, selected in groups.items():
        if not selected.any():
            continue
        table = {"rows": int(selected.sum()), "acc_pct": {}, "paired": {}}
        for arm in outputs:
            table["acc_pct"][arm] = {k: 100 * float(s[selected].mean(dtype=np.float64))
                                     for k, s in scores[arm].items()}
        for arm, ref in (("fixed", "original"), ("conditioned", "original"), ("conditioned", "fixed"),
                         ("conditioned", "conditioned_at_mean")):
            metrics = {}
            for metric in ("raw", "fallback"):
                delta = (scores[arm][metric] - scores[ref][metric])[selected]
                p, r = (decisions[n][metric][selected] for n in (arm, ref))
                y = truth[selected]
                hit, old = (p & y).any(1), (r & y).any(1)
                outside = y & ~val["possible"][selected]
                metrics[metric] = dict(delta_pp=100 * float(delta.mean(dtype=np.float64)),
                                       cluster_ci95_pp=paired_cluster_interval(delta, clusters[selected]),
                                       zero_hit_rescued=int((hit & ~old).sum()), new_zero_hit=int((old & ~hit).sum()),
                                       outside_convex_gained=int((outside & p & ~r).sum()),
                                       outside_convex_lost=int((outside & r & ~p).sum()))
            table["paired"][f"{arm}_vs_{ref}"] = metrics
        report[group] = table
    return report


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TriEvidence(**ckpt["model_args"])
    model.load_state_dict(ckpt.pop("model_state"))
    model.to(args.device).eval()
    source, val_data = open_data(args, "val", ckpt)
    print("Extract Validation", flush=True)
    val = features(model, val_data, args)
    rows = val_data.indices
    clusters = source.data.clusters[rows]
    with np.load(args.checkpoint.with_name("val_predictions.npz")) as ref:
        for key, expected in (("row_id", source.data.rows[rows]), ("cluster_id", clusters),
                              ("label_names", source.data.labels), ("targets", val["y"].numpy()),
                              ("predicted_raw", (val["base"] >= 0).numpy()),
                              ("predicted", empty_top10(val["base"], val["anchor"]).numpy())):
            np.testing.assert_array_equal(ref[key], expected, err_msg=f"Validation baseline mismatch: {key}")
    train_data = MultimodalPeptides(source, args.features, "train", peptide_feature_root=args.peptide_features)
    validate_cache(train_data, ckpt["config"], ckpt["encoder"])
    print("Extract Train", flush=True)
    train = features(model, train_data, args)
    del model
    state = train["state"]
    mean = sum(chunk.float().sum(0) for chunk in state.split(256))[None].to(args.device) / len(state)
    initial = CrossRead(state.shape[1], state.shape[-1], train["tokens"].shape[-1]).to(args.device)
    heads = {arm: copy.deepcopy(initial) for arm in ("fixed", "conditioned")}
    optim = {arm: torch.optim.AdamW(net.parameters(), lr=args.lr) for arm, net in heads.items()}
    for epoch in range(1, args.epochs + 1):
        losses = dict.fromkeys(heads, 0.)
        for idx in torch.randperm(len(state)).split(args.head_batch):
            current, tokens, base, y = (train[k][idx].to(args.device) for k in ("state", "tokens", "base", "y"))
            for arm, net in heads.items():
                delta = net(current if arm == "conditioned" else mean, tokens)
                loss = observed_label_loss(base + delta, y.float(), source.statistics["labels"])
                optim[arm].zero_grad(set_to_none=True)
                loss.backward()
                optim[arm].step()
                losses[arm] += float(loss.detach()) * len(idx)
        print(f"Epoch {epoch}/{args.epochs} Train loss " +
              " ".join(f"{k}={v/len(state):.6f}" for k, v in losses.items()), flush=True)
    outputs = {"original": val["base"]}
    for arm, head in heads.items():
        outputs[arm] = predict(head, val, mean, arm == "conditioned", args.device)
    outputs["conditioned_at_mean"] = predict(heads["conditioned"], val, mean, False, args.device)
    groups = {"all": np.ones(len(rows), bool),
              "multifunction": independent_functions(val["y"].numpy(),
                                                     source.statistics["function_projection"]).sum(1) >= 2}
    if args.nearest:
        with np.load(args.nearest) as n:
            order = np.argsort(n["row_id"])
            np.testing.assert_array_equal(n["row_id"][order], source.data.rows[rows])
            groups["similarity_lt70"] = n["similarity"][order] < .7
    report = dict(checkpoint=str(args.checkpoint), split="val", seed=args.seed, train_rows=len(state),
                  fixed_final_epoch=args.epochs, head_batch=args.head_batch, extraction_batch=args.batch_size,
                  learning_rate=args.lr, width=initial.output.shape[-1], head_parameters=sum(p.numel() for p in initial.parameters()),
                  groups=compare(outputs, val, groups, clusters))
    write_json(args.output / "results.json", report)
    torch.save({"train_mean": mean.cpu(), **{k: v.cpu().state_dict() for k, v in heads.items()}},
               args.output / "heads.pt")
    np.savez_compressed(args.output / "val_scores.npz", row_id=source.data.rows[rows],
                        **{k: v.numpy() for k, v in outputs.items()})
    print(report["groups"]["all"]["acc_pct"], flush=True)
    print(args.output / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--nearest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path("runs") / datetime.now().strftime("trievidence_cross_read_%Y%m%d_%H%M%S"))
    run(parser.parse_args())
