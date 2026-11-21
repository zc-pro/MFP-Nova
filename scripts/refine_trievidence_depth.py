"""Screen or recover a compressed latent branch from a packed TriEvidence checkpoint."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from mfp_nova.models.trievidence import TriEvidence, empty_top10, fuse_evidence
from mfp_nova.multimodal import MultimodalPeptides
from mfp_nova.research import to_device, validate_cache
from mfp_nova.research_layers import observed_label_loss, unlabeled_weights
from mfp_nova.trainer import seed_everything, write_json
from run_trievidence import batches, collect, diagnostics, open_data, row_jaccard


def svd_latent_ffn(state, depth, rank):
    result = dict(state)
    for block in range(depth):
        for index in (1, 4):
            prefix = f"helixset.latent_blocks.{block}.process.ff.net.{index}"
            weight = result.pop(prefix + ".weight")
            u, singular, vh = torch.linalg.svd(weight.float(), full_matrices=False)
            scale = singular[:rank].sqrt()
            result[prefix + ".0.weight"] = (scale[:, None] * vh[:rank]).to(weight.dtype)
            result[prefix + ".1.weight"] = (u[:, :rank] * scale).to(weight.dtype)
            result[prefix + ".1.bias"] = result.pop(prefix + ".bias")
    return result


def candidate_ranking_loss(logits, targets, statistics, unknown_weight=.3):
    """Rank each recorded positive above the highest-scored unrecorded candidates."""
    positive = targets > 0
    negatives = logits.masked_fill(positive, -torch.inf).topk(min(3, logits.size(1)), dim=1)
    available = negatives.values.isfinite()
    negative_scores = negatives.values.masked_fill(~available, 0)
    weights = unlabeled_weights(targets, statistics, unknown_weight).gather(1, negatives.indices)
    pairs = positive[:, :, None] & available[:, None, :]
    violations = torch.nn.functional.softplus(negative_scores[:, None, :] - logits[:, :, None])
    per_row = (violations * pairs * weights[:, None, :]).sum((1, 2)) / pairs.sum((1, 2)).clamp_min(1)
    return per_row.mean()


def refine(args, output):
    seed_everything(42)
    packed = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if packed["model_name"] != "trievidence":
        raise ValueError("Use the packed TriEvidence best.pt")
    source, val = open_data(args, "val", packed)
    idx = val.indices
    truth = torch.from_numpy(source.data.y[idx].astype(bool))
    with np.load(args.checkpoint.with_name("val_predictions.npz")) as baseline:
        if str(baseline["split"]) != "val":
            raise ValueError("The packed checkpoint needs its Validation export")
        for key, expected in (("row_id", source.data.rows[idx]), ("targets", truth.numpy()),
                              ("label_names", source.data.labels), ("cluster_id", source.data.clusters[idx])):
            np.testing.assert_array_equal(baseline[key], expected)
        a, c = [torch.from_numpy(baseline[key]) for key in ("theraset_logits", "aa34_logits")]
        reference = tuple(torch.from_numpy(baseline[key]) for key in ("predicted_raw", "predicted"))
    offsets = packed["model_state"]["offsets"]
    reference_acc = tuple(float(row_jaccard(truth, p).mean()) for p in reference)
    model_args = packed["model_args"]
    original_depth = model_args["helixset_args"].get("latent_depth", 6)
    if not 1 <= args.depth <= original_depth or args.epochs < 0:
        raise ValueError("Use a positive depth no larger than the checkpoint depth and nonnegative epochs")
    if args.representation and (args.epochs < 1 or args.depth != original_depth or args.ablate or args.ffn_rank):
        raise ValueError("Representation training requires the original depth and at least one training epoch")
    if args.ablate:
        if args.epochs or args.depth != original_depth:
            raise ValueError("Ablation compares original-depth weights without recovery training")
        flag = {"geometry": "structure_evidence", "triangles": "triangles"}[args.ablate]
        if model_args["helixset_args"].get(flag, True) is False:
            raise ValueError(f"The reference already disables {flag}")
        model_args["helixset_args"][flag] = False
    model_args["helixset_args"]["latent_depth"] = args.depth
    state = packed["model_state"]
    if args.ffn_rank is not None:
        if args.ablate or args.depth != original_depth or model_args["helixset_args"].get("latent_ffn_rank") is not None:
            raise ValueError("Start from the original full-rank checkpoint at the same depth")
        model_args["helixset_args"]["latent_ffn_rank"] = args.ffn_rank
    model = TriEvidence(**model_args)
    if args.ffn_rank is not None:
        state = svd_latent_ffn(state, args.depth, args.ffn_rank)
    model.load_state_dict({k: state[k] for k in model.state_dict()})
    # Evaluation needs only the changing branch. The other two were exported
    # from this packed model on the same Validation rows.
    helix = model.helixset.to(args.device).eval()
    loader = batches(val, args.batch_size)
    train = None
    if args.epochs:
        train = MultimodalPeptides(source, args.features, "train", peptide_feature_root=args.peptide_features)
        validate_cache(train, packed["config"], packed["encoder"])
        train_loader = batches(train, args.batch_size, shuffle=True)
        if args.representation:
            train_blocks = list(helix.pair_blocks[-2:]) + list(helix.latent_blocks[-2:])
            for block in train_blocks:
                block.requires_grad_(True)
        elif args.ffn_rank is not None:
            for block in helix.latent_blocks:
                for index in (1, 4):
                    block.process.ff.net[index].requires_grad_(True)
        else:
            helix.latent_blocks[-1].requires_grad_(True)
        helix.latent_classifier.requires_grad_(True)
        optimizer = torch.optim.AdamW((p for p in helix.parameters() if p.requires_grad),
                                      lr=args.learning_rate, weight_decay=.01)

    best, history = None, []
    print(f"depth={args.depth} ffn_rank={args.ffn_rank} | Validation={len(val)} | output={output}", flush=True)
    for epoch in range(args.epochs + 1):
        loss_sum = 0.
        if epoch:
            if args.representation:
                for block in train_blocks:
                    block.train()
            elif args.ffn_rank is None:
                helix.latent_blocks[-1].train()
            helix.latent_classifier.train()
            for step, batch in enumerate(train_loader, 1):
                batch = to_device(batch, args.device)
                optimizer.zero_grad(set_to_none=True)
                logits = helix(batch)["logits"]
                labels = batch["y"]
                statistics = source.statistics["labels"]
                loss = observed_label_loss(logits, labels, statistics)
                if args.ranking_weight:
                    loss = loss + args.ranking_weight * candidate_ranking_loss(logits, labels, statistics)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch["y"])
                if step % 256 == 0:
                    print(f"Epoch {epoch} train {step}/{len(train_loader)}", flush=True)
            helix.eval()
        h, = collect(lambda batch: (helix(batch)["logits"],), loader, args.device, progress=True)
        scores = fuse_evidence(a, c, h, offsets)
        pred = scores >= 0, empty_top10(scores, a)
        acc = tuple(float(row_jaccard(truth, p).mean()) for p in pred)
        delta = tuple(new - old for new, old in zip(acc, reference_acc))
        key = (min(delta), acc[0], acc[1])
        changes = [int((p != old).any(1).sum()) for p, old in zip(pred, reference)]
        record = {"epoch": epoch, "raw_acc": acc[0], "fallback_acc": acc[1],
                  "delta_pp": [100 * d for d in delta], "changed_rows": changes,
                  "train_loss": loss_sum / len(train) if epoch else None}
        history.append(record)
        print(json.dumps(record), flush=True)
        if (args.representation and epoch == args.epochs) or (
                not args.representation and (best is None or key > best[0])):
            best = key, epoch, h
            torch.save(dict(packed, model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                            latent_recovery_epoch=epoch), output / "best.pt")

    _, epoch, h = best
    report = diagnostics(source, val, a, c, h, offsets, "val", output)
    report.update(reference_checkpoint=str(args.checkpoint), depth=args.depth, ablation=args.ablate,
                  ffn_rank=args.ffn_rank, best_epoch=epoch,
                  representation=args.representation, ranking_weight=args.ranking_weight,
                  selection="fixed_final_epoch" if args.representation else "validation",
                  train_rows=len(train) if train is not None else 0,
                  parameters=sum(p.numel() for p in model.parameters()), history=history)
    write_json(output / "results.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--ffn-rank", type=int, help="SVD rank for the two FFN linear layers in each latent block")
    parser.add_argument("--ablate", choices=("geometry", "triangles"),
                        help="Disable only HelixSet explicit geometry or only its triangle updates")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--representation", action="store_true",
                        help="Tune the last two pair and latent blocks and use the fixed final epoch")
    parser.add_argument("--ranking-weight", type=float, default=0.)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ranking_weight < 0 or (args.ranking_weight != 0 and not args.representation):
        parser.error("--ranking-weight requires --representation and must be nonnegative")
    output = args.output or Path("runs") / datetime.now().strftime("trievidence_depth_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    refine(args, output)
