"""Replace only the TheraSet sequence context with projected HelixSet states."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mfp_nova.models.trievidence import TriEvidence, center_evidence, empty_top10, fuse_evidence
from mfp_nova.multimodal import MultimodalPeptides
from mfp_nova.research import to_device, validate_cache
from mfp_nova.research_layers import observed_label_loss
from mfp_nova.trainer import seed_everything, write_json
from run_trievidence import batches, collect, diagnostics, open_data, row_jaccard


def project(adapter, state, mask):
    return adapter(state) * mask[..., None], mask


def shared_context(encoder, batch, layer):
    if layer is None:
        final, mask = encoder(batch)
        return final, mask, final
    final, mask, residues = encoder(batch, return_residue_layer=layer)
    # Keep the final chemistry token identical in every arm.
    return final, mask, torch.cat([residues, final[:, -1:]], dim=1)


def fit_epoch(model, adapter, loader, optimizer, device, phase, sequence_layer):
    adapter.train()
    total, count = 0., 0
    for step, batch in enumerate(loader, 1):
        batch = to_device(batch, device)
        with torch.no_grad():
            _, mask, state = shared_context(model.helixset.sequence, batch, sequence_layer)
            target_context = model.theraset.encode_context(batch)
            if phase == "scores":
                target_logits = model.theraset(batch, sequence_context=target_context)["logits"]
        context = project(adapter, state, mask)
        if phase == "context":
            errors = (context[0] - target_context[0]).square().mean(-1)
            residues = (errors[:, :-1] * mask[:, :-1]).sum() / mask[:, :-1].sum().clamp_min(1)
            loss = .5 * (residues + errors[:, -1].mean())
        else:
            scores = model.theraset(batch, sequence_context=context)["logits"]
            matching = F.smooth_l1_loss(scores, target_logits)
            supervised = observed_label_loss(scores, batch["y"], model.theraset.statistics)
            loss = matching + supervised
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(batch["y"])
        count += len(batch["y"])
        if step == 1 or step % 256 == 0 or step == len(loader):
            print(f"{phase} train {step}/{len(loader)} loss={total/count:.6f}", flush=True)
    return total / count


def run(args):
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TriEvidence(**checkpoint["model_args"])
    model.load_state_dict(checkpoint.pop("model_state"))
    if model.theraset.v5_readout is not None or not model.theraset.contextual_evidence:
        raise ValueError("Use the original packed TheraSet and HelixSet baseline")
    depth = len(model.helixset.sequence.layers)
    if args.sequence_layer is not None and not 0 <= args.sequence_layer <= depth:
        raise ValueError(f"--sequence-layer must be between 0 and {depth}")
    model.to(args.device).eval().requires_grad_(False)
    source, val = open_data(args, "val", checkpoint)
    train = MultimodalPeptides(source, args.features, "train", peptide_feature_root=args.peptide_features)
    validate_cache(train, checkpoint["config"], checkpoint["encoder"])
    y = torch.from_numpy(source.data.y[val.indices]).bool()
    offsets = model.offsets.cpu()
    with np.load(args.checkpoint.with_name("val_predictions.npz")) as original:
        for key, expected in (("row_id", source.data.rows[val.indices]), ("targets", y.numpy()),
                              ("cluster_id", source.data.clusters[val.indices]), ("label_names", source.data.labels)):
            np.testing.assert_array_equal(original[key], expected)
        t, a, h = (torch.from_numpy(original[key]) for key in
                   ("theraset_logits", "aa34_logits", "latent_logits"))
        baseline = fuse_evidence(t, a, h, offsets)
        np.testing.assert_array_equal(baseline >= 0, original["predicted_raw"])
        np.testing.assert_array_equal(empty_top10(baseline, t), original["predicted"])
    helix_dim = model.helixset.sequence.plm.out_features
    thera_dim = model.theraset.sequence.plm.out_features
    adapter = nn.Linear(helix_dim, thera_dim).to(args.device)
    train_loader = batches(train, args.batch_size, shuffle=True)
    val_loader = batches(val, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=False)
    def two_way(first):
        return center_evidence(torch.stack([first, h]), offsets[[0, 2]]).sum(0) >= 0

    baseline_two = two_way(t)
    existing = y & baseline_two
    unique_t = existing & (t >= offsets[0]) & (h < offsets[2])
    unique_h = existing & (h >= offsets[2]) & (t < offsets[0])
    history = []

    def evaluate(name, train_loss):
        adapter.eval()

        def read(batch):
            final, mask, state = shared_context(model.helixset.sequence, batch, args.sequence_layer)
            result = model.theraset(batch, sequence_context=project(adapter, state, mask))
            latent = model.helixset(batch, sequence_context=(final, mask))["logits"]
            return result["logits"], result["direct"], latent

        candidate, direct, live_helix = collect(read, val_loader, args.device, progress=True)
        out = args.output / name
        out.mkdir()
        report = diagnostics(source, val, candidate, a, h, offsets, "val", out)
        two = two_way(candidate)
        report.update(stage=name, training_loss=train_loss, anchor_checkpoint=str(args.checkpoint),
                      sequence_layer=depth if args.sequence_layer is None else args.sequence_layer,
                      adapter_parameters=sum(p.numel() for p in adapter.parameters()),
                      diagnostic=dict(baseline_two_acc=float(row_jaccard(y, baseline_two).mean()),
                                      helixset_max_abs_delta=float((live_helix - h).abs().max()),
                                      retained_theraset_only=int((two & unique_t).sum()),
                                      retained_helixset_only=int((two & unique_h).sum()),
                                      total_theraset_only=int(unique_t.sum()),
                                      total_helixset_only=int(unique_h.sum())))
        write_json(out / "results.json", report)
        torch.save({"adapter_state": adapter.state_dict(), "anchor_checkpoint": str(args.checkpoint),
                    "sequence_layer": args.sequence_layer},
                   out / "adapter.pt")
        np.save(out / "direct_logits.npy", direct.numpy())
        history.append({"stage": name, "training_loss": train_loss,
                        "two_acc": report["arms"]["theraset_helixset"]["sample_jaccard"],
                        "fusion_acc": report["arms"]["fusion"]["sample_jaccard"]})
        write_json(args.output / "history.json", history)

    for phase, epochs in (("context", args.context_epochs), ("scores", args.score_epochs)):
        if not epochs:
            continue
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr if phase == "context" else args.lr / 5)
        for epoch in range(1, epochs + 1):
            loss = fit_epoch(model, adapter, train_loader, optimizer, args.device, phase, args.sequence_layer)
            evaluate(f"{phase}_{epoch}", loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--context-epochs", type=int, default=1)
    parser.add_argument("--score-epochs", type=int, default=1)
    parser.add_argument("--sequence-layer", type=int,
                        help="Residue state after this many HelixSet sequence blocks; default: final")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs") / datetime.now().strftime("trievidence_context_%Y%m%d_%H%M%S"))
    run(parser.parse_args())
