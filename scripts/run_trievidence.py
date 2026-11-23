"""Train the AA34 reader and evaluate a packed TriEvidence-MFP checkpoint."""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mfp_nova.metrics import evaluate_metrics
from evaluate_fine39_fixed_fusion import logit, read_theraset
from mfp_nova.models.trievidence import (AA34Reader, CandidateLabelReadout, READER_HEADS, TriEvidence,
                                        center_evidence, empty_top10, fuse_evidence)
from mfp_nova.multimodal import (BackbonePeptides, MultimodalData, MultimodalPeptides,
                                 collate_multimodal, independent_functions)
from mfp_nova.research import to_device, validate_cache
from mfp_nova.research_layers import observed_label_loss
from mfp_nova.trainer import seed_everything, write_json


MODEL_NAME = "trievidence"


def read_anchor(path, expected):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint["model_name"] != expected or checkpoint["track"] != "fine39"
            or "peptide_encoder" not in checkpoint["config"]
            or checkpoint["model_args"].get("esm_tail") is not None
            or checkpoint["config"]["data"].get("augment")):
        raise ValueError(f"Use the original Fine39 dual-cache {expected} checkpoint: {path}")
    return checkpoint


def initialize(theraset_path, helixset_path, channels, dropout, reader_head):
    anchors = [read_anchor(path, name) for path, name in
               ((theraset_path, "theraset"), (helixset_path, "helixset"))]
    theraset, helixset = anchors
    stats, helix_stats = (anchor["statistics"] for anchor in anchors)
    shared_inputs = ("dataset", "phys", "nodes", "structure_global", "chemical_vocab",
                     "localized_terminal_chemistry")
    if theraset["labels"] != helixset["labels"] or any(
            stats.get(key) != helix_stats.get(key) for key in shared_inputs):
        raise ValueError("The anchors require different label orders or input preprocessing")
    if helixset["model_args"].get("evidence_readout"):
        raise ValueError("Use standard HelixSet with its supervised latent-direct head")
    arguments = {"theraset_args": theraset["model_args"], "helixset_args": helixset["model_args"],
                 "cnn_channels": channels, "cnn_dropout": dropout, "reader_head": reader_head}
    model = TriEvidence(**arguments)
    model.theraset.load_state_dict(theraset["model_state"])
    # Only keys belonging to the latent-only architecture go into the packed model.
    model.helixset.load_state_dict({key: helixset["model_state"][key] for key in model.helixset.state_dict()})
    model.offsets[[0, 2]] = torch.logit(torch.tensor([float(theraset["threshold"]), float(helixset["threshold"])]))
    return model, arguments, anchors


def open_data(args, split, checkpoint):
    stats = checkpoint["statistics"]
    source = MultimodalData(args.data_root, "fine39", stats,
                            localize_terminals=stats.get("localized_terminal_chemistry", False))
    if len(source.data.indices("train")) != stats["train_rows"]:
        raise ValueError("Checkpoint must have been fitted on the full original Fine39 Train")
    dataset = MultimodalPeptides(source, args.features, split, peptide_feature_root=args.peptide_features)
    validate_cache(dataset, checkpoint["config"], checkpoint["encoder"])
    return source, dataset


def batches(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_multimodal)


def align_saved(export, source, dataset, score):
    rows = dataset.indices
    order = np.argsort(export["row_id"])
    np.testing.assert_array_equal(export["row_id"][order], source.data.rows[rows])
    np.testing.assert_array_equal(export["targets"][order], source.data.y[rows])
    np.testing.assert_array_equal(export["cluster_id"][order], source.data.clusters[rows])
    np.testing.assert_array_equal(export["label_names"], source.data.labels)
    if "ID" in export:
        np.testing.assert_array_equal(export["ID"][order].astype(str), source.data.ids[rows].astype(str))
    if "split" in export and str(export["split"]) != "val":
        raise ValueError("The reader may only use original Validation anchor predictions")
    return torch.from_numpy(logit(export[score][order]).astype(np.float32))


@torch.inference_mode()
def collect(read, loader, device, progress=False, timings=None):
    parts = []
    tick = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        parts.append(tuple(t.float().cpu() for t in read(to_device(batch, device))))
        if timings is not None:
            timings.append(time.perf_counter() - tick)
        if progress and (step == 1 or step % 64 == 0 or step == len(loader)):
            print(f"Evidence batch {step}/{len(loader)}", flush=True)
        tick = time.perf_counter()
    return tuple(torch.cat(tensors) for tensors in zip(*parts))


def row_jaccard(y, keep):
    return (y & keep).sum(1) / (y | keep).sum(1).clamp_min(1)


def tune_cnn_offset(y, theraset, aa34, latent, offsets):
    # One global AA34 cutoff, with both pretrained offsets and the equal weights fixed.
    candidates = torch.logit(torch.arange(4, 37, dtype=torch.float32) / 40)
    best = None
    trial = offsets.clone()
    for offset in candidates:
        trial[1] = offset
        logits = fuse_evidence(theraset, aa34, latent, trial)
        score = float(row_jaccard(y, logits >= 0).mean())
        key = (score, -abs(float(offset)))
        if best is None or key > best[0]:
            best = key, offset
    key, offset = best
    trial[1] = offset
    return key, trial, fuse_evidence(theraset, aa34, latent, trial)


def pass_train(reader, loader, device, optimizer, statistics, unlabeled_weight,
               teacher=None, teacher_offset=None, distill_weight=.5, review_weight=.25,
               auxiliary_loss=None):
    reader.train()
    loss_sum = 0.
    for batch in loader:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        targets = batch["y"]
        if hasattr(reader, "review_classifier"):
            primary, review = reader(batch, return_heads=True)
            logits = (primary + review) * .5
            primary_loss = observed_label_loss(primary, targets, statistics, unlabeled_weight)
            review_loss = observed_label_loss(review, targets, statistics, unlabeled_weight)
            if review_weight:
                # Revisit the primary reader's strongest unrecorded candidates.
                # The shared loss retains the original weak annotation weights.
                candidates = primary.detach().masked_fill(targets > 0, -torch.inf)
                values, indices = candidates.topk(min(3, targets.shape[1]), dim=1)
                mask = torch.zeros_like(targets).scatter_(1, indices, values.isfinite().to(targets.dtype))
                review_loss = review_loss + review_weight * observed_label_loss(
                    review, targets, statistics, unlabeled_weight, mask=mask)
            loss = (primary_loss + review_loss) * .5
        else:
            logits = reader(batch)
            loss = observed_label_loss(logits, targets, statistics, unlabeled_weight)
        if teacher is not None and distill_weight > 0:
            with torch.no_grad():
                target = teacher(batch)
                weight = 1 + 2 * torch.exp(-(target - teacher_offset).abs())
            residual = torch.nn.functional.smooth_l1_loss(logits, target, reduction="none")
            loss = loss + distill_weight * (weight * residual).mean()
        if auxiliary_loss is not None:
            loss = loss + auxiliary_loss(logits, batch)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * len(batch["y"])
    return loss_sum / len(loader.dataset)


def diagnostics(source, dataset, theraset, aa34, latent, offsets, split, output, shared=False, fused_logits=None):
    y = source.data.y[dataset.indices].astype(bool)
    truth = torch.from_numpy(y)
    mixed = fuse_evidence(theraset, aa34, latent, offsets) if fused_logits is None else fused_logits
    two = center_evidence(torch.stack((theraset, latent)), offsets[[0, 2]]).mean(0)
    sequence_two = center_evidence(torch.stack((theraset, aa34)), offsets[:2]).mean(0)
    third = "shared_readout" if shared else "helixset_latent"
    two_name = "theraset_shared" if shared else "theraset_helixset"
    raw, rescued = mixed >= 0, empty_top10(mixed, theraset)
    order = theraset.argsort(dim=1, descending=True, stable=True)
    missed_top5 = ~truth.gather(1, order[:, :5]).any(1)
    low_ranks = torch.zeros_like(truth).scatter_(1, order[:, 5:10], True)
    reference = theraset >= offsets[0]
    reference_hit = truth & reference
    support = source.statistics["labels"]["support"]
    projection = source.statistics["function_projection"]
    multi = torch.from_numpy(independent_functions(y, projection).sum(1) >= 2)

    def summary(logits, predicted):
        result = evaluate_metrics(y, logits.sigmoid().numpy(), train_support=support,
                                  function_projection=projection, predicted=predicted.numpy())
        hit = truth & predicted
        result.update(zero_hit_rows=int((~hit.any(1)).sum()),
                      empty_rows=int((~predicted.any(1)).sum()),
                      multifunction_sample_jaccard=float(row_jaccard(truth, predicted)[multi].mean()) if multi.any() else None,
                      top5_miss_rescues=int((hit.any(1) & missed_top5).sum()),
                      new_labels_at_theraset_6_10=int((hit & low_ranks & ~reference_hit).sum()),
                      lost_theraset_true_labels=int((reference_hit & ~hit).sum()))
        return result

    results = {
        "theraset": summary(theraset, reference),
        "aa34": summary(aa34, aa34 >= offsets[1]),
        third: summary(latent, latent >= offsets[2]),
        two_name: summary(two, two >= 0),
        f"{two_name}_empty_top10": summary(two, empty_top10(two, theraset)),
        "theraset_aa34": summary(sequence_two, sequence_two >= 0),
        "theraset_aa34_empty_top10": summary(sequence_two, empty_top10(sequence_two, theraset)),
        "fusion": summary(mixed, raw),
        "fusion_empty_top10": summary(mixed, rescued),
    }
    indices = dataset.indices
    np.savez_compressed(output / f"{split}_predictions.npz",
                        row_id=source.data.rows[indices], ID=source.data.ids[indices],
                        cluster_id=source.data.clusters[indices], label_names=source.data.labels,
                        targets=y, split=split, probabilities=mixed.sigmoid().numpy(), threshold=.5,
                        fusion_logits=mixed.numpy(),
                        predicted=rescued.numpy(), predicted_raw=raw.numpy(),
                        theraset_logits=theraset.numpy(), aa34_logits=aa34.numpy(), latent_logits=latent.numpy())
    report = {"model": MODEL_NAME, "split": split, "rows": len(y), "multifunction_rows": int(multi.sum()),
              "theraset_top5_misses": int(missed_top5.sum()), "third_branch": third,
              "branch_thresholds": offsets.sigmoid().tolist(), "arms": results}
    if fused_logits is not None:
        original = fuse_evidence(theraset, aa34, latent, offsets)
        before = empty_top10(original, theraset)
        before_hit, after_hit = before & truth, rescued & truth
        unknown = torch.tensor([label.endswith("__unknown") for label in source.data.labels])
        report["candidate_effect"] = {
            "baseline_raw_acc": float(row_jaccard(truth, original >= 0).mean()),
            "baseline_final_acc": float(row_jaccard(truth, before).mean()),
            "changed_final_rows": int((before != rescued).any(1).sum()),
            "rescued_zero_hits": int((~before_hit.any(1) & after_hit.any(1)).sum()),
            "new_zero_hits": int((before_hit.any(1) & ~after_hit.any(1)).sum()),
            "multifunction_gained_labels": int((after_hit & ~before_hit)[multi].sum()),
            "multifunction_lost_labels": int((before_hit & ~after_hit)[multi].sum()),
            "unknown_gained_labels": int((after_hit & ~before_hit)[:, unknown].sum()),
            "unknown_lost_labels": int((before_hit & ~after_hit)[:, unknown].sum()),
        }
    for name in ("theraset", "theraset_aa34" if shared else two_name, "fusion", "fusion_empty_top10"):
        print(f"{name} ACC={results[name]['sample_jaccard']:.4%} "
              f"ZeroHit={results[name]['zero_hit_rows']}", flush=True)
    return report


def fit_aa34(args, reader, source, backbone, val_backbone, a, h, offsets, teacher=None):
    # DataLoader initialization for the live anchors consumes RNG state.
    seed_everything(args.seed)
    stats = source.statistics["labels"]
    y = torch.from_numpy(source.data.y[val_backbone.indices].astype(bool))
    train_loader = batches(backbone, args.cnn_batch_size, shuffle=True)
    val_loader = batches(val_backbone, args.cnn_batch_size)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=args.learning_rate, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best, history = None, []
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        loss = pass_train(reader, train_loader, args.device, optimizer, stats, args.unlabeled_weight,
                          teacher, offsets[1].to(args.device), getattr(args, "distill_weight", .5),
                          args.review_weight)
        schedule.step()
        reader.eval()
        b, = collect(lambda batch: (reader(batch),), val_loader, args.device)
        if teacher is None:
            key, chosen, fused = tune_cnn_offset(y, a, b, h, offsets)
        else:
            chosen = offsets
            fused = fuse_evidence(a, b, h, chosen)
            key = (float(row_jaccard(y, fused >= 0).mean()),)
        record = {"epoch": epoch, "train_loss": loss, "raw_val_acc": key[0], "cnn_threshold": float(chosen[1].sigmoid()),
                  "fallback_val_acc": float(row_jaccard(y, empty_top10(fused, a)).mean()),
                  "seconds": time.perf_counter() - tick}
        history.append(record)
        print(json.dumps(record), flush=True)
        if best is None or teacher is not None or key > best[0]:
            weights = {k: v.detach().cpu().clone() for k, v in reader.state_dict().items()}
            best = key, chosen, b, epoch, weights

    _, chosen, b, epoch, weights = best
    reader.load_state_dict(weights)
    return chosen, b, epoch, history


def train(args, output):
    seed_everything(args.seed)
    started = time.perf_counter()
    model, model_args, anchors = initialize(args.theraset, args.helixset, args.channels, args.dropout, args.reader_head)
    theraset, helixset = anchors
    source, validation = open_data(args, "val", theraset)
    validate_cache(validation, helixset["config"], helixset["encoder"])
    if helixset["statistics"]["train_rows"] != theraset["statistics"]["train_rows"]:
        raise ValueError("The anchors must use the same Fine39 Train")
    backbone = BackbonePeptides(source, "train", validation.meta["max_length"])
    val_backbone = BackbonePeptides(source, "val", validation.meta["max_length"])
    np.testing.assert_array_equal(val_backbone.indices, validation.indices)
    anchor_paths = (args.theraset, args.helixset)
    origin = [{"checkpoint": str(path), "epoch": anchor.get("epoch"), "threshold": anchor["threshold"],
               "weight_source": anchor.get("weight_source")} for path, anchor in zip(anchor_paths, anchors)]
    static = {"model_name": MODEL_NAME, "model_args": model_args, "track": "fine39",
              "config": theraset["config"], "encoder": theraset["encoder"], "statistics": source.statistics,
              "labels": source.data.labels, "anchors": origin}
    del theraset, helixset, anchors

    model.to(args.device)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Validation | rows={len(validation)} | collecting the two frozen branches once", flush=True)
    a, h = collect(model.anchor_logits, batches(validation, args.batch_size), args.device, progress=True)
    offsets = model.offsets.cpu()
    if args.reader is not None:
        cached = torch.load(args.reader, map_location="cpu", weights_only=True)
        if (cached["labels"] != source.data.labels or cached["dataset"] != source.statistics["dataset"]
                or cached["channels"] != args.channels or cached.get("reader_head", "global") != args.reader_head):
            raise ValueError("Reader settings differ from the current Fine39 labels, dataset or model configuration")
        torch.testing.assert_close(cached["offsets"][[0, 2]], offsets[[0, 2]])
        model.aa34.load_state_dict(cached["model_state"])
        model.eval()
        b, = collect(lambda batch: (model.aa34(batch),), batches(val_backbone, args.cnn_batch_size), args.device)
        chosen, epoch, history = cached["offsets"], cached["epoch"], []
    else:
        chosen, b, epoch, history = fit_aa34(args, model.aa34, source, backbone, val_backbone, a, h, offsets)
    model.offsets.copy_(chosen)
    torch.save(dict(static, epoch=epoch, model_state=model.state_dict()), output / "best.pt")
    report = diagnostics(source, validation, a, b, h, chosen, "val", output)
    report.update(best_epoch=epoch, anchors=origin, history=history, train_rows=len(backbone),
                  source_reader=str(args.reader) if args.reader else None, reader_head=args.reader_head,
                  review_weight=args.review_weight if args.reader_head == "dual" and not args.reader else None,
                  seed=args.seed, elapsed_seconds=time.perf_counter() - started,
                  cnn_parameters=sum(p.numel() for p in model.aa34.parameters()),
                  total_parameters=sum(p.numel() for p in model.parameters()),
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  torch=str(torch.__version__), device=args.device, cnn_batch_size=args.cnn_batch_size,
                  inference_batch_size=args.batch_size, learning_rate=args.learning_rate,
                  unlabeled_weight=args.unlabeled_weight, dataset=source.statistics["dataset"])
    write_json(output / "results.json", report)


def train_cached(args, output):
    seed_everything(args.seed)
    started = time.perf_counter()
    source = MultimodalData(args.data_root, "fine39", sequence_only=True)
    backbone = BackbonePeptides(source, "train", 80)
    validation = BackbonePeptides(source, "val", 80)
    thera = read_theraset(args.theraset_val, "val")
    with np.load(args.helix_exits) as exits:
        a = align_saved(thera, source, validation, "probabilities")
        h = align_saved(exits, source, validation, "direct")
    offsets = torch.tensor([float(thera["threshold"]), .5, args.helix_threshold]).logit()
    reader = AA34Reader(source.statistics["labels"], args.channels, args.dropout, args.reader_head).to(args.device)
    teacher = None
    if args.distill_reader:
        checkpoint = torch.load(args.distill_reader, map_location="cpu", weights_only=True)
        if checkpoint["labels"] != source.data.labels or checkpoint["dataset"] != source.statistics["dataset"]:
            raise ValueError("Teacher must use the same Fine39 dataset and label order")
        torch.testing.assert_close(checkpoint["offsets"][[0, 2]], offsets[[0, 2]])
        offsets = checkpoint["offsets"]
        teacher = AA34Reader(source.statistics["labels"], checkpoint["channels"], args.dropout,
                             checkpoint.get("reader_head", "global"))
        teacher.load_state_dict(checkpoint["model_state"])
        teacher.to(args.device).requires_grad_(False).eval()
    print(f"Cached evidence | Train={len(backbone)} Validation={len(validation)}", flush=True)
    chosen, b, epoch, history = fit_aa34(args, reader, source, backbone, validation, a, h, offsets, teacher)
    torch.save({"model_state": reader.state_dict(), "offsets": chosen, "epoch": epoch, "channels": args.channels,
                "reader_head": args.reader_head,
                "review_weight": args.review_weight if args.reader_head == "dual" else None,
                "labels": source.data.labels, "dataset": source.statistics["dataset"]}, output / "reader.pt")
    report = diagnostics(source, validation, a, b, h, chosen, "val", output)
    report.update(best_epoch=epoch, history=history, train_rows=len(backbone), seed=args.seed,
                  theraset_export=str(args.theraset_val), helixset_export=str(args.helix_exits),
                  elapsed_seconds=time.perf_counter() - started, device=args.device, torch=str(torch.__version__),
                  cnn_channels=args.channels, cnn_batch_size=args.cnn_batch_size, dropout=args.dropout,
                  cnn_parameters=sum(p.numel() for p in reader.parameters()),
                  distill_reader=str(args.distill_reader) if teacher is not None else None,
                  distill_weight=args.distill_weight if teacher is not None else None,
                  selection="fixed_final_epoch_and_teacher_offset" if teacher is not None else "validation_raw",
                  reader_head=args.reader_head,
                  review_weight=args.review_weight if args.reader_head == "dual" else None,
                  learning_rate=args.learning_rate, unlabeled_weight=args.unlabeled_weight,
                  dataset=source.statistics["dataset"],
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    write_json(output / "results.json", report)


def fit_candidates(args, output):
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_args = checkpoint["model_args"]
    if (checkpoint["model_name"] != MODEL_NAME or checkpoint["track"] != "fine39"
            or "helixset_args" not in model_args or model_args.get("candidate_mode")):
        raise ValueError("Use an original packed Fine39 TriEvidence checkpoint")
    model = TriEvidence(**model_args)
    model.load_state_dict(checkpoint.pop("model_state"))
    model.to(device).eval().requires_grad_(False)
    model.reuse_input_compute = True

    if args.probe_memory:
        source, dataset = open_data(args, "train", checkpoint)
        backbone = BackbonePeptides(source, "train", checkpoint["encoder"]["max_length"])
        np.testing.assert_array_equal(backbone.indices, dataset.indices)
        longest = np.argsort(-np.asarray(backbone.lengths), kind="stable")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batch = next(iter(batches(Subset(dataset, longest[:args.batch_size].tolist()), args.batch_size)))
        batch = to_device(batch, device)
        with torch.no_grad():
            thera, helix = model.anchor_logits(batch)
            aa34 = model.aa34(batch)
        anchor_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        del batch
        head = CandidateLabelReadout(model.aa34.classifier[0].in_features, len(source.data.labels),
                                     args.candidate_mode, args.max_delta).to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batch = next(iter(batches(Subset(backbone, longest[:args.candidate_batch_size].tolist()),
                                  args.candidate_batch_size)))
        batch = to_device(batch, device)
        count = len(batch["y"])
        centered = center_evidence(torch.stack((thera, aa34, helix)), model.offsets)
        centered = centered[:, torch.arange(count, device=device) % len(thera)]
        with torch.no_grad():
            maps = model.aa34.motifs(batch) if head.mode == "local" else None
        logits, candidates = head(centered, maps, batch["mask"])
        mask = torch.zeros_like(batch["y"]).scatter_(1, candidates, 1.)
        loss = observed_label_loss(logits, batch["y"], source.statistics["labels"], args.unlabeled_weight, mask=mask)
        loss.backward()
        torch.optim.AdamW(head.parameters(), lr=args.learning_rate).step()
        report = dict(mode=args.candidate_mode, batch_size=len(thera), candidate_batch_size=count,
                      max_residues=batch["mask"].shape[1], loss=float(loss.detach()),
                      anchor_peak_bytes=anchor_peak,
                      candidate_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
        write_json(output / "results.json", report)
        print(json.dumps(report), flush=True)
        return

    def prepare(split):
        print(f"Collect fixed evidence | split={split}", flush=True)
        source, dataset = open_data(args, split, checkpoint)
        backbone = BackbonePeptides(source, split, checkpoint["encoder"]["max_length"])
        np.testing.assert_array_equal(backbone.indices, dataset.indices)

        def read(batch):
            thera, helix = model.anchor_logits(batch)
            return thera, model.aa34(batch), helix

        columns = collect(read, batches(dataset, args.batch_size), device, progress=True)
        centered = center_evidence(torch.stack(columns), model.offsets.cpu())
        rows = torch.as_tensor(source.data.rows[dataset.indices], device=device)
        sorted_rows, order = rows.sort()
        return source, backbone, columns, (sorted_rows, centered[:, order.cpu()].to(device))

    source, train, _, train_scores = prepare("train")
    _, validation, val_columns, val_scores = prepare("val")
    head = CandidateLabelReadout(model.aa34.classifier[0].in_features, len(source.data.labels),
                                 args.candidate_mode, args.max_delta).to(device)
    model.candidate_readout = head
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate)
    train_loader = batches(train, args.candidate_batch_size, shuffle=True)
    val_loader = batches(validation, args.candidate_batch_size)
    stats = source.statistics["labels"]
    labels = torch.from_numpy(source.data.y[validation.indices].astype(bool))
    baseline = fuse_evidence(*val_columns, model.offsets.cpu())
    baseline_raw = baseline >= 0
    baseline_final = empty_top10(baseline, val_columns[0])

    def run_batch(batch, cache):
        row, scores = cache
        centered = scores[:, torch.searchsorted(row, batch["row_id"])]
        with torch.no_grad():
            maps = model.aa34.motifs(batch) if head.mode == "local" else None
        return head(centered, maps, batch["mask"])

    history = []
    best = None
    for epoch in range(args.epochs + 1):
        loss_sum = 0.
        if epoch:
            head.train()
            for batch in train_loader:
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                logits, candidates = run_batch(batch, train_scores)
                mask = torch.zeros_like(batch["y"]).scatter_(1, candidates, 1.)
                loss = observed_label_loss(logits, batch["y"], stats, args.unlabeled_weight, mask=mask)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch["y"])
        head.eval()
        fused, = collect(lambda batch: (run_batch(batch, val_scores)[0],), val_loader, device)
        raw = fused >= 0
        final = empty_top10(fused, val_columns[0])
        if epoch == 0:
            torch.testing.assert_close(fused, baseline, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(raw, baseline_raw)
            torch.testing.assert_close(final, baseline_final)
        raw_acc = float(row_jaccard(labels, raw).mean())
        final_acc = float(row_jaccard(labels, final).mean())
        zero_hits = int((~(labels & final).any(1)).sum())
        record = {"epoch": epoch, "train_loss": loss_sum / len(train) if epoch else None,
                  "raw_val_acc": raw_acc, "fallback_val_acc": final_acc, "zero_hits": zero_hits,
                  "changed_raw_rows": int((raw != baseline_raw).any(1).sum()),
                  "changed_final_rows": int((final != baseline_final).any(1).sum())}
        history.append(record)
        print(json.dumps(record), flush=True)
        key = (final_acc, -zero_hits, raw_acc)
        if args.export_first_epoch and epoch in (0, 1):
            snapshot = output / f"epoch_{epoch}"
            snapshot.mkdir()
            report = diagnostics(source, validation, *val_columns, model.offsets.cpu(), "val",
                                 snapshot, fused_logits=fused)
            report.update(epoch=epoch, candidate_mode=args.candidate_mode)
            write_json(snapshot / "results.json", report)
        if best is None or key > best[0]:
            best = key, epoch, fused, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    _, epoch, fused, weights = best
    head.load_state_dict(weights)
    report = diagnostics(source, validation, *val_columns, model.offsets.cpu(), "val", output, fused_logits=fused)
    report.update(best_epoch=epoch, history=history, source_checkpoint=str(args.checkpoint), seed=args.seed,
                  candidate_mode=args.candidate_mode, candidate_max_delta=args.max_delta,
                  candidate_parameters=sum(p.numel() for p in head.parameters()),
                  train_rows=len(train), train_predictions="full_train_checkpoint", selection="validation_fallback")
    checkpoint.update(model_args=dict(model_args, candidate_mode=args.candidate_mode,
                                      candidate_max_delta=args.max_delta), epoch=epoch,
                      model_state={k: v.detach().cpu() for k, v in model.state_dict().items()})
    torch.save(checkpoint, output / "best.pt")
    write_json(output / "results.json", report)


def evaluate(args, output):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME:
        raise ValueError("Use the packed TriEvidence-MFP best.pt")
    source, dataset = open_data(args, args.split, checkpoint)
    if "shared_mode" in checkpoint["model_args"]:
        if args.reuse_input_compute or args.pack_esm_projections:
            raise ValueError("Input reuse applies to the original two-encoder TriEvidence")
        from mfp_nova.models.trievidence_shared import SharedTriEvidence
        model = SharedTriEvidence(**checkpoint["model_args"])
    else:
        model = TriEvidence(**checkpoint["model_args"])
        model.reuse_input_compute = args.reuse_input_compute
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device).eval()
    if args.pack_esm_projections:
        model.pack_esm_projections()

    latent_states = []
    if args.export_latent or args.export_latent_tokens:
        if args.split != "val" or "shared_mode" in checkpoint["model_args"]:
            raise ValueError("Latent export requires original TriEvidence on Validation")

        def capture_latent(_layer, _inputs, states):
            if args.export_latent_tokens:
                states = states.to(dtype=torch.float16)
            else:
                states = states.mean(1).float()
            latent_states.append(states.cpu())

        latent_hook = model.helixset.latent_blocks[-1].register_forward_hook(capture_latent)

    def read(batch):
        result = model(batch)
        columns = result["theraset_logits"], result["aa34_logits"], result["latent_logits"]
        return (*columns, result["logits"]) if checkpoint["model_args"].get("candidate_mode") else columns

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    inference_data = dataset
    if args.length_bucketed:
        lengths = [len(source.data.sequences[i]) for i in dataset.indices]
        order = np.argsort(lengths, kind="stable")
        inference_data = Subset(dataset, order.tolist())
    columns = collect(read, batches(inference_data, args.batch_size), args.device,
                      progress=True, timings=timings)
    if args.length_bucketed:
        original_order = torch.from_numpy(np.argsort(order))
        columns = tuple(column[original_order] for column in columns)
    if args.export_latent or args.export_latent_tokens:
        latent_hook.remove()
        states = torch.cat(latent_states)
        if args.length_bucketed:
            states = states[original_order]
        rows = source.data.rows[dataset.indices]
        if args.export_latent_tokens:
            np.savez_compressed(output / "val_latent_tokens.npz", row_id=rows, latent_tokens=states.numpy())
        if args.export_latent:
            pooled = states.float().mean(1) if args.export_latent_tokens else states
            np.savez_compressed(output / "val_latent_features.npz", row_id=rows, pooled_latent=pooled.numpy())
    a, b, h, *corrected = columns
    report = diagnostics(source, dataset, a, b, h, model.offsets.cpu(), args.split, output,
                         shared="shared_mode" in checkpoint["model_args"],
                         fused_logits=corrected[0] if corrected else None)
    steady = timings[5:-1] or timings
    report.update(checkpoint=str(args.checkpoint), best_epoch=checkpoint["epoch"], anchors=checkpoint["anchors"],
                  latent_recovery_epoch=checkpoint.get("latent_recovery_epoch"),
                  parameters=sum(p.numel() for p in model.parameters()),
                  inference={"batch_size": args.batch_size, "device": args.device,
                             "reuse_input_compute": getattr(model, "reuse_input_compute", False),
                             "pack_esm_projections": args.pack_esm_projections,
                             "length_bucketed": args.length_bucketed,
                             "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                             "seconds": sum(timings), "batch_p50_seconds": float(np.median(steady)),
                             "batch_p95_seconds": float(np.quantile(steady, .95)),
                             "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
                             if device.type == "cuda" else None})
    write_json(output / "results.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train", help="Fit the independent CNN and select on original Validation")
    fit.add_argument("--theraset", type=Path, required=True)
    fit.add_argument("--helixset", type=Path, required=True)
    fit.add_argument("--reader", type=Path, help="Pack a previously fitted reader and replay Validation")
    cached = commands.add_parser("train-cached", help="Train only the reader using saved original Validation scores")
    cached.add_argument("--theraset-val", type=Path, required=True)
    cached.add_argument("--helix-exits", type=Path, required=True)
    cached.add_argument("--helix-threshold", type=float, default=.64)
    cached.add_argument("--distill-reader", type=Path, help="Frozen teacher reader.pt; keep its offset and export the final epoch")
    cached.add_argument("--distill-weight", type=float, default=.5)
    for command in (fit, cached):
        command.add_argument("--epochs", type=int, default=25)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--cnn-batch-size", type=int, default=128)
        command.add_argument("--learning-rate", type=float, default=3e-4)
        command.add_argument("--unlabeled-weight", type=float, default=.3)
        command.add_argument("--channels", type=int, default=128)
        command.add_argument("--dropout", type=float, default=.2)
        command.add_argument("--reader-head", choices=READER_HEADS, default="global")
        command.add_argument("--review-weight", type=float, default=.25,
                             help="Dual AA34 auxiliary loss weight on high-scoring unrecorded labels; 0 uses equal supervision")
    candidate = commands.add_parser("fit-candidates", help="Fit a bounded candidate readout on frozen TriEvidence")
    candidate.add_argument("checkpoint", type=Path)
    candidate.add_argument("--candidate-mode", choices=("scores", "local"), required=True)
    candidate.add_argument("--epochs", type=int, default=5)
    candidate.add_argument("--seed", type=int, default=42)
    candidate.add_argument("--learning-rate", type=float, default=3e-4)
    candidate.add_argument("--unlabeled-weight", type=float, default=.3)
    candidate.add_argument("--candidate-batch-size", type=int, default=128)
    candidate.add_argument("--max-delta", type=float, default=1.)
    candidate.add_argument("--probe-memory", action="store_true",
                           help="Measure fixed-branch forward and candidate backward on long Train batches")
    candidate.add_argument("--export-first-epoch", action="store_true",
                           help="Export per-row Validation scores before and after the first epoch")
    score = commands.add_parser("evaluate", help="Evaluate a packed checkpoint with its saved cutoff")
    score.add_argument("checkpoint", type=Path)
    score.add_argument("--split", choices=("val", "test"), default="val")
    score.add_argument("--length-bucketed", action="store_true",
                       help="Batch similar lengths together and restore original row order for metrics")
    score.add_argument("--reuse-input-compute", action="store_true",
                       help="Reuse equivalent dual-ESM normalization and residue distances in TriEvidence")
    score.add_argument("--pack-esm-projections", action="store_true",
                       help="Reuse inputs and execute both original ESM projections in one linear operation")
    score.add_argument("--export-latent", action="store_true", help="Save pooled HelixSet Validation features")
    score.add_argument("--export-latent-tokens", action="store_true",
                       help="Save all final HelixSet latent tokens in float16 on Validation")
    for command in (fit, cached, candidate, score):
        command.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output", type=Path, help="Fresh run directory")
    for command in (fit, candidate, score):
        command.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
        command.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
        command.add_argument("--batch-size", type=int, default=16, help="Frozen branch inference batch size")
    args = parser.parse_args()
    if args.command != "evaluate" and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.command == "fit-candidates" and not 0 < args.max_delta < float("inf"):
        parser.error("--max-delta must be positive and finite")
    suffix = args.command.replace("-", "_") if args.command != "evaluate" else args.split
    output = args.output or Path("runs") / datetime.now().strftime(f"trievidence_{suffix}_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    print(f"TriEvidence-MFP | output={output}", flush=True)
    if args.command == "train":
        train(args, output)
    elif args.command == "train-cached":
        train_cached(args, output)
    elif args.command == "fit-candidates":
        fit_candidates(args, output)
    else:
        evaluate(args, output)


if __name__ == "__main__":
    main()
