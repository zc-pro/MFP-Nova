"""Mine difficult labels from aligned cluster-held-out Train predictions."""

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Subset

from mfp_nova.hh_v13 import load_oof
from mfp_nova.hypothesis_data import family_folds
from mfp_nova.models.trievidence import AA34Reader, center_evidence
from mfp_nova.multimodal import BackbonePeptides, MultimodalData, label_statistics
from mfp_nova.research_layers import unlabeled_weights
from mfp_nova.trainer import seed_everything, write_json
from run_trievidence import batches, collect, pass_train


def mine(targets, scores, clusters, min_clusters):
    candidates = np.where(~targets, scores, -np.inf).argmax(1)
    ahead = scores[np.arange(len(scores)), candidates, None] > scores
    observed_rows, observed_labels = np.where(targets & ahead)
    support = {}
    for row, label in zip(observed_rows, observed_labels):
        support.setdefault((label, candidates[row]), set()).add(clusters[row])
    pairs = np.zeros_like(targets)
    for row, label in zip(observed_rows, observed_labels):
        pairs[row, label] = len(support[label, candidates[row]]) >= min_clusters
    return candidates, pairs, support


def prepare(args, output, checkpoint):
    source = MultimodalData(args.data_root, "fine39", checkpoint["statistics"],
                            localize_terminals=checkpoint["statistics"].get("localized_terminal_chemistry", False))
    dataset = BackbonePeptides(source, "train", checkpoint["encoder"]["max_length"])
    indices, thera = load_oof(source.data, args.theraset_oof)
    np.testing.assert_array_equal(indices, dataset.indices)
    clusters = source.data.clusters[indices]
    targets = source.data.y[indices].astype(bool)
    folds = family_folds(clusters, 3)
    aa34 = np.empty_like(thera)
    for fold in range(3):
        seed_everything(args.seed)
        fit_mask = folds != fold
        stats = label_statistics(targets[fit_mask].astype(np.float32), clusters[fit_mask], source.data.labels)
        reader = AA34Reader(stats).to(args.device)
        optimizer = torch.optim.AdamW(reader.parameters(), lr=3e-4, weight_decay=.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.reader_epochs)
        fit = batches(Subset(dataset, np.flatnonzero(fit_mask)), args.batch_size, shuffle=True)
        held = batches(Subset(dataset, np.flatnonzero(folds == fold)), args.batch_size)
        for epoch in range(1, args.reader_epochs + 1):
            loss = pass_train(reader, fit, args.device, optimizer, stats, .3)
            schedule.step()
            print(f"AA34 OOF fold={fold} epoch={epoch} loss={loss:.6f}", flush=True)
        reader.eval()
        aa34[folds == fold] = collect(lambda batch: (reader(batch),), held, args.device)[0].numpy()

    # A branch-wide offset cancels when comparing labels in the same row.
    scores = center_evidence(torch.from_numpy(np.stack((thera, aa34))), torch.zeros(2)).mean(0).numpy()
    candidates, pairs, support = mine(targets, scores, clusters, args.min_pair_clusters)
    labels = source.data.labels
    write_json(output / "hard_pairs.json", {
        "min_pair_clusters": args.min_pair_clusters,
        "rows": len(targets), "paired_rows": int(pairs.any(1).sum()),
        "pairs": [{"observed": str(labels[i]), "competitor": str(labels[j]), "clusters": len(seen)}
                  for (i, j), seen in sorted(support.items(), key=lambda item: -len(item[1]))],
    })
    np.savez_compressed(output / "hard_pairs.npz", row_id=source.data.rows[indices],
                        label_names=labels, targets=targets, competitors=candidates, pairs=pairs)
    print(output / "hard_pairs.npz", flush=True)


def hard_pair_loss(path, source, training, device, weight):
    indices = training.indices
    with np.load(path) as saved:
        np.testing.assert_array_equal(saved["row_id"], source.data.rows[indices])
        np.testing.assert_array_equal(saved["label_names"], source.data.labels)
        np.testing.assert_array_equal(saved["targets"], source.data.y[indices])
        ids = torch.as_tensor(saved["row_id"], device=device)
        ids, order = ids.sort()
        competitors = torch.as_tensor(saved["competitors"], device=device)[order, None]
        pairs = torch.as_tensor(saved["pairs"], device=device)[order]
    if not pairs.any():
        raise ValueError("Hard-pair export has no recurring competitors")

    def auxiliary(logits, batch):
        positions = torch.searchsorted(ids, batch["row_id"])
        competitor = competitors[positions]
        anchor = logits.detach().gather(1, competitor)
        weak = unlabeled_weights(batch["y"], source.statistics["labels"], .3).gather(1, competitor)
        active = pairs[positions]
        return weight * (F.softplus(anchor - logits) * active * weak).sum() / active.sum().clamp_min(1)

    return auxiliary
