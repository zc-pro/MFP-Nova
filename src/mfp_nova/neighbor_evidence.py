"""Fine39 annotation evidence from representatives of eligible Train clusters."""

import numpy as np
import torch
from torch.nn.functional import normalize

from .data import CachedPeptides
from .hypothesis_data import family_folds


NEIGHBORS = 16


def pooled_cache(cache):
    vectors = np.stack([
        cache.values[cache.offsets[pos]:cache.offsets[pos + 1]].mean(axis=0, dtype=np.float32)
        for pos in cache.positions
    ])
    return normalize(torch.from_numpy(vectors))


def neighbor_tensors(queries, library, labels):
    """Recorded vote rates, best positive similarities and nearest similarity."""
    match = (queries @ library.T).topk(min(NEIGHBORS, len(library)), dim=1)
    observed = labels[match.indices]
    best = match.values[..., None].masked_fill(~observed, -torch.inf).amax(1)
    return observed.float().mean(1), best.masked_fill(~torch.isfinite(best), 0.), match.values[:, 0]


def read_neighbors(queries, library, labels):
    labels = torch.as_tensor(labels)
    evidence = []
    for batch in queries.split(256):
        evidence.append(neighbor_tensors(batch, library, labels))
    return tuple(torch.cat(parts).numpy() for parts in zip(*evidence))


def fold_neighbor_evidence(data, feature_root, train_rows, val_rows, include_reference=False):
    train_cache, val_cache = (CachedPeptides(data, feature_root, split) for split in ("train", "val"))
    keys = ("resolved_revision", "input_dim", "max_length", "sequence_view")
    if any(train_cache.meta[key] != val_cache.meta[key] for key in keys):
        raise ValueError("Train and Validation neighbor caches use different encoder definitions")
    train = train_cache.indices
    train_groups = data.clusters[train]
    if np.intersect1d(train_groups, data.clusters[val_rows]).size:
        raise ValueError("Train and Validation overlap in homology clusters")
    folds = family_folds(train_groups, 3)
    # The cache indices are in the original data row order. Use the first row
    # in each cluster so that representative selection is label independent.
    _, representatives = np.unique(train_groups, return_index=True)
    train_vectors, val_vectors = pooled_cache(train_cache), pooled_cache(val_cache)
    train_pos = np.searchsorted(train, train_rows)
    val_pos = np.searchsorted(val_cache.indices, val_rows)
    labels = data.y[train].astype(bool)
    train_evidence = (np.empty((len(train_rows), len(data.labels)), dtype=np.float32),
                      np.empty((len(train_rows), len(data.labels)), dtype=np.float32),
                      np.empty(len(train_rows), dtype=np.float32))
    for fold in range(3):
        held = folds[train_pos] == fold
        reps = representatives[folds[representatives] != fold]
        if held.any():
            evidence = read_neighbors(train_vectors[train_pos[held]], train_vectors[reps], labels[reps])
            for output, values in zip(train_evidence, evidence):
                output[held] = values
    reference = train_vectors[representatives], torch.from_numpy(labels[representatives])
    val_evidence = read_neighbors(val_vectors[val_pos], *reference)
    if include_reference:
        return train_evidence, val_evidence, reference
    return train_evidence, val_evidence
