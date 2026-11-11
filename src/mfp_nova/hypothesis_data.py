"""Read aligned annotation evidence and batch original, full-length sequences."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .data import PeptideData
from .models.hypothesis_backbone import pad_tokens, tokenize


TASKS = ("major15", "fine39", "rich48")


def family_folds(groups, count):
    names, inverse, sizes = np.unique(groups, return_inverse=True, return_counts=True)
    load = np.zeros(count, dtype=int)
    allocation = np.empty(len(names), dtype=int)
    for group in np.argsort(-sizes, kind="stable"):
        fold = load.argmin()
        allocation[group] = fold
        load[fold] += sizes[group]
    return allocation[inverse]


class SequenceStudyData(PeptideData):
    def __init__(self, root, track, annotations=None, read_sources=False):
        if track not in TASKS:
            raise ValueError(f"B0/H1–H8 tasks are {', '.join(TASKS)}")
        super().__init__(root, track)
        self.targets = self.y.copy()
        self.observed = self.y.astype(bool)
        self.audited = np.zeros(self.y.shape, dtype=bool)
        self.partial_ignorable = np.zeros(len(self.y), dtype=bool)
        self.sources = np.full(len(self.y), "", dtype=object)
        self.lengths = np.fromiter((len(seq.strip()) for seq in self.sequences), dtype=int, count=len(self.rows))
        if read_sources:
            self.read_sources()
        with np.load(self.root / "tracks" / track / "dataset.npz", allow_pickle=False) as archive:
            self.folds = archive["cv_fold"] if "cv_fold" in archive else None
            if "observed" in archive:
                observed = archive["observed"]
                if observed.shape != self.y.shape or not np.isin(observed, (0, 1)).all():
                    raise ValueError("The task's observed mask must align with its binary targets")
                self.observed = observed.astype(bool)
        if annotations:
            self.read_annotations(annotations)
        self._tokens = {}

    def read_sources(self):
        path = self.root / "records.csv"
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                records = csv.DictReader(stream)
                sources = {int(row["row_id"]): row.get("Source", row.get("source", "")) for row in records}
            self.sources[:] = [sources.get(int(row), "") for row in self.rows]

    def read_annotations(self, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            if archive["label_names"].tolist() != self.labels:
                raise ValueError("Annotation label order differs from the selected task")
            row_ids = archive["row_id"]
            if row_ids.ndim != 1 or not np.issubdtype(row_ids.dtype, np.integer) or len(np.unique(row_ids)) != len(row_ids):
                raise ValueError("Annotation row_id values must be unique")
            lookup = {int(row): pos for pos, row in enumerate(self.rows)}
            try:
                indices = np.array([lookup[int(row)] for row in row_ids], dtype=int)
            except KeyError as error:
                raise ValueError(f"Annotation row_id {error.args[0]} is absent from the selected task") from error
            if "ID" in archive and not np.array_equal(archive["ID"], self.ids[indices]):
                raise ValueError("Annotation IDs disagree with their row_id values")
            if "Y" in archive and "observed" not in archive:
                raise ValueError("Revised annotation targets require their observed mask")
            shape = (len(indices), len(self.labels))
            for key, field in (("observed", "observed"), ("Y", "targets"), ("audited", "audited"),
                               ("partial_ignorable", "partial_ignorable")):
                if key in archive:
                    values = archive[key]
                    expected = (len(indices),) if key == "partial_ignorable" else shape
                    if values.shape != expected or not np.isin(values, (0, 1)).all():
                        raise ValueError(f"Annotation {key} must be binary with shape {expected}")
                    getattr(self, field)[indices] = values
            for key in ("source_id", "group_id"):
                if key in archive and archive[key].shape != row_ids.shape:
                    raise ValueError(f"Annotation {key} must align with row_id")
            if "source_id" in archive:
                self.sources[indices] = archive["source_id"].astype(str)
            if "group_id" in archive:
                if len(indices) != len(self.rows):
                    raise ValueError("A replacement homology grouping must cover the entire dataset")
                self.clusters = self.clusters.astype(object)
                self.clusters[indices] = archive["group_id"].astype(str)
                self.folds = None

    def train_folds(self, indices, count):
        if count == 5 and self.folds is not None:
            groups = self.clusters[indices].tolist()
            folds = self.folds[indices]
            if len(set(zip(groups, folds.tolist()))) != len(set(groups)) or not np.isin(folds, np.arange(5)).all():
                raise ValueError("Released training folds disagree with the current homology groups")
            return folds
        return family_folds(self.clusters[indices], count)

    def tokens(self, indices):
        sequences = []
        for i in indices:
            if i not in self._tokens:
                self._tokens[i] = tokenize(self.sequences[i])
            sequences.append(self._tokens[i])
        return pad_tokens(sequences)


class StudyRows(Dataset):
    def __init__(self, data, indices):
        self.indices = np.asarray(indices)
        self.tokens = [tokenize(data.sequences[i]) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        return int(self.indices[position]), self.tokens[position]


def collate_rows(batch):
    indices, tokens = zip(*batch)
    return np.asarray(indices), *pad_tokens(tokens)


class LengthBatches(Sampler):
    def __init__(self, lengths, batch_size, shuffle, seed):
        self.order = np.argsort(lengths, kind="stable")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = self.order.copy()
        if self.shuffle:
            # Mix adjacent batches with similar lengths and shuffle the resulting batches.
            width = 16 * self.batch_size
            for start in range(0, len(order), width):
                self.rng.shuffle(order[start:start + width])
        batches = [order[start:start + self.batch_size].tolist() for start in range(0, len(order), self.batch_size)]
        if self.shuffle:
            self.rng.shuffle(batches)
        yield from batches


def study_loader(data, indices, batch_size, *, shuffle=False, seed=0, workers=0):
    batches = LengthBatches(data.lengths[indices], batch_size, shuffle, seed)
    return DataLoader(StudyRows(data, indices), batch_sampler=batches, collate_fn=collate_rows,
                      num_workers=workers, persistent_workers=bool(workers))
