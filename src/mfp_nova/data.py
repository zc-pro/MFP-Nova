"""Read the released data contract; never modify input arrays."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from .therapep import TRACKS as THERAPEP_TRACKS

TRACKS = ("major15", "fine39", *THERAPEP_TRACKS)


class PeptideData:
    def __init__(self, root, track="fine39"):
        self.root = Path(root)
        self.track = track
        if track in THERAPEP_TRACKS:
            from .therapep import read_data
            self.__dict__.update(read_data(self.root, track))
            return
        with np.load(self.root / "arrays/sequences.npz", allow_pickle=False) as source:
            self.rows = source["row_id"]
            self.ids = source["ID"]
            self.sequences = source["canonical_backbone"]
            self.available = source["canonical_available"]
            self.d_mask = source["D_annotation_mask"]
            self.raw_sequences = source["raw_sequence"]
        with np.load(self.root / f"tracks/{track}/dataset.npz", allow_pickle=False) as source:
            if not np.array_equal(self.rows, source["row_id"]):
                raise ValueError("Sequence and label row_id order differs")
            self.y = source["Y"]
            self.labels = source["label_names"].tolist()
            self.splits = source["split"]
            self.clusters = source["cluster_id"]
            self.augmented = source["is_augmented"]
            self.in_scope = source["row_in_scope"]
            self.masks = {key: source[key] for key in (
                "train_original_mask", "val_original_mask", "test_original_mask", "train_aug_mask")}
        if not np.isin(self.y, [0, 1]).all():
            raise ValueError("Expected binary observed-label targets")

    def indices(self, split, augment=False, sequence_only=True):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if augment and split != "train":
            raise ValueError("Augmentation is train-only")
        if augment and self.track in THERAPEP_TRACKS:
            raise ValueError("TheraPep tasks preserve published rows; no augmentation is defined")
        key = "train_aug_mask" if augment else f"{split}_original_mask"
        mask = self.masks[key].copy()
        if sequence_only:
            mask &= self.available
        return np.flatnonzero(mask)


class CachedPeptides(Dataset):
    def __init__(self, data, feature_root, split, augment=False):
        root = Path(feature_root) / split
        self.meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.values = np.load(root / "residues.npy", mmap_mode="r")
        self.special_tokens = (np.load(root / "special_tokens.npy", mmap_mode="r")
                               if self.meta.get("special_tokens_cached") else None)
        self.offsets = np.load(root / "offsets.npy")
        rows = np.load(root / "row_ids.npy")
        ids = np.load(root / "ids.npy")
        sequences = np.load(root / "sequences.npy")
        self.indices = data.indices(split, augment=augment)
        lookup = {int(row): i for i, row in enumerate(rows)}
        self.positions = np.array([lookup[int(data.rows[i])] for i in self.indices])
        if not np.array_equal(ids[self.positions], data.ids[self.indices]):
            raise ValueError("Feature IDs do not match target IDs")
        if not np.array_equal(sequences[self.positions], data.sequences[self.indices]):
            raise ValueError("Cached backbones differ from current sequences")
        self.targets = data.y[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        pos = self.positions[index]
        residues = torch.from_numpy(np.array(self.values[self.offsets[pos]:self.offsets[pos + 1]], dtype=np.float32))
        return residues, torch.tensor(self.targets[index], dtype=torch.float32)


def collate_features(samples):
    residues, targets = zip(*samples)
    lengths = torch.tensor([len(item) for item in residues])
    padded = torch.nn.utils.rnn.pad_sequence(residues, batch_first=True)
    mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
    return padded, mask, torch.stack(targets)
