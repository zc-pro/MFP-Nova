"""Frozen ESM extraction, shared by training feature preparation and prediction."""
import json
from pathlib import Path

import numpy as np
import torch

from .data import PeptideData


class FrozenEncoder:
    def __init__(self, name, revision="main", max_length=80, device="cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModel.from_pretrained(name, revision=revision, add_pooling_layer=False).to(self.device).eval()
        self.model.requires_grad_(False)
        self.metadata = {"name": name, "requested_revision": revision,
                         "resolved_revision": self.model.config._commit_hash,
                         "sequence_view": "uppercase_resolved_backbone",
                         "max_length": max_length, "input_dim": self.model.config.hidden_size,
                         "extraction_dtype": "float32", "storage_dtype": "float16"}

    @torch.inference_mode()
    def encode(self, sequences):
        # Case carries stereochemical annotations in the dataset. ESM consumes
        # residue identities only; source strings and D masks remain unchanged.
        sequences = [s.upper() for s in sequences]
        if any(not s or set(s) - set("ACDEFGHIKLMNPQRSTVWY") for s in sequences):
            raise ValueError("Encoder requires a nonempty, resolved 20-AA backbone")
        tokens = self.tokenizer(sequences, padding=True, truncation=False,
                                return_special_tokens_mask=True, return_tensors="pt")
        if tokens["input_ids"].shape[1] > self.max_length:
            raise ValueError("Sequence exceeds configured token limit; no silent truncation")
        special = tokens.pop("special_tokens_mask").bool().to(self.device)
        tokens = tokens.to(self.device)
        mask = tokens["attention_mask"].bool() & ~special
        if mask.sum(1).cpu().tolist() != [len(s) for s in sequences]:
            raise ValueError("Tokenizer residue count does not match sequence length")
        states = self.model(**tokens).last_hidden_state.float()
        return states, mask


def prepare_features(config, splits, device, resume_from=0):
    settings = config["encoder"]
    encoder = FrozenEncoder(settings["name"], settings["revision"], settings["max_length"], device)
    data = PeptideData(config["data"]["root"], config["data"]["track"])
    for split in splits:
        # Share one original-row cache between both tasks, including fine39 out-of-scope rows.
        selected = (data.splits == split) & data.available & ~data.augmented
        if split == "train" and config["data"]["augment"]:
            selected |= data.masks["train_aug_mask"] & data.available
        indices = np.flatnonzero(selected)
        start_row = resume_from if split == splits[0] else 0
        if not 0 <= start_row <= len(indices):
            raise ValueError("resume_from must be a completed row count within the first split")
        root = Path(config["data"]["features"]) / split
        lengths = np.array([len(s) for s in data.sequences[indices]])
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        shape = (int(offsets[-1]), encoder.metadata["input_dim"])
        if start_row:
            if (root / "metadata.json").exists():
                raise FileExistsError(f"Feature cache is already complete: {root}")
            values = np.load(root / "residues.npy", mmap_mode="r+")
            if values.shape != shape or values.dtype != np.float16:
                raise ValueError("Incomplete cache shape/dtype differs from the current extraction")
            print(f"{split}: continuing from {start_row}/{len(indices)}", flush=True)
        else:
            root.mkdir(parents=True, exist_ok=False)
            values = np.lib.format.open_memmap(root / "residues.npy", mode="w+", dtype=np.float16, shape=shape)
        batch_size = settings["batch_size"]
        for start in range(start_row, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            states, mask = encoder.encode(data.sequences[batch_indices].tolist())
            for j in range(len(batch_indices)):
                pos = start + j
                values[offsets[pos]:offsets[pos + 1]] = states[j, mask[j]].cpu().numpy()
            print(f"{split}: {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
        values.flush()
        np.save(root / "offsets.npy", offsets)
        np.save(root / "row_ids.npy", data.rows[indices])
        np.save(root / "ids.npy", data.ids[indices])
        np.save(root / "sequences.npy", data.sequences[indices])
        metadata = {**encoder.metadata, "split": split, "rows": len(indices),
                    "dataset": json.loads((data.root / "manifest.json").read_text(encoding="utf-8"))["version_date"]}
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
