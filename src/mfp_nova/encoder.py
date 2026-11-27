"""Frozen PLM extraction (ESM-2 & T5/ProtT5/ProstT5), shared by feature preparation and prediction."""
import json
from pathlib import Path

import numpy as np
import torch

from .data import PeptideData
from .therapep import TRACKS as THERAPEP_TRACKS

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


class FrozenEncoder:
    def __init__(self, name, revision="main", max_length=80, device="cpu", representation_layer=None):
        from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel
        self.device = torch.device(device)
        self.max_length = max_length
        cfg = AutoConfig.from_pretrained(name, revision=revision)
        self.is_t5 = getattr(cfg, "model_type", None) == "t5"
        self.is_prost_t5 = False
        if self.is_t5:
            if representation_layer is not None:
                raise ValueError("Prefix layer extraction is only supported for ESM encoders")
            self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, do_lower_case=False, use_fast=False)
            vocab = self.tokenizer.get_vocab()
            self.is_prost_t5 = "<AA2fold>" in vocab
            self.t5_aa_ids = set()
            for aa in ALPHABET:
                for tok in (f"\u2581{aa}", aa):
                    if tok in vocab:
                        self.t5_aa_ids.add(vocab[tok])
            # Force FP32 execution even if the HF repo config specifies float16 (avoids T5 FP16 overflow on V100).
            self.model = T5EncoderModel.from_pretrained(name, revision=revision).float()
            hidden_dim = int(self.model.config.d_model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
            options = {"attn_implementation": "eager"} if representation_layer is not None else {}
            self.model = AutoModel.from_pretrained(name, revision=revision, add_pooling_layer=False, **options)
            if representation_layer is not None:
                if not 0 < representation_layer < self.model.config.num_hidden_layers:
                    raise ValueError("Prefix extraction requires an intermediate ESM layer")
                self.model.encoder.layer = torch.nn.ModuleList(list(self.model.encoder.layer)[:representation_layer])
                self.model.encoder.emb_layer_norm_after = torch.nn.Identity()
            hidden_dim = int(self.model.config.hidden_size)
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.metadata = {"name": name, "requested_revision": revision,
                         "resolved_revision": self.model.config._commit_hash,
                         "sequence_view": "prost_t5_aa2fold_uppercase" if self.is_prost_t5 else "uppercase_resolved_backbone",
                         "max_length": max_length, "input_dim": hidden_dim,
                         "extraction_dtype": "float32", "storage_dtype": "float16"}
        if representation_layer is not None:
            self.metadata["representation_layer"] = representation_layer
            self.metadata["special_tokens_cached"] = True

    @torch.inference_mode()
    def encode(self, sequences):
        # Case carries stereochemical annotations in the dataset. PLM consumes
        # residue identities only; source strings and D masks remain unchanged.
        sequences = [s.upper() for s in sequences]
        if any(not s or set(s) - set(ALPHABET) for s in sequences):
            raise ValueError("Encoder requires a nonempty, resolved 20-AA backbone")
        if self.is_t5:
            prefix = "<AA2fold> " if self.is_prost_t5 else ""
            formatted = [prefix + " ".join(s) for s in sequences]
            tokens = self.tokenizer(formatted, padding=True, truncation=False, return_tensors="pt")
            if tokens["input_ids"].shape[1] > self.max_length:
                raise ValueError("Sequence exceeds configured token limit; no silent truncation")
            input_ids = tokens["input_ids"]
            attn = tokens["attention_mask"].bool()
            offset = 1 if self.is_prost_t5 else 0
            expected_extra = 2 if self.is_prost_t5 else 1  # [<AA2fold>] + L residues + </s>
            mask = torch.zeros_like(attn)
            for j, seq in enumerate(sequences):
                valid_len = int(attn[j].sum().item())
                if valid_len != len(seq) + expected_extra:
                    raise ValueError(f"T5 token count {valid_len} != expected {len(seq) + expected_extra}")
                res_ids = input_ids[j, offset:offset + len(seq)].tolist()
                if not set(res_ids).issubset(self.t5_aa_ids):
                    raise ValueError("T5 residue slice contains non-amino-acid or special token IDs")
                mask[j, offset:offset + len(seq)] = True
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            mask = mask.to(self.device)
            states = self.model(**tokens).last_hidden_state.float()
        else:
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
        if not torch.isfinite(states).all() or states.abs().max().item() > 60000.0:
            raise FloatingPointError("Encoder produced nonfinite or FP16-overflowing hidden states")
        return states, mask


def prepare_features(config, splits, device, resume_from=0, encoder_view="primary"):
    section_map = {"primary": ("encoder", "features"),
                   "peptide": ("peptide_encoder", "peptide_features"),
                   "orth": ("orth_encoder", "orth_features")}
    if encoder_view not in section_map:
        raise ValueError(f"Unknown encoder_view: {encoder_view}")
    enc_key, feat_key = section_map[encoder_view]
    settings = config[enc_key]
    feature_root = config["data"][feat_key]
    prefix = settings.get("representation_layer")
    encoder = FrozenEncoder(settings["name"], settings["revision"], settings["max_length"], device, prefix)
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
        root = Path(feature_root) / split
        lengths = np.array([len(s) for s in data.sequences[indices]])
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        shape = (int(offsets[-1]), encoder.metadata["input_dim"])
        if start_row:
            if (root / "metadata.json").exists():
                raise FileExistsError(f"Feature cache is already complete: {root}")
            values = np.load(root / "residues.npy", mmap_mode="r+")
            if values.shape != shape or values.dtype != np.float16:
                raise ValueError("Incomplete cache shape/dtype differs from the current extraction")
            if prefix is not None:
                special_values = np.load(root / "special_tokens.npy", mmap_mode="r+")
            print(f"{split}: continuing from {start_row}/{len(indices)}", flush=True)
        else:
            root.mkdir(parents=True, exist_ok=False)
            values = np.lib.format.open_memmap(root / "residues.npy", mode="w+", dtype=np.float16, shape=shape)
            if prefix is not None:
                special_values = np.lib.format.open_memmap(root / "special_tokens.npy", mode="w+", dtype=np.float16,
                                                          shape=(len(indices), 2, encoder.metadata["input_dim"]))
        batch_size = settings["batch_size"]
        for start in range(start_row, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            states, mask = encoder.encode(data.sequences[batch_indices].tolist())
            for j in range(len(batch_indices)):
                pos = start + j
                values[offsets[pos]:offsets[pos + 1]] = states[j, mask[j]].cpu().numpy()
                if prefix is not None:
                    special_values[pos, 0] = states[j, 0].cpu().numpy()
                    special_values[pos, 1] = states[j, len(data.sequences[batch_indices[j]]) + 1].cpu().numpy()
            print(f"{split}: {min(start + batch_size, len(indices))}/{len(indices)}", flush=True)
        values.flush()
        if prefix is not None:
            special_values.flush()
        np.save(root / "offsets.npy", offsets)
        np.save(root / "row_ids.npy", data.rows[indices])
        np.save(root / "ids.npy", data.ids[indices])
        np.save(root / "sequences.npy", data.sequences[indices])
        metadata = {**encoder.metadata, "split": split, "rows": len(indices),
                    "dataset": (data.dataset_id if data.track in THERAPEP_TRACKS else
                                json.loads((data.root / "manifest.json").read_text(encoding="utf-8"))["version_date"])}
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
