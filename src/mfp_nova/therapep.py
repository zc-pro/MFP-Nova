"""The authors' published TheraPep-AI CSV cohort, separate from therapeutic_2025."""
import csv
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch


TRACK = "therapep49"
DATASET = "therapep-ai-published-assets"
RELEASE = "https://github.com/terra-quantum-public/tq-therapep-ai/releases/tag/assets"
DOWNLOAD = "https://github.com/terra-quantum-public/tq-therapep-ai/releases/download/assets"
# Published column order, including both differently capitalized growth aliases.
LABELS = [
    "Analgesic", "Angiogenic", "Anti-gram+", "Anti-gram-", "AntiAngiogenesis",
    "AntiBreastcancer", "AntiCervicalcancer", "AntiColoncancer", "AntiHCV", "AntiHIV",
    "AntiHSV", "AntiLivercancer", "AntiLungcancer", "AntiMERS-Cov", "AntiProstatecancer",
    "AntiSARS", "AntiSkincancer", "AntiTubercular", "Antibacterial", "Anticancer",
    "Antifungal", "Antihypertensive", "Antiinflammatory", "Antileishmania", "Antimalarial",
    "Antimicrobial", "Antioxidant", "Antiparasitic", "Antiplasmodial", "Antitrypanosomic",
    "Antiviral", "Blood-brain_barrier_penetrating", "Cell_Communication", "Cell_penetrating",
    "Drug_delivery", "Glucose_metabolism", "Growth_Regeneration", "Growth_regeneration",
    "Growth_regulatory", "Immunoactive", "Lipid_metabolism", "Metabolic_regulatory",
    "Neuropeptide", "Opioid", "Osteogenic", "Quorum_sensing", "Thrombolytic", "Tumor_homing",
    "is_therapeutic",
]


def configure_task(config, task):
    """Use separate inputs/caches for the explicit new task; keep custom paths."""
    config["data"]["track"] = task
    if task != TRACK:
        return
    data = config["data"]
    if data["root"] == "data/therapeutic_2025":
        data["root"] = "data/therapep_ai"
    for key in ("features", "peptide_features"):
        if key in data and data[key].startswith("features/esm2_"):
            data[key] = "features/therapep_ai/" + Path(data[key]).name
    data["augment"] = False
    # All rows have the same sequence-only modality scope. Adding structure only
    # to positive peptides would reveal the synthetic-background target.
    for key in ("property_evidence", "structure_evidence", "chemical_evidence"):
        if key in config.get("model", {}):
            config["model"][key] = False
    if "model_name" in config:
        config.setdefault("loss", {})["modality_dropout"] = 0.
        config.setdefault("train", {})["metric_suite"] = "micro_f1"


def download_data(root):
    """Download only the three published CSVs; never overwrite existing inputs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        path = root / f"{split}.csv"
        if path.exists():
            continue
        partial = path.with_suffix(".csv.part")
        with urllib.request.urlopen(f"{DOWNLOAD}/{split}.csv") as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(path)
    return describe_data(read_data(root))


def read_data(root):
    """Preserve CSV membership, sequence, labels and duplicate rows verbatim."""
    sequences, labels, splits, ids, sources = [], [], [], [], []
    for split in ("train", "val", "test"):
        with (Path(root) / f"{split}.csv").open(newline="", encoding="utf-8-sig") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                sequence = row["Sequence"]
                target = json.loads(row["Label encoding"])
                if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                    raise ValueError(f"Unresolved backbone at {split}.csv row {index}")
                if len(target) != 49 or not all(v in (0, 1) for v in target):
                    raise ValueError(f"Expected published 49-bit labels at {split}.csv row {index}")
                if row["is_therapeutic"] not in ("True", "False"):
                    raise ValueError(f"Invalid therapeutic flag at {split}.csv row {index}")
                flag = row["is_therapeutic"] == "True"
                if bool(target[-1]) != flag or (not flag and any(target[:-1])):
                    raise ValueError(f"Inconsistent background/49-bit labels at {split}.csv row {index}")
                sequences.append(sequence)
                labels.append(target)
                splits.append(split)
                ids.append(f"therapep:{split}:{index}")
                sources.append(row["source"])
    n = len(sequences)
    sequences = np.asarray(sequences)
    split_array = np.asarray(splits)
    # Exact-sequence IDs describe duplicates; they do not impose a new split or
    # claim homology isolation. Counts/prototypes are still fitted on Train only.
    _, clusters = np.unique(sequences, return_inverse=True)
    return {"rows": np.arange(n), "ids": np.asarray(ids), "sequences": sequences,
            "raw_sequences": sequences, "available": np.ones(n, bool),
            "d_mask": np.zeros((n, 0), bool), "y": np.asarray(labels, np.uint8),
            "labels": list(LABELS), "splits": split_array, "clusters": clusters,
            "sources": np.asarray(sources), "augmented": np.zeros(n, bool), "in_scope": np.ones(n, bool),
            "masks": {**{f"{s}_original_mask": split_array == s for s in ("train", "val", "test")},
                      "train_aug_mask": split_array == "train"}, "dataset_id": DATASET}


def describe_data(data):
    return {"task": TRACK, "dataset": DATASET, "release": RELEASE, "labels": LABELS,
            "splits": {s: {"rows": int((data["splits"] == s).sum()),
                           "therapeutic": int(data["y"][data["splits"] == s, -1].sum()),
                           "sources": dict(Counter(data["sources"][data["splits"] == s]))}
                       for s in ("train", "val", "test")},
            "split_protocol": "author-published CSV membership; no resplitting or deduplication",
            "background_semantics": "synthetic assigned negatives, not experimentally verified inactivity"}


def init_multimodal(source, statistics, min_support, max_factors):
    from .multimodal import fit_numeric, label_statistics
    data = source.data
    n = len(data.rows)
    source.seq = {"canonical_tokens": np.zeros((n, 0), np.int64)}
    source.struct = {"structure_available": np.zeros(n, bool)}
    if statistics is None:
        train = data.indices("train")
        stats = label_statistics(data.y[train], data.clusters[train], data.labels, min_support, max_factors)
        stats.update(therapep_background_index=48, geometry_positive_clusters=[0]*49,
                     geometry_prototype_counts=[0]*49)
        statistics = {"phys": fit_numeric(np.empty((0, 515))),
                      "nodes": fit_numeric(np.empty((0, 2))),
                      "structure_global": fit_numeric(np.empty((0, 25))), "chemical_vocab": {},
                      "labels": stats, "function_projection": None, "dataset": data.dataset_id,
                      "cohort": "therapep_author_published_all49", "train_rows": len(train)}
        statistics["phys"]["zero_missing"] = True
    if statistics["labels"]["labels"] != data.labels:
        raise ValueError("Checkpoint label order differs from published TheraPep labels")
    source.statistics = statistics


def sequence_sample(data, i, esm):
    length = len(data.sequences[i])
    aa = torch.tensor(["ACDEFGHIKLMNPQRSTVWY".index(c)+1 for c in data.sequences[i]])
    quality = torch.zeros(3, 6)
    quality[0] = 1
    return {"esm": esm, "aa": aa, "mask": torch.ones(length, dtype=torch.bool),
            "d": torch.zeros(length, dtype=torch.long), "residue_chem": torch.zeros(length, dtype=torch.long),
            "chemical": torch.zeros(1, dtype=torch.long), "coords": torch.zeros(length, 3),
            "geometry": torch.zeros(length, 16), "struct_mask": torch.zeros(length, dtype=torch.bool),
            "dssp": torch.full((length,), 9, dtype=torch.long),
            "edge_type": torch.zeros(length, length, dtype=torch.long),
            "confidence_known": torch.zeros(length, dtype=torch.bool),
            "phys": torch.zeros(515), "phys_mask": torch.zeros(515, dtype=torch.bool),
            "struct_global": torch.zeros(25), "struct_global_mask": torch.zeros(25, dtype=torch.bool),
            "quality": quality, "y": torch.from_numpy(data.y[i].astype(np.float32)),
            "row_id": torch.tensor(int(data.rows[i]))}


def groups(data, indices):
    positive = data.y[indices, -1].astype(bool)
    lengths = np.array([len(data.sequences[i]) for i in indices])
    return {"therapeutic_positive": positive, "synthetic_background": ~positive,
            "length_2_8": lengths <= 8, "length_9_16": (lengths > 8) & (lengths <= 16),
            "length_17_32": (lengths > 16) & (lengths <= 32), "length_33_plus": lengths > 32,
            **{f"source_{name}": data.sources[indices] == name for name in sorted(set(data.sources))}}


def published_metrics(y, probabilities, threshold):
    """Disentangle 49-output scores, pure functional prediction and synthetic FPR."""
    from .metrics import evaluate_metrics
    from sklearn.metrics import average_precision_score, roc_auc_score
    positive = y[:, -1].astype(bool)
    background = ~positive
    threshold_array = np.asarray(threshold)
    function_threshold = threshold_array[:-1] if threshold_array.ndim else threshold
    flag_threshold = threshold_array[-1] if threshold_array.ndim else threshold
    flag_prediction = probabilities[:, -1] >= flag_threshold
    tp = int((positive & flag_prediction).sum())
    fp = int((background & flag_prediction).sum())
    fn = int((positive & ~flag_prediction).sum())
    flag = {"accuracy": float((positive == flag_prediction).mean()),
            "precision": tp/(tp+fp) if tp+fp else 0., "recall": tp/(tp+fn) if tp+fn else 0.,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "ap": float(average_precision_score(positive, probabilities[:, -1])) if positive.any() else None,
            "auroc": float(roc_auc_score(positive, probabilities[:, -1])) if positive.any() and background.any() else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": int((background & ~flag_prediction).sum())}
    return {
        "scope": "Published membership; 49-column overall scores are reported separately in metrics",
        "all49_labelwise_accuracy": float(((probabilities >= threshold) == y.astype(bool)).mean()),
        "positive_all49": evaluate_metrics(y[positive], probabilities[positive], threshold) if positive.any() else None,
        "all_functions48": evaluate_metrics(y[:, :-1], probabilities[:, :-1], function_threshold),
        "positive_functions48": evaluate_metrics(y[positive, :-1], probabilities[positive, :-1], function_threshold) if positive.any() else None,
        "therapeutic_flag": flag,
        "synthetic_background": {
            "rows": int(background.sum()),
            "any_function_fpr": float((probabilities[background, :-1] >= function_threshold).any(1).mean()) if background.any() else None,
            "therapeutic_flag_fpr": float((probabilities[background, -1] >= flag_threshold).mean()) if background.any() else None,
            "semantics": "Rejection of synthetic assigned negatives; not real biological inactivity FPR"},
    }
