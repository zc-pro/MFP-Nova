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
TRACKS = (TRACK, "therapep49_positive", "therapep48")
ROOTS = {TRACK: "data/therapep_ai", "therapep49_positive": "data/therapep49_positive",
         "therapep48": "data/therapep48"}
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
    if task not in TRACKS:
        return
    data = config["data"]
    if data["root"] == "data/therapeutic_2025":
        data["root"] = ROOTS[task]
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


def prepare_positive_variants(source="data/therapep_ai", output_root="data"):
    """Create positive-only CSVs with original IDs, retaining each published split."""
    source, output_root = Path(source), Path(output_root)
    results = {}
    for track in TRACKS[1:]:
        root = output_root / track
        if root.exists():
            results[track] = describe_data(read_data(root, track))
            continue
        root.mkdir(parents=True)
        offset = 0
        for split in ("train", "val", "test"):
            with (source/f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                fields = [f for f in reader.fieldnames if track != "therapep48" or f != "is_therapeutic"]
                fields += [f for f in ("row_id", "ID") if f not in fields]
                count = 0
                with (root/f"{split}.csv").open("x", encoding="utf-8", newline="") as out:
                    writer = csv.DictWriter(out, fieldnames=fields)
                    writer.writeheader()
                    for index, row in enumerate(reader):
                        count += 1
                        target = json.loads(row["Label encoding"])
                        if len(target) != 49 or any(v not in (0, 1) for v in target):
                            raise ValueError(f"Expected binary 49-bit source labels: {split}/{index}")
                        flag = row["is_therapeutic"]
                        if flag not in ("True", "False") or bool(target[-1]) != (flag == "True"):
                            raise ValueError(f"Inconsistent source flag: {split}/{index}")
                        synthetic = row["source"].startswith("synthetic_")
                        if synthetic != (flag == "False") or (synthetic and any(target)):
                            raise ValueError(f"Inconsistent synthetic source: {split}/{index}")
                        if synthetic:
                            continue
                        row["row_id"] = offset + index
                        row["ID"] = f"therapep:{split}:{index}"
                        if track == "therapep48":
                            row.pop("is_therapeutic")
                            row["Label encoding"] = json.dumps(target[:-1])
                        writer.writerow(row)
            offset += count
        names = LABELS[:-1] if track == "therapep48" else LABELS
        (root/"labels.json").write_text(json.dumps(names, indent=2)+"\n", encoding="utf-8")
        (root/"README.md").write_text(
            f"# {track}\n\nPositive-only subset of {source}; original split membership, order and duplicates retained.\n"
            "row_id and ID retain the original complete CSV identities, allowing aligned full-cohort ESM cache reuse.\n"
            + ("48 functional outputs; is_therapeutic column and final target bit removed.\n" if track == "therapep48" else
               "49 outputs; is_therapeutic is constant 1. Report the functional48 metrics separately.\n"),
            encoding="utf-8")
        results[track] = describe_data(read_data(root, track))
    return results


def read_data(root, track=TRACK):
    """Preserve CSV membership, sequence, labels and duplicate rows verbatim."""
    names = LABELS[:-1] if track == "therapep48" else LABELS
    label_file = Path(root)/"labels.json"
    if label_file.exists() and json.loads(label_file.read_text(encoding="utf-8")) != names:
        raise ValueError("CSV label order differs from the selected TheraPep task")
    sequences, labels, splits, ids, sources, rows, therapeutic = [], [], [], [], [], [], []
    for split in ("train", "val", "test"):
        with (Path(root) / f"{split}.csv").open(newline="", encoding="utf-8-sig") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                sequence = row["Sequence"]
                target = json.loads(row["Label encoding"])
                if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                    raise ValueError(f"Unresolved backbone at {split}.csv row {index}")
                if len(target) != len(names) or not all(v in (0, 1) for v in target):
                    raise ValueError(f"Expected published {len(names)}-bit labels at {split}.csv row {index}")
                if track == "therapep48":
                    flag = True
                    if "is_therapeutic" in row:
                        raise ValueError("therapep48 must omit the is_therapeutic column")
                else:
                    if row["is_therapeutic"] not in ("True", "False"):
                        raise ValueError(f"Invalid therapeutic flag at {split}.csv row {index}")
                    flag = row["is_therapeutic"] == "True"
                    if bool(target[-1]) != flag or (not flag and any(target[:-1])):
                        raise ValueError(f"Inconsistent background/49-bit labels at {split}.csv row {index}")
                if track != TRACK and (not flag or row["source"].startswith("synthetic_") or not any(target[:48])):
                    raise ValueError(f"Positive-only task includes a background/empty row: {split}/{index}")
                sequences.append(sequence)
                labels.append(target)
                splits.append(split)
                rows.append(int(row["row_id"]) if track != TRACK else int(row.get("row_id", len(rows))))
                ids.append(row["ID"] if track != TRACK else row.get("ID", f"therapep:{split}:{index}"))
                sources.append(row["source"])
                therapeutic.append(flag)
    n = len(sequences)
    sequences = np.asarray(sequences)
    split_array = np.asarray(splits)
    # Exact-sequence IDs describe duplicates; they do not impose a new split or
    # claim homology isolation. Counts/prototypes are still fitted on Train only.
    _, clusters = np.unique(sequences, return_inverse=True)
    return {"rows": np.asarray(rows, np.int64), "ids": np.asarray(ids), "sequences": sequences,
            "raw_sequences": sequences, "available": np.ones(n, bool),
            "d_mask": np.zeros((n, 0), bool), "y": np.asarray(labels, np.uint8),
            "labels": list(names), "splits": split_array, "clusters": clusters,
            "therapeutic": np.asarray(therapeutic, bool), "therapep_task": track,
            "sources": np.asarray(sources), "augmented": np.zeros(n, bool), "in_scope": np.ones(n, bool),
            "masks": {**{f"{s}_original_mask": split_array == s for s in ("train", "val", "test")},
                      "train_aug_mask": split_array == "train"}, "dataset_id": DATASET}


def describe_data(data):
    return {"task": data["therapep_task"], "dataset": DATASET, "release": RELEASE, "labels": data["labels"],
            "splits": {s: {"rows": int((data["splits"] == s).sum()),
                           "therapeutic": int(data["therapeutic"][data["splits"] == s].sum()),
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
        stats.update(geometry_positive_clusters=[0]*len(data.labels),
                     geometry_prototype_counts=[0]*len(data.labels))
        if data.track != "therapep48":
            stats["therapep_background_index"] = 48
        statistics = {"phys": fit_numeric(np.empty((0, 515))),
                      "nodes": fit_numeric(np.empty((0, 2))),
                      "structure_global": fit_numeric(np.empty((0, 25))), "chemical_vocab": {},
                      "labels": stats, "function_projection": None, "dataset": data.dataset_id,
                      "cohort": ("therapep_author_published_all49" if data.track == TRACK else data.track),
                      "train_rows": len(train)}
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
    positive = data.therapeutic[indices]
    lengths = np.array([len(data.sequences[i]) for i in indices])
    return {"therapeutic_positive": positive, "synthetic_background": ~positive,
            "length_2_8": lengths <= 8, "length_9_16": (lengths > 8) & (lengths <= 16),
            "length_17_32": (lengths > 16) & (lengths <= 32), "length_33_plus": lengths > 32,
            **{f"source_{name}": data.sources[indices] == name for name in sorted(set(data.sources))}}


def published_metrics(y, probabilities, threshold):
    """Disentangle 49-output scores, pure functional prediction and synthetic FPR."""
    from .metrics import evaluate_metrics
    from sklearn.metrics import average_precision_score, roc_auc_score
    if y.shape[1] == 48:
        functions = evaluate_metrics(y, probabilities, threshold)
        return {"scope": "Positive-only 48 functional outputs; no therapeutic flag or synthetic background",
                "all48_labelwise_accuracy": float(((probabilities >= threshold) == y.astype(bool)).mean()),
                "all_functions48": functions, "positive_functions48": functions,
                "therapeutic_flag": None,
                "synthetic_background": {"rows": 0, "any_function_fpr": None,
                                         "therapeutic_flag_fpr": None, "semantics": "No synthetic background"}}
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
