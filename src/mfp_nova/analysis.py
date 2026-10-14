"""Necessary dataset statistics, not a reimplementation of upstream audits."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def analyze_data(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    with np.load(root / "arrays/sequences.npz", allow_pickle=False) as source:
        rows, ids = source["row_id"], source["ID"]
        sequence, raw = source["canonical_backbone"], source["raw_sequence"]
        available, d_mask = source["canonical_available"], source["D_annotation_mask"]
    with np.load(root / "tracks/major15/dataset.npz", allow_pickle=False) as source:
        original = ~source["is_augmented"]
        split, cluster = source["split"], source["cluster_id"]
    cluster_splits = defaultdict(set)
    canonical_splits = defaultdict(set)
    for i in np.flatnonzero(original):
        cluster_splits[int(cluster[i])].add(str(split[i]))
        if available[i]:
            canonical_splits[str(sequence[i])].add(str(split[i]))
    lengths = np.array([len(s) for s in sequence[original & available]])
    canonical_counts = Counter(sequence[original & available].tolist())
    raw_counts = Counter(raw[original].tolist())
    annotations = Counter()
    with (root / "records.csv").open(encoding="utf-8-sig", newline="") as f:
        for record in csv.DictReader(f):
            if record["is_augmented"] == "False":
                for key in ("primary_structure_available", "chemical_annotation_conflict", "terminal_annotation_conflict"):
                    annotations[key] += record[key] == "True"
                annotations["ptm_field_nonempty"] += bool(record["Post translation modifications"].strip())
                annotations["helm_field_nonempty"] += bool(record["HELM notation"].strip())
    result = {
        "dataset": manifest["dataset"], "version_date": manifest["version_date"],
        "total_rows": len(rows), "original_rows": int(original.sum()),
        "candidate_augmentation_rows": int((~original).sum()),
        "row_id_unique": len(np.unique(rows)) == len(rows), "id_unique": len(np.unique(ids)) == len(ids),
        "original_split_counts": {s: int((original & (split == s)).sum()) for s in ("train", "val", "test")},
        "original_canonical_available": int((original & available).sum()),
        "original_canonical_unresolved": int((original & ~available).sum()),
        "original_d_annotated_rows": int((original & d_mask.any(1)).sum()),
        "raw_non_20aa_rows": sum(bool(set(s.upper()) - set("ACDEFGHIKLMNPQRSTVWY")) for s in raw[original]),
        "canonical_length": {"min": int(lengths.min()), "max": int(lengths.max()),
                             "quantiles_25_50_75_95_99": np.quantile(lengths, [.25, .5, .75, .95, .99]).tolist()},
        "raw_exact_duplicate_groups": sum(c > 1 for c in raw_counts.values()),
        "canonical_exact_duplicate_groups": sum(c > 1 for c in canonical_counts.values()),
        "cross_split_clusters": sum(len(s) > 1 for s in cluster_splits.values()),
        "cross_split_exact_canonical_sequences": sum(len(s) > 1 for s in canonical_splits.values()),
        "similarity_protocol": "Retain supplied repaired connected components; 90% global edit relation plus verified MMseqs candidates. No exhaustive similarity recomputation in this scaffold.",
        "annotation_counts_original": dict(annotations),
        "auxiliary_inputs": {"physchem_features": manifest["physchem_features"],
                             "tabular_features": manifest["tabular_features_total"],
                             "primary_backbone_structures": manifest["primary_backbone_structures"],
                             "pdb_archive": (root / "raw/TertiaryStructure.tar.gz").exists(),
                             "3di": "No delivered 3Di array identified"},
        "label_semantics": "Multilabel binary observed annotations; 0 is unannotated, not confirmed negative. __unknown means no specific subtype annotation.",
        "tracks": {},
    }
    for track in ("major15", "fine39"):
        with np.load(root / f"tracks/{track}/dataset.npz", allow_pickle=False) as source:
            if not np.array_equal(rows, source["row_id"]):
                raise ValueError(f"Sequence/target alignment differs in {track}")
            y, labels = source["Y"], source["label_names"].tolist()
            if not np.isin(y, [0, 1]).all():
                raise ValueError(f"Nonbinary targets in {track}")
            counts, support, sequence_counts = {}, {}, {}
            for s in ("train", "val", "test"):
                mask = source[f"{s}_original_mask"]
                counts[s] = int(mask.sum())
                sequence_counts[s] = int((mask & available).sum())
                support[s] = y[mask].sum(0).tolist()
            supervised = original & source["row_in_scope"]
            cardinality = y[supervised].sum(1)
            train_y = y[source["train_original_mask"]].astype(np.int64)
            seq_support = {s: y[source[f"{s}_original_mask"] & available].sum(0).tolist()
                           for s in ("train", "val", "test")}
            result["tracks"][track] = {
                "num_labels": len(labels), "labels": labels, "supervised_original_rows": int(supervised.sum()),
                "excluded_from_supervision": int((original & ~source["row_in_scope"]).sum()),
                "split_counts": counts, "sequence_split_counts": sequence_counts,
                "positive_support": support, "sequence_positive_support": seq_support,
                "train_support_min_max": [min(support["train"]), max(support["train"])],
                "label_cardinality_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(cardinality, return_counts=True))},
                "mean_cardinality": float(cardinality.mean()),
                "multilabel_fraction": float((cardinality >= 2).mean()),
                "selected_augmentation_rows": int(source["augmentation_selected_mask"].sum()),
                "train_label_cooccurrence": (train_y.T @ train_y).tolist(),
            }
    return result
