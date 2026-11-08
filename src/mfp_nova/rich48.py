"""Build the author-positive 48-label cohort with source-aligned multimodal inputs.

No fitted preprocessing or model weights are copied. Run with
python -m mfp_nova.rich48; only NumPy is needed to build the raw dataset.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .therapep import DATASET, LABELS, read_data


def build_rich48(author_root="data/therapep_ai", source_root="data/therapeutic_2025",
                 output="data/therapeutic_2025/rich48"):
    author_root, source_root, output = map(Path, (author_root, source_root, output))
    author = read_data(author_root)
    take = np.flatnonzero(author["therapeutic"])
    rows, ids = author["rows"][take], author["ids"][take]
    sequences, splits = author["sequences"][take], author["splits"][take]
    sources, y = author["sources"][take], author["y"][take, :48]
    with (source_root/"records.csv").open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    with np.load(source_root/"arrays/sequences.npz", allow_pickle=False) as f:
        original = {key: f[key] for key in f.files}
    if [int(r["row_id"]) for r in records] != original["row_id"].tolist():
        raise ValueError("Source records and sequence row_id order differ")
    lookup = defaultdict(list)
    for i, record in enumerate(records):
        if record["is_augmented"] == "False" and original["canonical_available"][i]:
            key = (str(original["canonical_backbone"][i]), record["Is_natural_peptide"] == "True")
            lookup[key].append(i)
    donors, candidate_counts = [], []
    for sequence, source in zip(sequences, sources):
        if source not in ("TheraPepDB", "TheraPepDB_recovered"):
            raise ValueError(f"Unexpected positive source: {source}")
        candidates = lookup[(str(sequence), source == "TheraPepDB")]
        donors.append(candidates[0] if len(candidates) == 1 else -1)
        candidate_counts.append(len(candidates))
    donors = np.asarray(donors, np.int64)
    matched = donors >= 0
    n = len(rows)
    width = max(max(map(len, sequences)), original["canonical_tokens"].shape[1])
    lengths = np.array(list(map(len, sequences)))
    residue_mask = np.arange(width)[None] < lengths[:, None]
    aa = np.zeros((n, width), np.uint8)
    for i, sequence in enumerate(sequences):
        aa[i, :len(sequence)] = ["ACDEFGHIKLMNPQRSTVWY".index(c)+1 for c in sequence]
    d = np.zeros((n, width), bool)
    source_width = original["D_annotation_mask"].shape[1]
    d[matched, :source_width] = original["D_annotation_mask"][donors[matched]]
    d &= residue_mask
    _, clusters = np.unique(sequences, return_inverse=True)

    # A fresh output prevents overwriting another dataset or a running experiment.
    output.mkdir(parents=True, exist_ok=False)
    (output/"arrays").mkdir()
    (output/"tracks/rich48").mkdir(parents=True)
    np.savez_compressed(output/"arrays/sequences.npz", row_id=rows, ID=ids,
                        raw_sequence=sequences, canonical_backbone=sequences,
                        canonical_available=np.ones(n, bool), canonical_tokens=aa,
                        D_annotation_mask=d, residue_mask=residue_mask,
                        alphabet=original["alphabet"])
    masks = {f"{s}_original_mask": splits == s for s in ("train", "val", "test")}
    np.savez_compressed(output/"tracks/rich48/dataset.npz", row_id=rows, Y=y,
                        label_names=np.asarray(LABELS[:48]), split=splits, cluster_id=clusters,
                        is_augmented=np.zeros(n, bool), row_in_scope=np.ones(n, bool),
                        train_aug_mask=splits == "train", **masks)
    names = json.dumps(LABELS[:48], indent=2)+"\n"
    (output/"tracks/rich48/labels.json").write_text(names, encoding="utf-8")
    (output/"labels.json").write_text(names, encoding="utf-8")

    with np.load(source_root/"arrays/physchem_raw.npz", allow_pickle=False) as f:
        phys = np.full((n, f["X"].shape[1]), np.nan, np.float32)
        available = np.zeros_like(phys, bool)
        phys[matched] = f["X"][donors[matched]]
        available[matched] = f["value_available"][donors[matched]]
        phys[~available] = np.nan
        np.savez_compressed(output/"arrays/physchem_raw.npz", row_id=rows,
                            X=phys, value_available=available, columns=f["columns"])
        phys_present = available.any(1)
    del original, phys, available

    with np.load(source_root/"arrays/structure_primary.npz", allow_pickle=False) as f:
        structure = {"row_id": rows}
        edge_keys = {"edge_offsets", "edge_i", "edge_j", "edge_type", "edge_distance_A"}
        metadata_keys = {"dssp_code_names", "scalar_columns"}
        for key in f.files:
            if key == "row_id" or key in edge_keys:
                continue
            raw = f[key]
            if key in metadata_keys:
                structure[key] = raw
                continue
            fill = np.nan if np.issubdtype(raw.dtype, np.floating) else (-1 if key in ("dssp", "raw_structure_index") else 0)
            value = np.full((n, *raw.shape[1:]), fill, dtype=raw.dtype)
            value[matched] = raw[donors[matched]]
            structure[key] = value
        source_edges = {key: f[key] for key in edge_keys}
        offsets = source_edges["edge_offsets"]
        sizes = np.zeros(n, np.int64)
        sizes[matched] = np.diff(offsets)[donors[matched]]
        structure["edge_offsets"] = np.r_[0, np.cumsum(sizes)]
        for key in edge_keys - {"edge_offsets"}:
            raw = source_edges[key]
            pieces = [raw[offsets[j]:offsets[j+1]] for j in donors if j >= 0]
            structure[key] = np.concatenate(pieces) if pieces else np.empty(0, raw.dtype)
        structure["scalar_X"][~structure["scalar_value_available"]] = np.nan
        if np.any(structure["residue_mask"].sum(1) > lengths):
            raise ValueError("Source structure exceeds matched sequence length")
        np.savez_compressed(output/"arrays/structure_primary.npz", **structure)
        structure_present = structure["structure_available"]
    del structure, source_edges

    with (source_root/"symbolic_sequences.jsonl").open(encoding="utf-8") as f:
        symbols = {int(item["row_id"]): item for item in map(json.loads, f)}
    chemical_fields = ("HELM notation", "N-terminal modification", "C-terminal modification",
                       "Post translation modifications")
    fields = ["row_id", "ID", "Sequence", "source", "source_row_id", "source_ID",
              "source_raw_sequence", "source_cluster_id", "candidate_count", "match_status",
              "source_evidence", *chemical_fields]
    with (output/"records.csv").open("w", encoding="utf-8", newline="") as f, \
            (output/"symbolic_sequences.jsonl").open("w", encoding="utf-8") as sf:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, j in enumerate(donors):
            record = records[j] if j >= 0 else {}
            status = "unique_sequence_and_source" if j >= 0 else ("ambiguous" if candidate_counts[i] else "unmatched")
            writer.writerow({"row_id": int(rows[i]), "ID": ids[i], "Sequence": sequences[i],
                             "source": sources[i], "source_row_id": record.get("row_id", ""),
                             "source_ID": record.get("ID", ""), "source_raw_sequence": record.get("Sequence", ""),
                             "source_cluster_id": record.get("cluster_id", ""),
                             "candidate_count": candidate_counts[i], "match_status": status,
                             "source_evidence": record.get("source_evidence", "unavailable"),
                             **{k: record.get(k, "") for k in chemical_fields}})
            item = dict(symbols[int(record["row_id"])]) if j >= 0 else {"HELM_tokens": [], "parsed": False}
            item["row_id"] = int(rows[i])
            sf.write(json.dumps(item, ensure_ascii=False)+"\n")

    # Convenient sequence/target exports for a 48-output CNN baseline.
    for split in ("train", "val", "test"):
        with (output/f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["row_id", "ID", "Sequence", "Label encoding", "source", "Length"])
            writer.writeheader()
            for i in np.flatnonzero(splits == split):
                writer.writerow({"row_id": int(rows[i]), "ID": ids[i], "Sequence": sequences[i],
                                 "Label encoding": json.dumps(y[i].tolist()), "source": sources[i],
                                 "Length": int(lengths[i])})
    counts = {}
    for split in ("train", "val", "test"):
        mask = splits == split
        counts[split] = {"rows": int(mask.sum()), "matched": int((mask & matched).sum()),
                         "unmatched": int((mask & (np.asarray(candidate_counts) == 0)).sum()),
                         "ambiguous": int((mask & (np.asarray(candidate_counts) > 1)).sum()),
                         "structure_available": int((mask & structure_present).sum()),
                         "physchem_available": int((mask & phys_present).sum())}
    manifest = {"dataset": "rich48", "version_date": DATASET, "feature_dataset": DATASET,
                "labels": 48, "original_rows": n, "splits": counts,
                "author_root": str(author_root), "source_root": str(source_root),
                "matching": "Exact canonical sequence and author natural/recovered source category; unique original donors only; no labels used",
                "split_protocol": "Author positive members and duplicates retained; exact-sequence cluster IDs, no homology isolation",
                "chemistry": "Transferred source annotations are backbone/source matches, not experimentally verified chemical identity",
                "preprocessing": "Raw values only; fit imputation, scaling and vocabulary on rich48 Train",
                "feature_identity": "Author row_id, ID and sequence preserved for full-author ESM cache subset reuse"}
    (output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output/"README.md").write_text(
        "# rich48\n\n48 author functional labels, positive-only, original author Train/Val/Test.\n"
        "Sequence, structure and raw physicochemical views; unavailable/ambiguous views remain missing.\n"
        "No synthetic negatives, augmentation, relabeling, resplitting or fitted preprocessing.\n"
        "records.csv stores source donor identities and matching status. Matching does not certify chemical identity.\n"
        "Structure provenance and quality remain from the original package; structures are not all experimental.\n"
        "Train/Val/Test CSVs support a sequence-only baseline on identical members.\n\n"
        + json.dumps(counts, indent=2)+"\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", default="data/therapep_ai")
    parser.add_argument("--source-root", default="data/therapeutic_2025")
    parser.add_argument("--output", default="data/therapeutic_2025/rich48")
    args = parser.parse_args()
    print(json.dumps(build_rich48(args.author_root, args.source_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
