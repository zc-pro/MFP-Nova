"""Export split inputs and unique sequence queries for TPpred-LE / SA-MTP."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from mfp_nova.paper_benchmark import ROOTS, TRACKS, read_data


def prepare(task, root, output, query_source=None):
    data = read_data(root, task)
    # Only feature queries are deduplicated. Every training/evaluation row remains.
    if query_source is None:
        sequence_ids = {seq: f"pep{i:07d}" for i, seq in enumerate(dict.fromkeys(data["sequences"]))}
    else:
        lines = (query_source / "queries.fasta").read_text(encoding="utf-8").splitlines()
        sequence_ids = {seq: name[1:] for name, seq in zip(lines[::2], lines[1::2])}
        if not set(data["sequences"]).issubset(sequence_ids):
            raise ValueError("Query source does not cover this task's sequences")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "queries.fasta").open("w", encoding="utf-8") as stream:
        for sequence, query_id in sequence_ids.items():
            stream.write(f">{query_id}\n{sequence}\n")
    info = {"task": task, "data_root": str(root), "labels": data["labels"],
            "unique_sequences": len(sequence_ids), "query_length": "full sequence, no truncation",
            "feature_status": "queries only; PSSM and SS2 have not been generated", "splits": {}}
    for split in ("train", "val", "test"):
        indices = np.flatnonzero(data["splits"] == split)
        folder = output / split
        folder.mkdir()
        with (folder / "seqs.fasta").open("w", encoding="utf-8") as fasta, \
                (folder / "labels.csv").open("w", newline="", encoding="utf-8") as labels, \
                (folder / "rows.csv").open("w", newline="", encoding="utf-8") as rows:
            label_writer, row_writer = csv.writer(labels), csv.writer(rows)
            label_writer.writerow(data["labels"])
            row_writer.writerow(["row_id", "ID", "feature_id"])
            for i in indices:
                sequence = data["sequences"][i]
                feature_id = sequence_ids[sequence]
                fasta.write(f">{feature_id}\n{sequence}\n")
                label_writer.writerow(data["y"][i].tolist())
                row_writer.writerow([int(data["rows"][i]), data["ids"][i], feature_id])
        info["splits"][split] = {"rows": len(indices),
                                 "over_50": sum(len(data["sequences"][i]) > 50 for i in indices)}
    (output / "task.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TRACKS, default="therapep48")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--query-source", type=Path, help="Reuse sequence IDs from an existing structure export")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.task, (args.data_root or PROJECT / ROOTS[args.task]).resolve(), args.output.resolve(), args.query_source)


if __name__ == "__main__":
    main()
