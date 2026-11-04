"""Export TheraPep inputs and unique sequence queries for TPpred-LE / SA-MTP."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from mfp_nova.therapep import ROOTS, TRACKS, read_data


def prepare(task, root, output):
    data = read_data(root, task)
    output.mkdir(parents=True, exist_ok=False)
    # Only feature queries are deduplicated. Every training/evaluation row remains.
    sequence_ids = {}
    for sequence in data["sequences"]:
        if sequence not in sequence_ids:
            sequence_ids[sequence] = f"pep{len(sequence_ids):07d}"
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.task, (args.data_root or PROJECT / ROOTS[args.task]).resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
