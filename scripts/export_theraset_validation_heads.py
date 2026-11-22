"""Export pre-correction and final logits from the original Fine39 TheraSet."""

import argparse
from pathlib import Path

import numpy as np

from mfp_nova.research import load_model
from run_trievidence import batches, collect, open_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    parser.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, checkpoint = load_model(args.checkpoint, args.device)
    if (checkpoint["model_name"] != "theraset" or checkpoint["track"] != "fine39"
            or model.v5_readout is not None or not model.evidence_decoder):
        raise ValueError("Use the original Fine39 TheraSet baseline with its evidence decoder")
    source, val = open_data(args, "val", checkpoint)

    def read(batch):
        result = model(batch)
        return result["direct"], result["logits"]

    direct, final = collect(read, batches(val, args.batch_size), args.device, progress=True)
    rows = val.indices
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, row_id=source.data.rows[rows], ID=source.data.ids[rows],
                            cluster_id=source.data.clusters[rows], label_names=source.data.labels,
                            targets=source.data.y[rows], split="val", epoch=checkpoint["best_epoch"],
                            threshold=checkpoint["threshold"], direct_logits=direct.numpy(),
                            final_logits=final.numpy(), correction_logits=(final - direct).numpy())
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
