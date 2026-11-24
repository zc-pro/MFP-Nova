"""Pack and verify the original TheraSet and strip_only HelixSet as one model."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from compare_duoevidence_pairs import align
from mfp_nova.models.therahelix_fusion import TheraHelixFusion
from mfp_nova.reference_fallback import apply_train_reference, paired_cluster_interval
from run_trievidence import batches, collect, open_data


MODEL_NAME = "therahelix_fusion"


def pack(args):
    duo = torch.load(args.duo, map_location="cpu", weights_only=True)
    strip = torch.load(args.strip, map_location="cpu", weights_only=True)
    helix_args = strip["model_args"]["helixset_args"]
    if (duo["model_name"] != "duoevidence" or strip["model_name"] != "trievidence"
            or duo["track"] != "fine39" or duo["labels"] != strip["labels"]
            or not helix_args.get("compact_inactive") or helix_args.get("mix_latents", True)
            or helix_args.get("triangles", True)):
        raise ValueError("Use aligned Fine39 DuoEvidence and strip_only checkpoints")
    model_args = dict(theraset_args=duo["model_args"], helixset_args=helix_args)
    thera_args = model_args["theraset_args"]
    if (not thera_args["peptide_input_dim"]
            or any(thera_args[key] != helix_args[key] for key in ("input_dim", "peptide_input_dim"))
            or any(duo["statistics"].get(key) != strip["statistics"].get(key) for key in (
                "dataset", "phys", "nodes", "structure_global", "chemical_vocab",
                "localized_terminal_chemistry"))):
        raise ValueError("The readers must use the same dual-ESM and input preprocessing")
    model = TheraHelixFusion(**model_args)
    state = {key: value for key, value in duo["model_state"].items() if key.startswith("theraset.")}
    state.update({key: value for key, value in strip["model_state"].items() if key.startswith("helixset.")})
    # The matched pair comparison used the original TheraSet threshold and a fixed HelixSet threshold of .64.
    state["offsets"] = torch.tensor([float(duo["model_state"]["offsets"][0]), math.log(.64 / .36)])
    model.load_state_dict(state)
    checkpoint = {key: duo[key] for key in ("config", "encoder", "statistics", "labels", "track")}
    checkpoint.update(model_name=MODEL_NAME, model_args=model_args, model_state=model.state_dict())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(args.output)


def decisions(data, indices, score, theraset):
    raw = score >= 0
    empty = ~raw.any(1)
    initial = raw.copy()
    initial[empty, score[empty].argmax(1)] = True
    _, final = apply_train_reference(data, indices, raw, initial, theraset)
    return raw, final


def jaccard(y, predicted):
    return (y & predicted).sum(1) / (y | predicted).sum(1).clip(min=1)


def compare_reference(paths, data, indices, split, theraset, helixset, score, raw, final):
    with np.load(paths[0]) as duo, np.load(paths[1]) as strip:
        idx = align(duo, strip, split)
        for key, expected in (("row_id", data.rows[indices]), ("cluster_id", data.clusters[indices]),
                              ("targets", data.y[indices]), ("label_names", data.labels)):
            np.testing.assert_array_equal(duo[key], expected)
        a = duo["theraset_logits"]
        h = strip["latent_logits"][idx]
        limit = np.log(999999.)
        ref_score = ((a.clip(-limit, limit) - duo["offsets"][0])
                     + (h.clip(-limit, limit) - np.log(.64 / .36))) / 2
    ref_raw, ref_final = decisions(data, indices, ref_score, a)
    y = data.y[indices].astype(bool)
    baseline = jaccard(y, ref_final)
    delta = jaccard(y, final) - baseline
    return {
        "baseline_acc": float(baseline.mean()),
        "delta_pp": float(delta.mean() * 100),
        "ci95_pp": paired_cluster_interval(delta, data.clusters[indices]),
        "changed_raw_rows": int((raw != ref_raw).any(1).sum()),
        "changed_final_rows": int((final != ref_final).any(1).sum()),
        "max_abs_delta": {
            name: float(np.abs(observed - expected).max())
            for name, observed, expected in (("theraset_logits", theraset, a),
                                              ("latent_logits", helixset, h),
                                              ("logits", score, ref_score))
        },
    }


def evaluate(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != MODEL_NAME:
        raise ValueError(f"Expected a {MODEL_NAME} checkpoint")
    model = TheraHelixFusion(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device).eval()
    source, dataset = open_data(args, args.split, checkpoint)
    data, indices = source.data, dataset.indices

    def read(batch):
        output = model(batch, reuse_inputs=not args.separate_inputs)
        return output["theraset_logits"], output["latent_logits"], output["logits"]

    theraset, helixset, score = (column.numpy() for column in collect(
        read, batches(dataset, args.batch_size), args.device, progress=True))
    raw, final = decisions(data, indices, score, theraset)
    y = data.y[indices].astype(bool)
    report = {
        "split": args.split, "rows": len(y), "raw_acc": float(jaccard(y, raw).mean()),
        "acc": float(jaccard(y, final).mean()), "zero_hits": int((~(final & y).any(1)).sum()),
        "reuse_inputs": not args.separate_inputs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    if args.reference:
        report["reference"] = compare_reference(
            args.reference, data, indices, args.split, theraset, helixset, score, raw, final)
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / f"{args.split}_predictions.npz", split=args.split, row_id=data.rows[indices],
        cluster_id=data.clusters[indices], label_names=np.asarray(data.labels), targets=y,
        theraset_logits=theraset, latent_logits=helixset, logits=score, predicted_raw=raw, predicted=final)
    result = json.dumps(report, indent=2)
    (args.output / "results.json").write_text(result + "\n", encoding="utf-8")
    print(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    combine = commands.add_parser("pack")
    combine.add_argument("duo", type=Path)
    combine.add_argument("strip", type=Path)
    combine.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("evaluate")
    compare.add_argument("checkpoint", type=Path)
    compare.add_argument("--split", choices=("val", "test"), default="val")
    compare.add_argument("--device", default="cuda:0")
    compare.add_argument("--batch-size", type=int, default=16)
    compare.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    compare.add_argument("--features", type=Path, default=Path("features/esm2_650m"))
    compare.add_argument("--peptide-features", type=Path, default=Path("features/esm2_peptide_150m"))
    compare.add_argument("--reference", type=Path, nargs=2, metavar=("DUO_NPZ", "STRIP_NPZ"))
    compare.add_argument("--separate-inputs", action="store_true")
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (pack if args.command == "pack" else evaluate)(args)


if __name__ == "__main__":
    main()
