"""Export both TheraSet logits on the Train fold held out by a saved model."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mfp_nova.hh_v13 import export_fold, fold_indices
from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from mfp_nova.research import load_model, validate_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    config = checkpoint["config"]
    train = config["train"]
    fold = train.get("hh_oof_fold")
    if (checkpoint["model_name"] != "theraset" or checkpoint["track"] != "fine39" or fold not in range(3)
            or train.get("xbridge_inner_fold") is not None or train.get("limit_samples") or train.get("max_steps")):
        raise ValueError("Expected a full original Train-fold TheraSet checkpoint")
    paths = config["data"]
    _, held = fold_indices(paths["root"], fold)
    source = MultimodalData(paths["root"], "fine39", checkpoint["statistics"])
    dataset = MultimodalPeptides(source, paths["features"], "train",
                                 peptide_feature_root=paths.get("peptide_features"), indices=held)
    validate_cache(dataset, config, checkpoint["encoder"])
    loader = DataLoader(dataset, batch_size=train["batch_size"], collate_fn=collate_multimodal)
    args.output_dir.mkdir(parents=True)
    export_fold(args.output_dir, model, loader, device, fold, checkpoint["best_epoch"], False, 0.)
    print(args.output_dir / "oof_logits.npz")


if __name__ == "__main__":
    main()
