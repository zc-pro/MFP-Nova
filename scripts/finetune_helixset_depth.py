"""Recover a two-layer HelixSet decoder from the original Fine39 best.pt."""

import argparse
from datetime import datetime
from pathlib import Path

import torch

from mfp_nova.research import train


def recovery_config(config, checkpoint, device, epochs, learning_rate):
    settings = config["train"]
    settings.update(
        initialize_from=str(checkpoint), device=device, epochs=epochs,
        learning_rate=learning_rate, warmup_epochs=1, lr_decay_epochs=epochs,
        aux_warmup_epochs=1, rdrop_start_epoch=1, rdrop_ramp_epochs=1,
        patience=0, metric_suite="jaccard", fixed_threshold=.64,
        save_selection_candidates=False, export_validation_predictions=True,
    )
    if settings.get("ema_decay") is not None:
        settings["ema_start_epoch"] = 1
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--peptide-features", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.epochs < 2 or not 0 < args.learning_rate < 1:
        parser.error("Use at least two epochs and a positive learning rate below 1")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["model_name"] != "helixset" or checkpoint["track"] != "fine39":
        parser.error("Use the original Fine39 HelixSet best.pt")
    config = recovery_config(checkpoint["config"], args.checkpoint, args.device, args.epochs, args.learning_rate)
    del checkpoint
    for key, path in (("root", args.data_root), ("features", args.features),
                      ("peptide_features", args.peptide_features)):
        if path is not None:
            config["data"][key] = str(path)
    run = args.output_dir or Path("runs") / datetime.now().strftime("helixset_decoder2_recovery_%Y%m%d_%H%M%S_%f")
    config["train"]["output"] = str(run.parent)
    result = train(config, run.name)
    print(f"Selected epoch {result['best_epoch']} | "
          f"ACC@0.64={result['validation']['sample_jaccard']:.4%} | "
          f"Macro-AP={result['validation']['macro_ap']:.4%}", flush=True)
    print(f"Results: {run / 'results.json'}\nPredictions: {run / 'best_validation.npz'}", flush=True)


if __name__ == "__main__":
    main()
