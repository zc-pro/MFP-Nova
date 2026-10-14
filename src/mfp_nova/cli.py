import argparse
import json
from pathlib import Path

import yaml


def main(argv=None):
    parser = argparse.ArgumentParser(description="MFP-Nova research commands")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--data-root", default="data/therapeutic_2025")
    analyze.add_argument("--output", default="docs/data_summary.json")
    for command in ("features", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", default="configs/sequence.yaml")
        sub.add_argument("--data-root")
        sub.add_argument("--feature-root")
        sub.add_argument("--track", choices=("major15", "fine39"))
        sub.add_argument("--device")
        sub.add_argument("--batch-size", type=int, help="Encoder batch for features; head batch for train")
        if command == "features":
            sub.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
        else:
            sub.add_argument("--name", required=True)
            sub.add_argument("--seed", type=int)
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--loss", choices=("asl", "bce"))
            sub.add_argument("--global-only", action="store_true", help="Disable the local CNN branch")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-root", default="data/therapeutic_2025")
    evaluate.add_argument("--feature-root", default="features/esm2_35m")
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--batch-size", type=int, default=128)
    evaluate.add_argument("--predictions", help="Optional aligned sample probabilities in NPZ format")
    predict = commands.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--sequences", required=True, help="One resolved uppercase 20-AA backbone per line")
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "analyze":
        from .analysis import analyze_data
        result = analyze_data(args.data_root)
        from .trainer import write_json
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({"original_rows": result["original_rows"], "tracks": result["tracks"]}, ensure_ascii=False, indent=2))
    elif args.command in ("features", "train"):
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        for arg, key in ((args.data_root, "root"), (args.feature_root, "features"), (args.track, "track")):
            if arg is not None:
                config["data"][key] = arg
        if args.device is not None:
            config["train"]["device"] = args.device
        if args.batch_size is not None:
            section = "encoder" if args.command == "features" else "train"
            config[section]["batch_size"] = args.batch_size
        if args.command == "features":
            from .encoder import prepare_features
            prepare_features(config, args.splits, config["train"]["device"])
        else:
            from .trainer import train
            for key in ("seed", "epochs"):
                value = getattr(args, key)
                if value is not None:
                    config["train"][key] = value
            if args.loss is not None:
                config["loss"]["name"] = args.loss
            if args.global_only:
                config["model"]["use_local"] = False
            train(config, args.name)
    elif args.command == "evaluate":
        from .inference import evaluate
        evaluate(args.checkpoint, args.data_root, args.feature_root, args.split, args.device, args.output,
                 predictions=args.predictions, batch_size=args.batch_size)
    else:
        from .inference import predict_sequences
        sequences = [line.strip() for line in Path(args.sequences).read_text(encoding="utf-8").splitlines() if line.strip()]
        result = predict_sequences(args.checkpoint, sequences, args.device)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
