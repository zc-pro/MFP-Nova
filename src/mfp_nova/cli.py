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
        sub.add_argument("--config")
        sub.add_argument("--data-root")
        sub.add_argument("--feature-root")
        sub.add_argument("--track", choices=("major15", "fine39"))
        sub.add_argument("--device")
        sub.add_argument("--batch-size", type=int, help="Encoder batch for features; head batch for train")
        sub.add_argument("--esm", choices=("35m", "650m"), help="Select frozen ESM size and default feature directory")
        if command == "features":
            sub.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
            sub.add_argument("--resume-from", type=int, default=0,
                             help="Completed rows in the first split, from the interrupted extraction log")
        else:
            sub.add_argument("--model", choices=("sequence", "trirelay", "geofactor", "protoroute"), default="sequence")
            sub.add_argument("--task", choices=("major15", "fine39"), help="Alias for --track")
            sub.add_argument("--resume", help="last.pt with optimizer and RNG state; creates a new run")
            sub.add_argument("--ablate", nargs="*", help="Model switches to disable, e.g. feedback high_order")
            sub.add_argument("--threshold-strategy", choices=("global", "per_label"))
            sub.add_argument("--limit-samples", type=int, help="Diagnostic subset, not a performance experiment")
            sub.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
            sub.add_argument("--accumulation-steps", type=int)
            sub.add_argument("--max-steps", type=int, help="Bounded diagnostic optimizer steps; no Test access")
            sub.add_argument("--name")
            sub.add_argument("--seed", type=int)
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--loss", choices=("asl", "bce"))
            sub.add_argument("--global-only", action="store_true", help="Disable the local CNN branch")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-root", default="data/therapeutic_2025")
    evaluate.add_argument("--feature-root")
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--predictions", help="Optional aligned sample probabilities in NPZ format")
    predict = commands.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    prediction_input = predict.add_mutually_exclusive_group(required=True)
    prediction_input.add_argument("--sequences", help="Sequence model: one resolved backbone per line")
    prediction_input.add_argument("--inputs", help="Research model: preprocessed multimodal NPZ")
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output", required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--runs", nargs="+", required=True, help="Research run directories")
    summarize.add_argument("--split", choices=("val", "test"), default="val")
    summarize.add_argument("--output", help="Optional JSON; otherwise print")
    gpu_check = commands.add_parser("gpu-check", help="Real-cache full-model server probes and bounded short training")
    gpu_check.add_argument("--models", nargs="+", choices=("trirelay", "geofactor", "protoroute"), default=["trirelay", "geofactor", "protoroute"])
    gpu_check.add_argument("--tasks", nargs="+", choices=("major15", "fine39"), default=["major15", "fine39"])
    gpu_check.add_argument("--esm", choices=("35m", "650m"), default="650m")
    gpu_check.add_argument("--feature-root")
    gpu_check.add_argument("--data-root", default="data/therapeutic_2025")
    gpu_check.add_argument("--device", default="cuda:0")
    gpu_check.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8, 16])
    gpu_check.add_argument("--precisions", nargs="+", choices=("fp32", "fp16", "bf16"), default=["fp32", "fp16"])
    gpu_check.add_argument("--steps", type=int, default=32)
    gpu_check.add_argument("--samples", type=int, default=128)
    gpu_check.add_argument("--output")
    gpu_check.add_argument("--inspect-only", action="store_true", help="Read GPU occupancy/cache metadata without training")
    args = parser.parse_args(argv)
    if args.command == "analyze":
        from .analysis import analyze_data
        result = analyze_data(args.data_root)
        from .trainer import write_json
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({"original_rows": result["original_rows"], "tracks": result["tracks"]}, ensure_ascii=False, indent=2))
    elif args.command in ("features", "train"):
        model_name = args.model if args.command == "train" else "sequence"
        config_path = args.config or f"configs/{model_name}.yaml"
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if args.esm:
            config["encoder"]["name"] = "facebook/esm2_t12_35M_UR50D" if args.esm=="35m" else "facebook/esm2_t33_650M_UR50D"
            config["data"]["features"] = f"features/esm2_{args.esm}"
            if args.command == "features":
                config["encoder"]["batch_size"] = 8 if args.esm=="650m" else 32
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
            prepare_features(config, args.splits, config["train"]["device"], resume_from=args.resume_from)
        else:
            if args.model != "sequence":
                from .research import train
                config["model_name"] = args.model
                if args.task:
                    config["data"]["track"] = args.task
                if args.ablate:
                    for key in args.ablate:
                        if key not in config["model"] or not isinstance(config["model"][key], bool):
                            raise ValueError(f"Unknown ablation switch: {key}")
                        config["model"][key] = False
                if args.threshold_strategy:
                    config["train"]["threshold_strategy"] = args.threshold_strategy
                if args.limit_samples:
                    config["train"]["limit_samples"] = args.limit_samples
                for key in ("precision", "accumulation_steps", "max_steps"):
                    value = getattr(args, key)
                    if value is not None:
                        config["train"][key] = value
            else:
                from .trainer import train
            for key in ("seed", "epochs"):
                value = getattr(args, key)
                if value is not None:
                    config["train"][key] = value
            if args.loss is not None:
                if model_name != "sequence":
                    raise ValueError("Research losses are configured in their YAML; --loss is a sequence-model option")
                config["loss"]["name"] = args.loss
            if args.global_only:
                if model_name != "sequence":
                    raise ValueError("Use --ablate for research models; --global-only is a sequence-model option")
                config["model"]["use_local"] = False
            name = args.name or f"{model_name}_{config['data']['track']}_seed{config['train']['seed']}"
            if model_name == "sequence":
                train(config, name)
            else:
                train(config, name, resume=args.resume)
    elif args.command == "evaluate":
        import torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if "model_name" in checkpoint:
            from .research import evaluate
        else:
            from .inference import evaluate
        feature_root = args.feature_root or checkpoint["config"]["data"]["features"]
        batch_size = args.batch_size or checkpoint["config"]["train"].get("batch_size", 128)
        evaluate(args.checkpoint, args.data_root, feature_root, args.split, args.device, args.output,
                 predictions=args.predictions, batch_size=batch_size)
    elif args.command == "gpu-check":
        from .runtime_check import inspect, run
        feature_root = args.feature_root or f"features/esm2_{args.esm}"
        if args.inspect_only:
            print(json.dumps(inspect(feature_root, args.device), ensure_ascii=False, indent=2))
        else:
            if not args.output:
                parser.error("gpu-check requires --output for real measurements")
            run(args.models, args.tasks, feature_root, args.device, args.output, args.esm,
                args.batch_sizes, args.precisions, args.steps, args.samples, args.data_root)
    elif args.command == "summarize":
        from .research import summarize_results
        result = summarize_results(args.runs, args.split)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with Path(args.output).open("x", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        if args.inputs:
            from .research import predict_inputs
            result = predict_inputs(args.checkpoint, args.inputs, args.device)
        else:
            import torch
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            if "model_name" in checkpoint:
                from .research import predict_sequences
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
