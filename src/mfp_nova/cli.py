import argparse
import json
from pathlib import Path

import yaml

from .models import RESEARCH_MODELS, LEGACY_MODELS
from .models.hypotheses import MODELS as HYPOTHESIS_MODELS
from .data import TRACKS
from .therapep import TRACKS as THERAPEP_TRACKS


def main(argv=None):
    parser = argparse.ArgumentParser(description="MFP-Nova research commands")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--data-root", default="data/therapeutic_2025")
    analyze.add_argument("--output", default="docs/data_summary.json")
    download = commands.add_parser("download-therapep", help="Download author-published Train/Val/Test CSVs")
    download.add_argument("--output", default="data/therapep_ai")
    variants = commands.add_parser("prepare-therapep-variants", help="Create aligned positive-only 49/48-label CSVs")
    variants.add_argument("--source", default="data/therapep_ai")
    variants.add_argument("--output-root", default="data")
    rich = commands.add_parser("prepare-rich48", help="Match author positive48 members to raw multimodal records")
    rich.add_argument("--author-root", default="data/therapep_ai")
    rich.add_argument("--source-root", default="data/therapeutic_2025")
    rich.add_argument("--output", default="data/therapeutic_2025/rich48")
    for command in ("features", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--config")
        sub.add_argument("--data-root")
        sub.add_argument("--feature-root")
        sub.add_argument("--peptide-feature-root", help="Short-peptide cache directory for HelixSet/TheraSet")
        sub.add_argument("--track", choices=TRACKS)
        sub.add_argument("--device")
        sub.add_argument("--batch-size", type=int, help="Encoder batch for features; head batch for train")
        sub.add_argument("--esm", choices=("35m", "650m"), help="Select ESM size and matching feature directory")
        if command == "features":
            sub.add_argument("--encoder-view", choices=("primary", "peptide"), default="primary",
                             help="Extract primary ESM or the configured short-peptide encoder")
            sub.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
            sub.add_argument("--resume-from", type=int, default=0,
                             help="Completed rows in the first split, from the interrupted extraction log")
        else:
            sub.add_argument("--model", choices=("sequence", *HYPOTHESIS_MODELS, *RESEARCH_MODELS), default="sequence")
            sub.add_argument("--metric-suite", choices=("traditional5", "task_specific", "micro_f1"),
                             help="Research models: primary report and checkpoint selection protocol")
            sub.add_argument("--task", choices=TRACKS, help="Alias for --track")
            sub.add_argument("--resume", help="last.pt with optimizer and RNG state; creates a new run")
            sub.add_argument("--stage", choices=("A", "B", "C"), help="PrivilegedBridge training phase")
            sub.add_argument("--initialize-from", help="B0 checkpoint for H1–H6, or preceding PrivilegedBridge phase")
            sub.add_argument("--annotations", help="B0/H1–H8: aligned NPZ of confirmed label and observation states")
            sub.add_argument("--unlabeled-weight", type=float, help="Sequence study: optional weak BCE weight for unannotated zero bits")
            sub.add_argument("--fold-dir", help="Sequence study: share training-only B0 fold checkpoints")
            sub.add_argument("--head-epochs", type=int, help="Sequence study: frozen head training epochs")
            sub.add_argument("--finetune-epochs", type=int, help="Sequence study: H4/H6 backbone fine-tuning epochs")
            sub.add_argument("--evidence-selection", choices=("attention", "random", "none"), help="H6 residue mask control")
            sub.add_argument("--termiscale-routing", choices=("content", "position", "uniform"),
                             help="TermiScale: content+position, position-only, or equal scale weights")
            sub.add_argument("--termiscale-pooling", choices=("conditional", "fixed", "uniform"),
                             help="TermiScale: conditional endpoints, learned label-only endpoints, or uniform LME")
            sub.add_argument("--motifspan-global-only", action="store_true",
                             help="Train the MotifSpan backbone with its global head only")
            sub.add_argument("--setreread-interval", action="store_true",
                             help="SetReread: use a short CNN with conditional contiguous-interval readout")
            sub.add_argument("--theraset-b0-backbone", action="store_true",
                             help="Compare the B0 backbone under the pure AA34 TheraSet training protocol")
            sub.add_argument("--theraset-b0-local-max", action="store_true",
                             help="Use the AA34 B0 backbone with local max pooling and the global label query")
            sub.add_argument("--theraset-b0-local-cnn", action="store_true",
                             help="Use the TheraSet multiscale CNN before B0 local max pooling and the Transformer")
            sub.add_argument("--resume-in-place", action="store_true", help="Research models: append to the same interrupted run")
            sub.add_argument("--stop-file", help="Research models: pause after a complete epoch when this file exists")
            sub.add_argument("--ablate", nargs="*", help="Model switches to disable, e.g. feedback high_order")
            sub.add_argument("--threshold-strategy", choices=("global", "per_label"))
            sub.add_argument("--limit-samples", type=int, help="Diagnostic subset, not a performance experiment")
            sub.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
            sub.add_argument("--accumulation-steps", type=int)
            sub.add_argument("--num-workers", type=int, help="DataLoader worker processes; 0 loads in the main process")
            sub.add_argument("--lr-decay-epochs", type=int,
                             help="Research models: cosine reaches its floor at this epoch; independent of maximum epochs")
            sub.add_argument("--latent-depth", type=int, help="HelixSet latent fusion depth")
            sub.add_argument("--max-steps", type=int, help="Bounded diagnostic optimizer steps; no Test access")
            sub.add_argument("--reconstruction-decay", nargs=2, type=int, metavar=("START", "END"),
                             help="Research models: cosine-decay reconstruction weight after warmup")
            sub.add_argument("--reconstruction-floor", type=float, help="Fraction of initial reconstruction weight retained after decay (default 0.2)")
            sub.add_argument("--ema-decay", type=float, help="Optional parameter EMA; evaluate online and averaged weights in the same run")
            sub.add_argument("--ema-start-epoch", type=int, help="First EMA update epoch (default: after LR warmup)")
            sub.add_argument("--consistency-weight", type=float, help="TriRelay: optional clean-EMA / masked-student consistency (candidate: 0.1)")
            sub.add_argument("--rdrop-weight", type=float, help="HelixSet/TheraSet: symmetric Bernoulli dropout consistency; 0 disables it")
            sub.add_argument("--rdrop-start-epoch", type=int, help="First R-Drop epoch for supported models")
            sub.add_argument("--rdrop-ramp-epochs", type=int, help="R-Drop weight ramp length")
            sub.add_argument("--consistency-start-epoch", type=int, help="First consistency epoch (default: EMA start + 1)")
            sub.add_argument("--consistency-ramp-epochs", type=int, help="Consistency weight ramp length (default: 5)")
            sub.add_argument("--function-aux-weight", type=float, help="TriRelay: real paired major15/fine39 auxiliary supervision (candidate: 0.1)")
            sub.add_argument("--function-hierarchy-weight", type=float, help="Soft subtype <= parent constraint (default: 0.02 when auxiliary is enabled)")
            sub.add_argument("--relation-view-weight", type=float, help="TriRelay: masked label-relation consistency plus confident evidence anchoring (candidate: 0.1)")
            sub.add_argument("--relation-dropout", type=float, help="Pair edges/factors removed in the auxiliary relation view (default: 0.2)")
            sub.add_argument("--reconstruction-gradient", choices=("sum", "cap", "project"), help="TriRelay: sum=original; cap=shared auxiliary norm cap; project=conflict projection plus norm cap")
            sub.add_argument("--reconstruction-grad-ratio", type=float, help="Maximum shared reconstruction/main gradient norm ratio (default: 1)")
            sub.add_argument("--shared-private", action=argparse.BooleanOptionalAction, default=None,
                             help="TriRelay: preserve private pre-relay evidence alongside shared interactions")
            sub.add_argument("--branch-supervision", choices=("mixed_private", "shared_private"),
                             help="TriRelay: original mixed/private branches or symmetric pre-feedback shared/private supervision")
            sub.add_argument("--evidence-fusion", choices=("residual", "normalized_competitive", "normalized_residual"),
                             help="Shared-private TriRelay: original addition, RMS competition, or RMS addition")
            sub.add_argument("--name")
            sub.add_argument("--seed", type=int)
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--loss", choices=("asl", "bce"))
            sub.add_argument("--global-only", action="store_true", help="Disable the local CNN branch")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data-root", help="Defaults to the checkpoint's data root")
    evaluate.add_argument("--feature-root")
    evaluate.add_argument("--peptide-feature-root")
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--predictions", help="Optional aligned sample probabilities in NPZ format")
    evaluate.add_argument("--annotations", help="Sequence study: aligned annotation NPZ for evaluation")
    evaluate.add_argument("--evidence-diagnostics", action="store_true", help="H6: run keep/delete and matched random controls")
    evaluate.add_argument("--decision", help="Validation-fitted research-model decision.json; raw ranking scores stay unchanged")
    evaluate.add_argument("--without-private-evidence", action="store_true",
                          help="Shared-private TriRelay: use shared evidence alone in both reads, retaining weights and saved threshold")
    predict = commands.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    prediction_input = predict.add_mutually_exclusive_group(required=True)
    prediction_input.add_argument("--sequences", help="Sequence model: one resolved backbone per line")
    prediction_input.add_argument("--inputs", help="Research model: preprocessed multimodal NPZ")
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--output", required=True)
    predict.add_argument("--decision", help="Validation-fitted research-model decision.json")
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--runs", nargs="+", required=True, help="Research run directories")
    summarize.add_argument("--split", choices=("val", "test"), default="val")
    summarize.add_argument("--output", help="Optional JSON; otherwise print")
    gpu_check = commands.add_parser("gpu-check", help="Real-cache full-model server probes and bounded short training")
    gpu_check.add_argument("--models", nargs="+", choices=RESEARCH_MODELS, default=list(LEGACY_MODELS))
    gpu_check.add_argument("--tasks", nargs="+", choices=TRACKS, default=["major15", "fine39"])
    gpu_check.add_argument("--esm", choices=("35m", "650m"), default="650m")
    gpu_check.add_argument("--feature-root")
    gpu_check.add_argument("--peptide-feature-root")
    gpu_check.add_argument("--data-root", default="data/therapeutic_2025")
    gpu_check.add_argument("--device", default="cuda:0")
    gpu_check.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8, 16])
    gpu_check.add_argument("--precisions", nargs="+", choices=("fp32", "fp16", "bf16"), default=["fp32", "fp16"])
    gpu_check.add_argument("--steps", type=int, default=32)
    gpu_check.add_argument("--samples", type=int, default=128)
    gpu_check.add_argument("--output")
    gpu_check.add_argument("--inspect-only", action="store_true", help="Read GPU occupancy/cache metadata without training")
    args = parser.parse_args(argv)
    if args.command == "train" and args.setreread_interval and args.model != "setreread":
        parser.error("--setreread-interval requires --model setreread")
    if args.command == "prepare-rich48":
        from .rich48 import build_rich48
        print(json.dumps(build_rich48(args.author_root, args.source_root, args.output), ensure_ascii=False, indent=2))
    elif args.command == "prepare-therapep-variants":
        from .therapep import prepare_positive_variants
        print(json.dumps(prepare_positive_variants(args.source, args.output_root), ensure_ascii=False, indent=2))
    elif args.command == "download-therapep":
        from .therapep import download_data
        print(json.dumps(download_data(args.output), ensure_ascii=False, indent=2))
    elif args.command == "analyze":
        from .analysis import analyze_data
        result = analyze_data(args.data_root)
        from .trainer import write_json
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        print(json.dumps({"original_rows": result["original_rows"], "tracks": result["tracks"]}, ensure_ascii=False, indent=2))
    elif args.command == "train" and args.model in HYPOTHESIS_MODELS:
        if args.theraset_b0_backbone or args.theraset_b0_local_max or args.theraset_b0_local_cnn:
            parser.error("TheraSet B0 comparisons require --model theraset")
        from .hypothesis import train_from_cli
        config = yaml.safe_load(Path(args.config or "configs/b0.yaml").read_text(encoding="utf-8"))
        train_from_cli(args, config)
    elif args.command in ("features", "train"):
        model_name = args.model if args.command == "train" else "sequence"
        config_path = args.config or f"configs/{model_name}.yaml"
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if args.command == "train":
            if args.theraset_b0_backbone or args.theraset_b0_local_max or args.theraset_b0_local_cnn:
                if args.model != "theraset":
                    parser.error("TheraSet B0 comparisons require --model theraset")
                config["model"].update(b0_backbone=True, evidence_decoder=False, chemical_evidence=False)
                if args.theraset_b0_local_max:
                    config["model"]["b0_local_max"] = True
                if args.theraset_b0_local_cnn:
                    config["model"]["b0_local_cnn"] = True
            if args.termiscale_routing is not None or args.termiscale_pooling is not None:
                if args.model != "termiscale":
                    raise ValueError("TermiScale routing/pooling options require --model termiscale")
                for key in ("routing", "pooling"):
                    value = getattr(args, "termiscale_"+key)
                    if value is not None:
                        config["model"][key] = value
            if args.motifspan_global_only:
                if args.model != "motifspan":
                    raise ValueError("--motifspan-global-only requires --model motifspan")
                config["model"]["global_only"] = True
            if args.setreread_interval:
                config["model"]["interval_evidence"] = True
            if args.stage is not None or args.initialize_from is not None:
                if args.model != "privilegedbridge":
                    raise ValueError("stage/initialize-from are PrivilegedBridge options")
            if args.resume and args.initialize_from:
                raise ValueError("Use resume within a phase or initialize-from between phases")
            if args.model == "privilegedbridge":
                from .models.privilegedbridge import configure_stage
                configure_stage(config, args.stage or config["model"].get("stage", "A"))
                if args.initialize_from:
                    config["train"]["initialize_from"] = args.initialize_from
        from .therapep import configure_task
        configure_task(config, (args.task if args.command == "train" else None)
                       or args.track or config["data"]["track"])
        peptide_view = args.command == "features" and args.encoder_view == "peptide"
        if peptide_view and args.esm:
            raise ValueError("--encoder-view peptide uses peptide_encoder; omit --esm")
        if args.esm:
            if config.get("model", {}).get("esm_tail") and args.esm != "650m":
                raise ValueError("This HelixSet ESM-tail configuration uses ESM650M")
            config["encoder"]["name"] = "facebook/esm2_t12_35M_UR50D" if args.esm=="35m" else "facebook/esm2_t33_650M_UR50D"
            track = config["data"]["track"]
            config["data"]["features"] = (f"features/rich48/esm2_{args.esm}" if track == "rich48" else
                                           f"features/therapep_ai/esm2_{args.esm}" if track in THERAPEP_TRACKS else
                                           f"features/esm2_{args.esm}")
            layer = config["encoder"].get("representation_layer")
            if layer is not None:
                config["data"]["features"] += f"_layer{layer}"
            if args.command == "features":
                config["encoder"]["batch_size"] = 8 if args.esm=="650m" else 32
        feature_key = "peptide_features" if peptide_view else "features"
        for arg, key in ((args.data_root, "root"), (args.feature_root, feature_key), (args.track, "track")):
            if arg is not None:
                config["data"][key] = arg
        if args.peptide_feature_root is not None:
            config["data"]["peptide_features"] = args.peptide_feature_root
        if args.device is not None:
            config["train"]["device"] = args.device
        if args.batch_size is not None:
            section = ("peptide_encoder" if peptide_view else "encoder") if args.command == "features" else "train"
            config[section]["batch_size"] = args.batch_size
        if args.command == "features":
            from .encoder import prepare_features
            prepare_features(config, args.splits, config["train"]["device"], resume_from=args.resume_from,
                             encoder_view=args.encoder_view)
        else:
            if args.branch_supervision is not None and args.model != "trirelay":
                raise ValueError("Branch supervision is a TriRelay training option")
            if args.evidence_fusion is not None and args.model != "trirelay":
                raise ValueError("Evidence fusion is a TriRelay model option")
            if (args.relation_view_weight is not None or args.relation_dropout is not None) and args.model != "trirelay":
                raise ValueError("Relation views are a TriRelay training option")
            if args.metric_suite is not None:
                if args.model == "sequence":
                    raise ValueError("--metric-suite requires a research model")
                config["train"]["metric_suite"] = args.metric_suite
            if args.model != "sequence":
                from .research import train
                config["model_name"] = args.model
                if args.latent_depth is not None:
                    if args.model != "helixset" or args.latent_depth < 1:
                        raise ValueError("--latent-depth requires HelixSet and a positive depth")
                    if args.resume:
                        raise ValueError("A latent-depth comparison requires a fresh run")
                    config["model"]["latent_depth"] = args.latent_depth
                if args.shared_private is not None:
                    if args.model != "trirelay":
                        raise ValueError("Shared/private evidence is a TriRelay model option")
                    config["model"]["shared_private"] = args.shared_private
                if args.evidence_fusion is not None:
                    config["model"]["evidence_fusion"] = args.evidence_fusion
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
                for key in ("precision", "accumulation_steps", "lr_decay_epochs", "max_steps", "reconstruction_decay", "reconstruction_floor", "ema_decay", "ema_start_epoch",
                            "rdrop_start_epoch", "rdrop_ramp_epochs",
                            "consistency_weight", "consistency_start_epoch", "consistency_ramp_epochs",
                            "function_aux_weight", "function_hierarchy_weight",
                            "reconstruction_gradient", "reconstruction_grad_ratio", "branch_supervision",
                            "relation_view_weight", "relation_dropout"):
                    value = getattr(args, key)
                    if value is not None:
                        config["train"][key] = value
            else:
                if args.latent_depth is not None or args.lr_decay_epochs is not None:
                    raise ValueError("Latent depth / LR decay controls require a research model")
                if args.shared_private is not None:
                    raise ValueError("Shared/private evidence is a TriRelay model option")
                if args.reconstruction_gradient is not None or args.reconstruction_grad_ratio is not None:
                    raise ValueError("Reconstruction gradient balancing is a TriRelay training option")
                if args.function_aux_weight is not None or args.function_hierarchy_weight is not None:
                    raise ValueError("Function auxiliary supervision is a TriRelay training option")
                if any(getattr(args, key) is not None for key in ("consistency_weight", "consistency_start_epoch", "consistency_ramp_epochs")):
                    raise ValueError("Consistency is a TriRelay training option")
                if args.ema_decay is not None or args.ema_start_epoch is not None:
                    raise ValueError("EMA is a research-model option")
                if args.reconstruction_decay is not None or args.reconstruction_floor is not None:
                    raise ValueError("Reconstruction scheduling is a research-model option")
                from .trainer import train
            for key in ("seed", "epochs", "num_workers"):
                value = getattr(args, key)
                if value is not None:
                    config["train"][key] = value
            if args.rdrop_weight is not None:
                if model_name not in ("helixset", "theraset", "termiscale", "motifspan", "privilegedbridge"):
                    raise ValueError("This model does not support R-Drop")
                config["loss"]["rdrop"] = args.rdrop_weight
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
                if args.resume_in_place or args.stop_file:
                    raise ValueError('Pause/in-place resume currently requires a research model')
                train(config, name)
            else:
                continuation = {}
                if args.resume_in_place:
                    continuation['resume_in_place'] = True
                if args.stop_file:
                    continuation['stop_file'] = args.stop_file
                try:
                    train(config, name, resume=args.resume, **continuation)
                except KeyboardInterrupt:
                    run = Path(config['train']['output']) / name
                    if (run/'last.pt').exists():
                        print(f'\nTraining interrupted. Resume from the last complete epoch in {run}. '
                              'The unfinished epoch will be replayed.', flush=True)
                    else:
                        print(f'\nTraining interrupted before the first complete epoch. '
                              f'Restart epoch 1 in the same directory: {run}', flush=True)
                    raise SystemExit(130) from None
    elif args.command == "evaluate":
        import torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if checkpoint.get("model_name") in HYPOTHESIS_MODELS:
            from .hypothesis import evaluate
            if args.decision or args.feature_root or args.peptide_feature_root or args.without_private_evidence:
                parser.error("B0/H1–H8 evaluate raw sequences with the checkpoint's saved decision rule")
            if args.evidence_diagnostics and checkpoint["model_name"] != "h6":
                parser.error("--evidence-diagnostics requires an H6 checkpoint")
            evaluate(checkpoint, args.device, args.split, args.output, data_root=args.data_root,
                     annotations=args.annotations, batch_size=args.batch_size, predictions=args.predictions,
                     diagnostics=args.evidence_diagnostics)
            return
        if args.without_private_evidence and "model_name" not in checkpoint:
            raise ValueError("Private evidence ablation requires a shared-private TriRelay checkpoint")
        if "model_name" in checkpoint:
            from .research import evaluate
        else:
            from .inference import evaluate
        feature_root = args.feature_root or checkpoint["config"]["data"]["features"]
        batch_size = args.batch_size or checkpoint["config"]["train"].get("batch_size", 128)
        extra = {}
        if args.peptide_feature_root:
            if "model_name" not in checkpoint:
                raise ValueError("Peptide cache requires a research model checkpoint")
            extra['peptide_feature_root'] = args.peptide_feature_root
        if args.without_private_evidence:
            extra['without_private_evidence'] = True
        if args.decision:
            if "model_name" not in checkpoint:
                raise ValueError("Function decisions are supported for research models")
            extra['decision_path'] = args.decision
        data_root = args.data_root or checkpoint["config"]["data"]["root"]
        evaluate(args.checkpoint, data_root, feature_root, args.split, args.device, args.output,
                 predictions=args.predictions, batch_size=batch_size, **extra)
    elif args.command == "gpu-check":
        from .runtime_check import inspect, run
        feature_root = args.feature_root or ("features/esm2_peptide_150m" if args.models == ["setreread"]
                                            else f"features/esm2_{args.esm}")
        if args.inspect_only:
            print(json.dumps(inspect(feature_root, args.device), ensure_ascii=False, indent=2))
        else:
            if not args.output:
                parser.error("gpu-check requires --output for real measurements")
            run(args.models, args.tasks, feature_root, args.device, args.output, args.esm,
                args.batch_sizes, args.precisions, args.steps, args.samples, args.data_root,
                peptide_feature_root=args.peptide_feature_root)
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
            result = predict_inputs(args.checkpoint, args.inputs, args.device, decision_path=args.decision)
        else:
            import torch
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            sequences = [line.strip() for line in Path(args.sequences).read_text(encoding="utf-8").splitlines() if line.strip()]
            if checkpoint.get("model_name") in HYPOTHESIS_MODELS:
                from .hypothesis import predict
                if args.decision:
                    parser.error("B0/H1–H8 predict with the checkpoint's saved decision rule")
                result = predict(checkpoint, sequences, args.device)
            else:
                if "model_name" in checkpoint:
                    from .research import predict_sequences
                else:
                    if args.decision:
                        raise ValueError("Function decisions are supported for research models")
                    from .inference import predict_sequences
                extra = {'decision_path': args.decision} if args.decision else {}
                result = predict_sequences(args.checkpoint, sequences, args.device, **extra)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
