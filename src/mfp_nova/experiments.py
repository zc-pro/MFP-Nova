"""Fixed five-seed plan and read-only summaries; training stays in research.py."""
import argparse
import copy
import json
import os
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from .models import (RESEARCH_MODELS, LEGACY_MODELS, HH_MODELS, HH_V15_RANKER,
                     HH_V16, HH_V17, HH_V20, PEPTIDE_V3, PEPTIDE_V6, PEPTIDE_V7)
from .data import TRACKS
from .therapep import configure_task

MODELS = LEGACY_MODELS  # Preserve the original default/all five-model sweep.
# HH and the numbered peptide models have their own comparison protocols.
SUPPORTED_MODELS = tuple(model for model in RESEARCH_MODELS
                         if model not in (*HH_MODELS, HH_V15_RANKER, HH_V16, HH_V17, HH_V20,
                                          PEPTIDE_V3, PEPTIDE_V6, PEPTIDE_V7))
TASKS = ("major15", "fine39")  # Keep the existing default/all comparison cohort.
SEEDS = (42, 101, 202, 303, 404)
PROJECT = Path(__file__).resolve().parents[2]
ALIASES = {"accuracy": "sample_jaccard", "precision": "sample_precision", "coverage": "sample_recall",
           "absolute_true": "exact_match", "absolute_false": "hamming_loss"}


def training_settings(model, config=None, task=None):
    """Preserve the original three-model plan; newer models use YAML defaults."""
    config = config or yaml.safe_load((PROJECT/f"configs/{model}.yaml").read_text(encoding="utf-8"))
    if task is not None:
        config = copy.deepcopy(config)
        configure_task(config, task)
    settings = config["train"]
    return {"batch_size": settings["batch_size"] if model not in ("trirelay", "geofactor", "protoroute") else 64,
            "precision": settings.get("precision", "fp32"),
            "accumulation_steps": settings.get("accumulation_steps", 1),
            "metric_suite": settings.get("metric_suite", "task_specific"),
            "threshold_strategy": settings.get("threshold_strategy", "global"),
            "ema_decay": settings.get("ema_decay"),
            "ema_start_epoch": settings.get("ema_start_epoch")}


def run_family(model, task, seed):
    settings = training_settings(model, task=task)
    tag = f"{settings['precision']}_b{settings['batch_size']}"
    if model in ("helixset", "theraset"):
        tag += f"_a{settings['accumulation_steps']}_ema_{settings['metric_suite']}"
    encoder = "esm150m" if model == "setreread" else "esm650m"
    return f"{model}_{task}_{encoder}_seed{seed}_rtx5070_{tag}"


def metric_aliases(score):
    """Exact aliases from recorded legacy metrics, never fabricate old epochs."""
    score = dict(score)
    for alias, original in ALIASES.items():
        if alias not in score and original in score:
            score[alias] = score[original]
    return score


def scalar_metrics(mapping, prefix=""):
    values = {}
    for key, value in mapping.items():
        if key in ("undefined_metrics", "zero_division"):
            continue
        name = prefix+key
        if isinstance(value, dict):
            values.update(scalar_metrics(value, name+"."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values[name] = value
    return values


def aggregate(values):
    present = [v for v in values if v is not None]
    return {"mean": statistics.mean(present), "std": statistics.stdev(present) if len(present)>1 else None,
            "available_seeds": len(present)} if present else None


def seed_list(value):
    seeds = [int(s.strip()) for s in value.split(",") if s.strip()]
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in SEEDS for s in seeds):
        raise ValueError(f"Choose distinct seeds from the fixed protocol: {SEEDS}")
    return seeds


def live_training_runs():
    """Use live process commands; a checkpoint alone is not proof of running."""
    if os.name == "nt":
        command = ('Get-CimInstance Win32_Process | Where-Object { '
                   r'$_.Name -match "^(python|pythonw|mfp-nova)(\.exe)?$" } | '
                   'Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress')
        response = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                                  check=True, capture_output=True, text=True)
        rows = json.loads(response.stdout or "[]")
        rows = rows if isinstance(rows, list) else [rows]
        processes = [(int(r["ProcessId"]), r.get("CommandLine") or "") for r in rows]
    else:
        response = subprocess.run(["ps", "-eo", "pid=,args="], check=True, capture_output=True, text=True)
        processes = [(int(line.split(None, 1)[0]), line.split(None, 1)[1])
                     for line in response.stdout.splitlines() if len(line.split(None, 1))==2]
    found = {}
    for pid, command in processes:
        if not re.search(r"(?:mfp_nova\.cli|mfp-nova).*\btrain\b", command):
            continue
        match = re.search(r'''--name(?:=|\s+)(?:"([^"]+)"|'([^']+)'|(\S+))''', command)
        if match:
            found[next(v for v in match.groups() if v)] = pid
    return found


def compatibility_issues(metadata, result, directory, identity=None):
    issues = []
    identity = metadata or result or identity
    expected = yaml.safe_load((PROJECT/f"configs/{identity['model']}.yaml").read_text(encoding="utf-8"))
    configure_task(expected, identity['task'])
    fixed = training_settings(identity["model"], expected)
    defaults = {"precision": "fp32", "accumulation_steps": 1, "threshold_strategy": "global",
                "metric_suite": "task_specific", "ema_decay": None}
    if result:
        defaults["metric_suite"] = result.get("metric_protocol", {}).get("metric_suite", defaults["metric_suite"])
        for key in ("precision", "accumulation_steps", "threshold_strategy", "metric_suite"):
            if result.get(key, defaults[key]) != fixed[key]:
                issues.append(f"{key} differs from {fixed[key]}")
    if metadata:
        config = metadata["config"]
        if metadata["encoder"]["name"] != expected["encoder"]["name"]:
            issues.append(f"encoder differs from {expected['encoder']['name']}")
        if config.get("peptide_encoder") != expected.get("peptide_encoder"):
            issues.append("peptide encoder configuration differs from the fixed protocol")
        for key in ("model", "loss"):
            if config[key]!=expected[key]:
                issues.append(f"{key} configuration differs from the fixed protocol")
        settings = config["train"]
        if expected["loss"].get("rdrop", 0):
            for key in ("rdrop_start_epoch", "rdrop_ramp_epochs"):
                if settings.get(key) != expected["train"][key]:
                    issues.append(f"train.{key} differs from the fixed protocol")
        if settings.get("reconstruction_decay") is not None:
            issues.append("train.reconstruction_decay differs from the fixed protocol")
        if settings.get("reconstruction_gradient", "sum") != "sum":
            issues.append("train.reconstruction_gradient differs from the fixed protocol")
        for key, value in {**fixed, **{k: expected["train"][k] for k in (
                           "epochs", "learning_rate", "weight_decay", "patience", "warmup_epochs",
                           "aux_warmup_epochs", "gradient_clip") if k in expected["train"]}}.items():
            actual = settings.get(key, defaults.get(key, value))
            if actual != value:
                issues.append(f"train.{key} differs from {value}")
        if settings.get("limit_samples") or settings.get("max_steps") or config["data"].get("augment", False):
            issues.append("diagnostic/augmented training is not the original-row formal protocol")
        task = identity["task"]
        cohort = ("therapep_author_published_all49" if task == "therapep49" else
                  task if task in ("therapep48", "therapep49_positive") else
                  "sequence_only_original" if identity["model"] in ("termiscale", "motifspan", "setreread") else
                  "rich48_author_positive_multimodal" if task == "rich48" else "original_resolved_backbone")
        if metadata.get("cohort") not in (None, cohort):
            issues.append("training cohort differs")
    elif identity["model"] in ("helixset", "theraset", "setreread"):
        if not directory.name.startswith(run_family(identity["model"], identity["task"], identity["seed"])):
            issues.append("results-only directory does not identify the model's training settings")
    elif "esm650m" not in directory.name or "fp32_b64" not in directory.name:
        issues.append("results-only directory does not identify ESM-650M / FP32 / batch 64")
    return issues


def discover_runs(root):
    records = []
    for directory in sorted(Path(root).glob("*")):
        if not directory.is_dir():
            continue
        metadata_path, result_path = directory/"metadata.json", directory/"results.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
        identity = result or metadata
        if identity is None:
            match = re.match(r"^(trirelay|geofactor|protoroute|helixset|theraset)_(major15|fine39|therapep49)_esm650m_seed(\d+)_", directory.name)
            if not match:
                continue
            identity = {"model": match[1], "task": match[2], "seed": int(match[3])}
        if identity.get("model") not in SUPPORTED_MODELS or identity.get("task") not in TRACKS:
            continue
        if identity.get("diagnostic_subset"):
            continue
        issues = compatibility_issues(metadata, result, directory, identity)
        family = run_family(identity["model"], identity["task"], identity["seed"])
        if issues and not (directory.name==family or directory.name.startswith(family+"_")):
            # Other encoders and intentional ablations are separate experiments,
            # not corrupt attempts at this fixed five-seed job.
            continue
        if result:
            suite = result.get("metric_suite", result.get("metric_protocol", {}).get("metric_suite"))
            threshold_ok = result.get("threshold") == .5 if suite == "micro_f1" else result.get("threshold_fit_split") == "val"
            if not threshold_ok or not result.get("validation"):
                issues.append("result lacks completed Validation under its threshold protocol")
        records.append({"directory": directory, "metadata": metadata, "result": result,
                        "model": identity["model"], "task": identity["task"], "seed": int(identity["seed"]),
                        "issues": issues, "evidence": "metadata" if metadata else "result_and_run_name_only"})
    return records


def make_plan(root="runs", models=LEGACY_MODELS, tasks=TASKS, seeds=SEEDS, running=None):
    records = discover_runs(root)
    running = live_training_runs() if running is None else running
    jobs = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for model in models:
        for task in tasks:
            for seed in seeds:
                base = run_family(model, task, seed)
                matches = [r for r in records if (r["model"], r["task"], r["seed"])==(model, task, seed)]
                compatible = [r for r in matches if not r["issues"]]
                complete = [r for r in compatible if r["result"]]
                active = {name: pid for name, pid in running.items() if name==base or name.startswith(base+"_") or
                          any(name==r["directory"].name for r in matches)}
                job = {"model": model, "task": task, "seed": seed, "run_name": base, "resume": None,
                       "result_path": None, "running": active, "issues": [],
                               "train_settings": training_settings(model, task=task)}
                if complete:
                    # Count one result per seed. Prefer the furthest completed
                    # continuation, never choose a repeat by its Validation score.
                    chosen = max(complete, key=lambda r: (r["result"].get("global_step", 0),
                                                           (r["directory"]/"results.json").stat().st_mtime))
                    result = chosen["result"]
                    job.update(status="completed", run_name=chosen["directory"].name,
                               result_path=str(chosen["directory"]/"results.json"),
                               protocol_evidence=chosen["evidence"], best_epoch=result["best_epoch"],
                               macro_ap=result["validation"]["macro_ap"], micro_ap=result["validation"]["micro_ap"],
                               micro_f1=result["validation"]["micro_f1"], completed_attempts=len(complete))
                elif active:
                    job.update(status="running", run_name=next(iter(active)))
                elif matches and not compatible:
                    job.update(status="incompatible", issues=[issue for r in matches for issue in r["issues"]])
                else:
                    checkpoints = [r["directory"]/"last.pt" for r in compatible if (r["directory"]/"last.pt").exists()]
                    if checkpoints:
                        checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                        job.update(status="unfinished_checkpoint", resume=str(checkpoint), run_name=f"{base}_resume_{stamp}")
                    elif compatible or (Path(root)/base).exists():
                        job.update(status="unfinished_no_checkpoint", run_name=f"{base}_retry_{stamp}")
                    else:
                        job["status"] = "not_started"
                jobs.append(job)
    counts = {name: sum(j["status"]==name for j in jobs) for name in (
        "completed", "running", "unfinished_checkpoint", "unfinished_no_checkpoint", "not_started", "incompatible")}
    return {"fixed_seeds": list(SEEDS), "requested_seeds": list(seeds), "total": len(jobs),
            "counts": counts, "jobs": jobs, "checkpoint_status_note": "unfinished checkpoint is not loaded until Trainer resumes it"}


def summarize(plan, label=None, group=None):
    comparisons = []
    for model in SUPPORTED_MODELS:
        for task in TRACKS:
            jobs = [j for j in plan["jobs"] if j["model"]==model and j["task"]==task]
            if not jobs:
                continue
            records, missing = [], []
            for job in jobs:
                if job["status"]!="completed":
                    continue
                path = Path(job["result_path"])
                result = json.loads(path.read_text(encoding="utf-8"))
                score = result["validation"]
                if group:
                    entry = result.get("validation_groups", {}).get(group)
                    if not entry or entry.get("metrics") is None:
                        missing.append({"seed": job["seed"], "reason": "group_not_recorded_or_no_samples"})
                        continue
                    score = entry["metrics"]
                if label:
                    metadata_path = path.parent/"metadata.json"
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
                    names = result.get("labels") or metadata.get("labels")
                    if names is None:
                        names = json.loads((PROJECT/f"data/therapeutic_2025/tracks/{task}/labels.json").read_text(encoding="utf-8"))["labels"]
                    if label not in names:
                        raise ValueError(f"Unknown label {label!r} for {task}")
                    index = names.index(label)
                    score = {key.removeprefix("label_"): score[key][index] for key in (
                        "label_ap", "label_auroc", "label_precision", "label_recall", "label_f1", "label_mcc",
                        "label_sensitivity", "label_specificity", "label_positive_support") if key in score}
                score = metric_aliases(score)
                records.append({"seed": job["seed"], "run": str(path.parent), "metrics": score,
                                "protocol_evidence": job["protocol_evidence"],
                                **{key: result.get(key) for key in ("best_epoch", "epochs_completed", "trainable_parameters",
                                    "elapsed_seconds", "peak_cuda_memory_bytes")}})
            scalars = [scalar_metrics(r["metrics"]) for r in records]
            keys = sorted(set().union(*(s.keys() for s in scalars))) if scalars else []
            comparisons.append({"model": model, "task": task, "seeds": [r["seed"] for r in records],
                "metric_suite": training_settings(model)["metric_suite"],
                "seed_count": len(records), "complete_five_seed": set(r["seed"] for r in records)==set(SEEDS),
                "metrics": {k: aggregate([r.get(k) for r in scalars]) for k in keys},
                "resources": {k: aggregate([r[k] for r in records]) for k in ("best_epoch", "epochs_completed",
                    "trainable_parameters", "elapsed_seconds", "peak_cuda_memory_bytes")},
                "missing": missing, "per_seed": records})
    return {"scope": "best_checkpoint_validation_fitted_threshold; no Test selection", "std_ddof": 1,
            "label": label, "group": group, "fixed_seeds": list(SEEDS), "comparisons": comparisons}


def print_plan(plan):
    print(f"Completed {plan['counts']['completed']}/{plan['total']} | fixed seeds={','.join(map(str, SEEDS))}")
    for job in plan["jobs"]:
        detail = (f"best={job['best_epoch']} MacroAP={job['macro_ap']:.4f} MicroAP={job['micro_ap']:.4f} MicroF1={job['micro_f1']:.4f}"
                  if job["status"]=="completed" else " ".join(job["issues"]))
        if job["running"]:
            detail += f" live PIDs={list(job['running'].values())}"
        print(f"{job['status']:25} {job['model']:10} {job['task']:7} seed={job['seed']:3} {detail}")
    print("unfinished_* means not observed running now; the checkpoint is validated by Trainer on resume.")


def print_summary(report):
    for item in report["comparisons"]:
        print(f"\n{item['task']} / {item['model']} | n={item['seed_count']}/5 seeds={item['seeds']} | "
              + f"metric_suite={item['metric_suite']} | "
              + ("five-seed" if item["complete_five_seed"] else "provisional"))
        def value(key, resource=False):
            stat = item["resources" if resource else "metrics"].get(key)
            if stat is None:
                return "NA (not recorded/undefined)"
            deviation = "NA" if stat["std"] is None else f"{stat['std']:.4f}"
            return f"{stat['mean']:.4f} +/- {deviation} (n={stat['available_seeds']})"
        sections = [
            ("AP/F1", ("macro_ap", "micro_ap", "macro_f1", "micro_f1", "sample_f1")),
            ("Five", ("accuracy", "precision", "coverage", "absolute_true", "absolute_false")),
            ("Tail", ("head_macro_ap", "medium_macro_ap", "tail_macro_ap")),
            ("MFP", ("mfp_detection.precision", "mfp_detection.recall", "mfp_detection.f1")),
        ] if report["label"] is None else [("Label", tuple(item["metrics"]))]
        for name, keys in sections:
            print(name+" | "+" | ".join(f"{k}={value(k)}" for k in keys))
        print("Resources | "+" | ".join(f"{k}={value(k, True)}" for k in item["resources"]))
        if item["missing"]:
            print("Missing:", item["missing"])
        if any(r["protocol_evidence"]=="result_and_run_name_only" for r in item["per_seed"]):
            print("Legacy results-only record included; its complete metadata is not available here.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "summary"))
    parser.add_argument("--model", choices=("all", *SUPPORTED_MODELS), default="all")
    parser.add_argument("--task", choices=("all", *TRACKS), default="all")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--group")
    args = parser.parse_args(argv)
    if args.label and args.group:
        parser.error("Choose --label or --group; group summaries do not duplicate per-label arrays")
    plan = make_plan(args.runs_root, LEGACY_MODELS if args.model=="all" else (args.model,),
                     TASKS if args.task=="all" else (args.task,), seed_list(args.seeds))
    result = plan if args.command=="plan" else summarize(plan, args.label, args.group)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    elif args.command=="plan":
        print_plan(plan)
    else:
        print_summary(result)


if __name__ == "__main__":
    main()
