"""Compare finished Fine39 runs on the same observed Validation annotations."""

import argparse
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tarfile import open as open_tar
from zipfile import ZipFile

import numpy as np
import pandas as pd

from mfp_nova.data import PeptideData
from mfp_nova.hypothesis_data import family_folds
from mfp_nova.metrics import evaluate_metrics, fit_global_threshold
from mfp_nova.multimodal import function_projection, independent_functions
from mfp_nova.xbridge_data import recovery_diagnostics


def load_run(path, prefix=""):
    filenames = ("results.json", "metadata.json", "history.jsonl", "best_validation.npz")
    if path.is_dir():
        files = [(path / prefix / name).read_bytes() for name in filenames]
    elif path.suffix == ".zip":
        with ZipFile(path) as archive:
            files = [archive.read(prefix + name) for name in filenames]
    else:
        with open_tar(path, "r:*") as archive:
            files = [archive.extractfile(prefix + name).read() for name in filenames]
    with np.load(BytesIO(files[3]), allow_pickle=False) as saved:
        predictions = dict(saved)
    return json.loads(files[0]), json.loads(files[1]), [json.loads(line) for line in files[2].splitlines()], predictions


def divide(numerator, denominator):
    return numerator / np.maximum(denominator, 1)


def jaccard_rows(targets, decisions):
    return divide((targets & decisions).sum(1), (targets | decisions).sum(1))


def counts(targets, decisions):
    hits = (targets & decisions).sum(1)
    return {"tp": int(hits.sum()), "fp_unrecorded": int((decisions & ~targets).sum()),
            "fn": int((targets & ~decisions).sum()), "zero_hit": int((hits == 0).sum()),
            "empty": int((~decisions.any(1)).sum()),
            "exact": int((targets == decisions).all(1).sum()),
            "mean_predictions": float(decisions.sum(1).mean())}


def paired_interval(values, clusters):
    _, group_indices = np.unique(clusters, return_inverse=True)
    sizes = np.bincount(group_indices)
    sample = np.random.default_rng(42).integers(len(sizes), size=(2000, len(sizes)))
    boot = np.bincount(group_indices, weights=values)[sample].sum(1) / sizes[sample].sum(1)
    return np.quantile(boot, [.025, .975]).tolist()


def rank_outcomes(scores, targets):
    rank = np.argsort(-scores, axis=1, kind="stable")
    hits = np.take_along_axis(targets, rank, axis=1)
    return rank, hits


def crossfit_threshold(targets, scores, clusters):
    folds = family_folds(clusters, 5)
    predictions = np.zeros_like(targets)
    thresholds = []
    for fold in range(5):
        held = folds == fold
        threshold = fit_global_threshold(targets[~held], scores[~held])
        predictions[held] = scores[held] >= threshold
        thresholds.append(threshold)
    return predictions, thresholds


def by_count(rank, count):
    decisions = np.zeros(rank.shape, dtype=bool)
    np.put_along_axis(decisions, rank, np.arange(rank.shape[1])[None] < count[:, None], axis=1)
    return decisions


def history_table(history):
    return pd.DataFrame([{
        "epoch": row["epoch"], "train_jaccard": row["train"]["sample_jaccard"],
        "val_jaccard": row["validation"]["sample_jaccard"],
        "train_macro_ap": row["train"]["macro_ap"], "val_macro_ap": row["validation"]["macro_ap"],
        "val_micro_ap": row["validation"]["micro_ap"], "val_micro_f1": row["validation"]["micro_f1"],
        "val_precision": row["validation"]["sample_precision"], "val_coverage": row["validation"]["coverage"],
        "train_classification_loss": row["losses"]["classification"], "val_loss": row["validation_loss"],
        **{key: row["losses"].get(key) for key in ("bridge", "bridge_pairs", "bridge_active", "evidence_std")},
        "bridge_weighted": row["losses"].get("bridge", 0.) * row["loss_settings"].get("bridge", 0.),
        "lr": row["lr"], "seconds": row["seconds"], "train_seconds": row.get("train_seconds"),
        "data_wait_seconds": row.get("data_wait_seconds"), "validation_seconds": row.get("validation_seconds"),
        "amp_skipped": row["skipped_amp_updates"], "steps": row["global_step"],
    } for row in history])


def plot_history(table, output, best_epoch):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True, layout="constrained")
    for axis, columns, title in zip(
            axes.flat,
            [("train_jaccard", "val_jaccard"), ("train_macro_ap", "val_macro_ap", "val_micro_ap"),
             ("train_classification_loss", "val_loss"), ("val_precision", "val_coverage")],
            ("Sample Jaccard", "Average precision", "Observed-label classification loss",
             "Validation precision and coverage")):
        for column in columns:
            axis.plot(table.epoch, table[column], label=column.replace("_", " "), linewidth=1.6)
        axis.axvline(best_epoch, color="0.5", linestyle="--", linewidth=.8)
        axis.set_title(title, loc="left")
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def nearest_similarity(data, query, reference):
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein

    reference_sequences = np.char.upper(data.sequences[reference]).tolist()
    reference_lengths = np.array([len(seq) for seq in reference_sequences])
    result = []
    for start in range(0, len(query), 128):
        queries = np.char.upper(data.sequences[query[start:start + 128]]).tolist()
        distance = process.cdist(queries, reference_sequences, scorer=Levenshtein.distance,
                                 dtype=np.uint8, workers=4)
        lengths = np.maximum(np.array([len(seq) for seq in queries])[:, None], reference_lengths)
        result.extend(1. - (distance / lengths).min(1))
    return np.array(result)


def compare_bridge(path, data_root, output, arm="on"):
    off, on = [load_run(path, f"bridge_{variant}/") for variant in ("off", arm)]
    configs = [run[1]["config"] for run in (off, on)]
    weights = [config["loss"].get("bridge", 0.) for config in configs]
    modes = [config["loss"].get("theraset_bridge") for config in configs]
    expected_modes = [None, None] if arm == "on" else ["off", arm]
    shared = [{**config, "loss": {key: value for key, value in config["loss"].items()
                                 if key not in ("bridge", "theraset_bridge")}}
              for config in configs]
    if (shared[0] != shared[1] or modes != expected_modes or weights[0] != 0 or weights[1] <= 0
            or off[1]["git_commit"] != on[1]["git_commit"]):
        raise ValueError("The two configs must differ only in bridge mode and loss weight")
    settings = configs[0]["train"]
    fold = settings["xbridge_inner_fold"]
    data = PeptideData(data_root, "fine39")
    pool = data.indices("train")
    held = family_folds(data.clusters[pool], 5) == fold
    reference, rows = pool[~held], pool[held]
    support = data.y[reference].sum(0)
    if settings.get("limit_samples"):
        from mfp_nova.multimodal import MultimodalData, diagnostic_indices

        source = MultimodalData(data_root, "fine39", train_indices=reference)
        reference = diagnostic_indices(source, reference, settings["limit_samples"])
        rows = diagnostic_indices(source, rows, settings["limit_samples"])
    targets = data.y[rows].astype(bool)
    clusters = data.clusters[rows]
    projection = function_projection(data.labels, "fine39")
    for result, metadata, history, saved in (off, on):
        for key, expected in (("row_id", data.rows[rows]), ("ID", data.ids[rows]),
                              ("cluster_id", clusters), ("targets", targets), ("label_names", data.labels)):
            if not np.array_equal(saved[key], expected):
                raise ValueError(f"Internal Validation {key} disagrees with Train fold {fold}")
        if (saved["split"].item() != "train_inner" or float(saved["threshold"]) != .5
                or metadata["train_rows"] != len(reference) or len(history) != result["epochs_completed"]):
            raise ValueError("The pair needs matching Train-only cohorts, fixed .50 and complete logs")
    steps = [[row["global_step"] for row in run[2]] for run in (off, on)]
    amp_skipped = [sum(row["skipped_amp_updates"] for row in run[2]) for run in (off, on)]
    if steps[0] != steps[1]:
        raise ValueError("Runs used different scheduled optimizer budgets")
    similarity = nearest_similarity(data, rows, reference)
    scores = [run[3]["probabilities"] for run in (off, on)]
    predictions = [probabilities >= .5 for probabilities in scores]
    jac = [jaccard_rows(targets, pred) for pred in predictions]
    delta = jac[1] - jac[0]
    hit = [(targets & pred).any(1) for pred in predictions]
    added, removed = predictions[1] & ~predictions[0], predictions[0] & ~predictions[1]
    strata = {}
    for name, mask in (("all", np.ones(len(rows), dtype=bool)), ("below_0.5", similarity < .5),
                       ("0.5_to_0.7", (similarity >= .5) & (similarity < .7)),
                       ("below_0.7", similarity < .7), ("at_least_0.7", similarity >= .7)):
        if not mask.any():
            continue
        observed = targets[mask]
        strata[name] = {
            "rows": int(mask.sum()),
            "off": {**counts(observed, predictions[0][mask]), "jaccard": float(jac[0][mask].mean()),
                    **recovery_diagnostics(observed, scores[0][mask], projection)},
            "on": {**counts(observed, predictions[1][mask]), "jaccard": float(jac[1][mask].mean()),
                   **recovery_diagnostics(observed, scores[1][mask], projection)},
            "delta_jaccard": float(delta[mask].mean()),
            "conditional_cluster_interval": paired_interval(delta[mask], clusters[mask]),
            "rescued_zero_hit": int((~hit[0] & hit[1] & mask).sum()),
            "new_zero_hit": int((hit[0] & ~hit[1] & mask).sum()),
            "added_unrecorded": int((added[mask] & ~observed).sum()),
            "removed_unrecorded": int((removed[mask] & ~observed).sum()),
        }
    output.mkdir(parents=True, exist_ok=False)
    summary = {"scope": "Fine39 Train inner fold; intervals conditional on the selected checkpoints",
               "fold": fold, "fit_rows": len(reference), "held_rows": len(rows),
               "epochs": len(steps[0]), "scheduled_steps": steps[0][-1],
               "optimizer_steps": [steps[0][-1] - skipped for skipped in amp_skipped],
               "amp_skipped": amp_skipped,
               "bridge_weights": weights, "compared_arms": ["off", arm], "batch_size": settings["batch_size"],
               "best_epochs": [run[0]["best_epoch"] for run in (off, on)], "strata": strata}
    summary["matched_epochs"] = [
        {"epoch": a["epoch"], "off_jaccard": a["validation"]["sample_jaccard"],
         "on_jaccard": b["validation"]["sample_jaccard"],
         "delta_macro_ap": b["validation"]["macro_ap"] - a["validation"]["macro_ap"]}
        for a, b in zip(off[2], on[2], strict=True)]
    for name, run, probabilities in zip(("off", "on"), (off, on), scores):
        metrics = evaluate_metrics(targets, probabilities, .5, support, projection)
        if not np.isclose(metrics["sample_jaccard"], run[0]["validation"]["sample_jaccard"]):
            raise ValueError(f"Bridge {name} predictions do not reproduce the selected Validation result")
        summary[name] = {key: metrics[key] for key in
                         ("sample_jaccard", "sample_precision", "coverage", "macro_ap", "micro_ap",
                          "micro_f1", "mfp_detection", "mfp_function_prediction")}
        history_table(run[2]).to_csv(output / f"bridge_{name}_history.csv", index=False)
    pd.DataFrame({"row_id": data.rows[rows], "cluster_id": clusters, "similarity": similarity,
                  "off_jaccard": jac[0], "on_jaccard": jac[1],
                  "off_hit": hit[0], "on_hit": hit[1]}).to_csv(output / "validation_rows.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding="utf-8")
    print(output.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbridge", type=Path)
    parser.add_argument("--theraset", type=Path)
    parser.add_argument("--v7", type=Path)
    parser.add_argument("--nearest", type=Path)
    parser.add_argument("--bridge-pair", type=Path, help="Run directory or tar with bridge_off and bridge_on")
    parser.add_argument("--bridge-arm", choices=("on", "align", "positive"), default="on",
                        help="Experimental subdirectory name suffix for --bridge-pair")
    parser.add_argument("--data-root", type=Path, default=Path("data/therapeutic_2025"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bridge_pair:
        output = args.output or Path("runs") / datetime.now().strftime("xbridge_bridge_analysis_%Y%m%d_%H%M%S")
        compare_bridge(args.bridge_pair, args.data_root, output, args.bridge_arm)
        return
    if not (args.xbridge and args.theraset and args.nearest):
        parser.error("Provide --bridge-pair, or --xbridge, --theraset and --nearest")
    output = args.output or Path("runs") / datetime.now().strftime("xbridge_b256_analysis_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)

    runs = {"XBridge": load_run(args.xbridge), "TheraSet": load_run(args.theraset, "baseline/")}
    if args.v7:
        runs.update({f"V7 {head}": load_run(args.v7, f"{head}/") for head in ("flat", "structured")})
    data = PeptideData(args.data_root, "fine39")
    rows = data.indices("val")
    clusters = data.clusters[rows]
    labels = list(data.labels)
    targets = data.y[rows].astype(bool)
    support = data.y[data.indices("train")].sum(0)
    projection = np.asarray(function_projection(labels, "fine39"))
    truth_axes = independent_functions(targets, projection)
    truth_count = targets.sum(1)
    lengths = np.array([len(seq) for seq in data.sequences[rows]])
    with np.load(args.nearest, allow_pickle=False) as nearest:
        mapping = dict(zip(nearest["row_id"], nearest["similarity"], strict=True))
        similarity = np.array([mapping[row] for row in data.rows[rows]])
    reports, decisions, ranked_hits = {}, {}, {}
    for name, (result, metadata, history, saved) in runs.items():
        for key, expected in (("row_id", data.rows[rows]), ("ID", data.ids[rows]),
                              ("cluster_id", clusters), ("targets", targets), ("label_names", labels)):
            if not np.array_equal(saved[key], expected):
                raise ValueError(f"{name} Validation {key} differs from the current Fine39 cohort")
        if saved["split"].item() != "val" or float(saved["threshold"]) != .5:
            raise ValueError(f"{name} prediction split or threshold differs")
        scores = saved["probabilities"]
        pred = decisions[name] = scores >= .5
        rank, hits = rank_outcomes(scores, targets)
        ranked_hits[name] = hits
        metrics = evaluate_metrics(targets, scores, .5, support, projection)
        if not np.isclose(metrics["sample_jaccard"], result["validation"]["sample_jaccard"]):
            raise ValueError(f"{name} exported predictions do not reproduce the result")
        reports[name] = {key: metrics[key] for key in (
            "sample_jaccard", "sample_precision", "sample_recall", "macro_ap", "micro_ap", "micro_f1",
            "exact_match", "head_macro_ap", "medium_macro_ap", "tail_macro_ap", "mfp_detection",
            "mfp_function_prediction")}
        reports[name].update(counts(targets, pred))
        reports[name]["recovery"] = recovery_diagnostics(targets, scores, projection)
        reports[name]["family_jaccard"] = float(jaccard_rows(truth_axes, independent_functions(pred, projection)).mean())
        reports[name]["raw_family_jaccard"] = float(jaccard_rows(targets @ projection > 0, pred @ projection > 0).mean())
        reports[name]["second_hit_top5"] = float((hits[truth_count >= 2, :5].sum(1) >= 2).mean())
        reports[name]["third_hit_top5"] = float((hits[truth_count >= 3, :5].sum(1) >= 3).mean())
        reports[name]["run"] = {key: result.get(key) for key in
                                 ("best_epoch", "epochs_completed", "elapsed_seconds", "peak_cuda_memory_bytes",
                                  "global_step", "trainable_parameters")}
        reports[name]["commit"] = metadata["git_commit"]
        reports[name]["gpu"] = metadata["gpu"]
        reports[name]["precision"] = metadata["config"]["train"]["precision"]
        reports[name]["packages"] = metadata["packages"]
        if name in ("XBridge", "TheraSet"):
            table = history_table(history)
            table.to_csv(output / f"{name.lower()}_history.csv", index=False)
            threshold = fit_global_threshold(targets, scores)
            crossfit, thresholds = crossfit_threshold(targets, scores, clusters)
            fallback = pred | ((~pred.any(1))[:, None] & by_count(rank, np.ones(len(rank), dtype=int)))
            variants = {
                "fixed": pred, "same_val_optimized": scores >= threshold,
                "crossfit": crossfit, "empty_top1": fallback,
                "true_count_topk": by_count(rank, truth_count),
                "theraset_count_topk": by_count(rank, (runs["TheraSet"][3]["probabilities"] >= .5).sum(1)),
            }
            reports[name]["decision_diagnostics"] = {
                "same_val_threshold": threshold, "crossfit_thresholds": thresholds,
                "variants": {key: {**counts(targets, variant),
                                   "jaccard": float(jaccard_rows(targets, variant).mean())}
                             for key, variant in variants.items()}}
            if name == "XBridge":
                plot_history(table, output / "training_curves.png", result["best_epoch"])

    xbridge, baseline = decisions["XBridge"], decisions["TheraSet"]
    x_jaccard = jaccard_rows(targets, xbridge)
    b_jaccard = jaccard_rows(targets, baseline)
    difference = x_jaccard - b_jaccard
    x_hit = (targets & xbridge).sum(1) > 0
    b_hit = (targets & baseline).sum(1) > 0
    x_top5 = ranked_hits["XBridge"][:, :5].any(1)
    b_top5 = ranked_hits["TheraSet"][:, :5].any(1)
    transitions = {
        "rescued_baseline_zero_hit": int((~b_hit & x_hit).sum()),
        "new_zero_hit": int((b_hit & ~x_hit).sum()),
        "both_zero_hit": int((~b_hit & ~x_hit).sum()),
        "both_hit": int((b_hit & x_hit).sum()),
        "top5_hit_only_xbridge": int((~b_top5 & x_top5).sum()),
        "top5_hit_only_theraset": int((b_top5 & ~x_top5).sum()),
        "top5_rescues_baseline_zero_hit": int((~b_hit & x_top5).sum()),
        "top5_rescues_baseline_zero_hit_and_outside_baseline_top5": int((~b_hit & ~b_top5 & x_top5).sum()),
        "rows_better_tied_worse": [int((difference > 0).sum()), int((difference == 0).sum()), int((difference < 0).sum())],
    }
    x_positive, b_positive = xbridge & targets, baseline & targets
    x_false, b_false = xbridge & ~targets, baseline & ~targets
    replace_fp = jaccard_rows(targets, x_positive | b_false)
    replace_tp = jaccard_rows(targets, b_positive | x_false)
    paired = {
        "delta_jaccard": float(difference.mean()),
        "cluster_resample_interval": paired_interval(difference, clusters),
        "new_tp": int((x_positive & ~b_positive).sum()),
        "lost_tp": int((b_positive & ~x_positive).sum()),
        "new_fp_unrecorded": int((x_false & ~b_false).sum()),
        "removed_fp_unrecorded": int((b_false & ~x_false).sum()),
        "delta_from_tp_two_order_average": float(((replace_fp - b_jaccard + x_jaccard - replace_tp) / 2).mean()),
        "delta_from_fp_two_order_average": float(((replace_tp - b_jaccard + x_jaccard - replace_fp) / 2).mean()),
        "transitions": transitions,
    }
    strata = {}
    for name, mask in (
            ("similarity_<0.5", similarity < .5), ("similarity_0.5_0.7", (similarity >= .5) & (similarity < .7)),
            ("similarity_0.7_0.8", (similarity >= .7) & (similarity < .8)), ("similarity_0.8_0.9", similarity >= .8),
            ("similarity_<0.7", similarity < .7), ("labels_1", truth_count == 1),
            ("labels_2", truth_count == 2), ("labels_3+", truth_count >= 3),
            ("length_2_8", lengths <= 8), ("length_9_16", (lengths > 8) & (lengths <= 16)),
            ("length_17_32", (lengths > 16) & (lengths <= 32)), ("length_33+", lengths > 32),
            ("both_hit", x_hit & b_hit), ("only_xbridge_hit", x_hit & ~b_hit), ("only_theraset_hit", b_hit & ~x_hit)):
        strata[name] = {"rows": int(mask.sum()), "xbridge_j": float(x_jaccard[mask].mean()),
                        "theraset_j": float(b_jaccard[mask].mean()), "delta_j": float(difference[mask].mean()),
                        "contribution": float(difference[mask].sum() / len(targets)),
                        "xbridge_zero": int((mask & ~x_hit).sum()), "theraset_zero": int((mask & ~b_hit).sum()),
                        "xbridge_top1": float(ranked_hits["XBridge"][mask, 0].mean()),
                        "theraset_top1": float(ranked_hits["TheraSet"][mask, 0].mean()),
                        "xbridge_top5": float(x_top5[mask].mean()), "theraset_top5": float(b_top5[mask].mean())}
    label_table = pd.DataFrame({"label": labels, "train_support": support, "val_support": targets.sum(0)})
    for name, prefix in (("XBridge", "x"), ("TheraSet", "b")):
        pred = decisions[name]
        label_table[f"{prefix}_ap"] = runs[name][0]["validation"]["label_ap"]
        label_table[f"{prefix}_tp"] = (pred & targets).sum(0)
        label_table[f"{prefix}_fn"] = (~pred & targets).sum(0)
        label_table[f"{prefix}_fp"] = (pred & ~targets).sum(0)
    label_table["extra_fp"] = label_table.x_fp - label_table.b_fp
    label_table["delta_ap"] = label_table.x_ap - label_table.b_ap
    label_table.to_csv(output / "labels.csv", index=False)
    pd.DataFrame.from_dict(strata, orient="index").rename_axis("stratum").to_csv(output / "strata.csv")
    pd.DataFrame({"row_id": data.rows[rows], "cluster_id": clusters, "similarity": similarity,
                  "xbridge_jaccard": x_jaccard, "theraset_jaccard": b_jaccard,
                  "xbridge_hit": x_hit, "theraset_hit": b_hit,
                  "xbridge_top5_hit": x_top5, "theraset_top5_hit": b_top5}).to_csv(output / "validation_rows.csv", index=False)
    (output / "summary.json").write_text(json.dumps({"models": reports, "paired": paired, "strata": strata},
                                                    ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
