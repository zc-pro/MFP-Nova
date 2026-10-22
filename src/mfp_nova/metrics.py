import numpy as np
from sklearn.metrics import average_precision_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score


def evaluate_metrics(targets, probabilities, threshold=0.5, train_support=None, function_projection=None, predicted=None):
    active = targets.sum(0) > 0
    ap = [float(average_precision_score(targets[:, c], probabilities[:, c])) if active[c] else None
          for c in range(targets.shape[1])]
    if predicted is None:
        predicted = probabilities >= threshold
    else:
        predicted = np.asarray(predicted, dtype=bool)
        if predicted.shape != targets.shape:
            raise ValueError("Decision labels must have the target shape")
    result = {
        "macro_ap": float(np.mean([v for v in ap if v is not None])) if active.any() else None,
        "micro_ap": float(average_precision_score(targets.ravel(), probabilities.ravel())) if active.any() else None,
        "label_ap": ap,
        "label_positive_support": targets.sum(0).astype(int).tolist(),
        "ap_labels_with_positives": int(active.sum()),
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predicted, average="micro", zero_division=0)),
        "micro_precision": float(precision_score(targets, predicted, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(targets, predicted, average="micro", zero_division=0)),
        "sample_jaccard": float(jaccard_score(targets, predicted, average="samples", zero_division=0)),
    }
    actual = targets.astype(bool)
    tp = (actual & predicted).sum(0).astype(float)
    fp = (~actual & predicted).sum(0).astype(float)
    fn = (actual & ~predicted).sum(0).astype(float)
    tn = (~actual & ~predicted).sum(0).astype(float)
    ratio = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b!=0)
    precision, recall = ratio(tp, tp+fp), ratio(tp, tp+fn)
    label_f1 = ratio(2*tp, 2*tp+fp+fn)
    denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    auc = [float(roc_auc_score(actual[:, c], probabilities[:, c])) if actual[:, c].any() and not actual[:, c].all() else None
           for c in range(targets.shape[1])]
    result.update({
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "sample_precision": float(ratio((actual & predicted).sum(1), predicted.sum(1)).mean()),
        "sample_recall": float(ratio((actual & predicted).sum(1), actual.sum(1)).mean()),
        "exact_match": float((actual==predicted).all(1).mean()),
        "hamming_loss": float((actual!=predicted).mean()),
        "label_f1": label_f1.tolist(), "label_precision": precision.tolist(), "label_recall": recall.tolist(),
        "label_sensitivity": [float(recall[i]) if tp[i]+fn[i] else None for i in range(len(tp))],
        "label_specificity": [float(tn[i]/(tn[i]+fp[i])) if tn[i]+fp[i] else None for i in range(len(tp))],
        "label_mcc": [float((tp[i]*tn[i]-fp[i]*fn[i])/denominator[i]) if denominator[i] else None for i in range(len(tp))],
        "label_auroc": auc,
        "macro_auroc": float(np.mean([a for a in auc if a is not None])) if any(a is not None for a in auc) else None,
        "micro_auroc": float(roc_auc_score(actual.ravel(), probabilities.ravel())) if actual.any() and not actual.all() else None,
    })
    result.update({
        "accuracy": result["sample_jaccard"], "precision": result["sample_precision"],
        "coverage": result["sample_recall"], "absolute_true": result["exact_match"],
        "absolute_false": result["hamming_loss"],
        "sample_f1": float(ratio(2*(actual & predicted).sum(1), actual.sum(1)+predicted.sum(1)).mean()),
    })
    for name, values in (("macro_mcc", result["label_mcc"]),
                         ("macro_sensitivity", result["label_sensitivity"]),
                         ("macro_specificity", result["label_specificity"])):
        valid = [v for v in values if v is not None]
        result[name] = float(np.mean(valid)) if valid else None
    t, f, n, z = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    micro_denominator = np.sqrt((t+f)*(t+n)*(z+f)*(z+n))
    result["micro_mcc"] = float((t*z-f*n)/micro_denominator) if micro_denominator else None
    result["micro_sensitivity"] = float(t/(t+n)) if t+n else None
    result["micro_specificity"] = float(z/(z+f)) if z+f else None
    # Keep the existing sklearn zero_division=0 convention, and expose why a
    # denominator or ranking statistic is unavailable rather than inventing it.
    undefined = {
        "label_ap": {str(i): "no_positive_targets" for i in range(len(tp)) if not active[i]},
        "label_auroc": {str(i): "requires_positive_and_negative_targets" for i, a in enumerate(auc) if a is None},
        "label_mcc": {str(i): "zero_mcc_denominator" for i, d in enumerate(denominator) if not d},
        "label_sensitivity": {str(i): "no_positive_targets" for i in range(len(tp)) if not tp[i]+fn[i]},
        "label_specificity": {str(i): "no_negative_targets" for i in range(len(tp)) if not tn[i]+fp[i]},
        "label_precision": {str(i): "no_positive_predictions; zero_division=0" for i in range(len(tp)) if not tp[i]+fp[i]},
        "label_recall": {str(i): "no_positive_targets; zero_division=0" for i in range(len(tp)) if not tp[i]+fn[i]},
        "label_f1": {str(i): "no_positive_targets_or_predictions; zero_division=0" for i in range(len(tp)) if not 2*tp[i]+fp[i]+fn[i]},
    }
    for name, value in result.items():
        if value is None:
            undefined[name] = "no_valid_labels" if name.startswith("macro_") else "missing_class_or_zero_denominator"
    if train_support is not None:
        support = np.asarray(train_support)
        for name, mask in [("head", support>=1000), ("medium", (support>=200)&(support<1000)), ("tail", support<200)]:
            values = [a for a, selected in zip(ap, mask) if selected and a is not None]
            result[f"{name}_macro_ap"] = float(np.mean(values)) if values else None
            result[f"{name}_ap_labels"] = len(values)
            if not values:
                undefined[f"{name}_macro_ap"] = "no_labels_in_train_stratum" if not mask.any() else "no_positive_targets_in_stratum"
    if function_projection is not None:
        from .multimodal import independent_functions
        truth = independent_functions(actual, function_projection)
        guessed = independent_functions(predicted, function_projection)
        mfp_true, mfp_pred = truth.sum(1)>=2, guessed.sum(1)>=2
        result["mfp_detection"] = {
            "precision": float(precision_score(mfp_true, mfp_pred, zero_division=0)),
            "recall": float(recall_score(mfp_true, mfp_pred, zero_division=0)),
            "f1": float(f1_score(mfp_true, mfp_pred, zero_division=0)),
            "support": int(mfp_true.sum()), "predicted": int(mfp_pred.sum()),
            "true_positive": int((mfp_true & mfp_pred).sum()),
            "false_positive": int((~mfp_true & mfp_pred).sum()),
            "false_negative": int((mfp_true & ~mfp_pred).sum())}
        result["mfp_function_prediction"] = {
            "precision": float(precision_score(truth[mfp_true], guessed[mfp_true], average="samples", zero_division=0)),
            "recall": float(recall_score(truth[mfp_true], guessed[mfp_true], average="samples", zero_division=0)),
            "f1": float(f1_score(truth[mfp_true], guessed[mfp_true], average="samples", zero_division=0))} if mfp_true.any() else None
        if not mfp_true.any():
            undefined["mfp_function_prediction"] = "no_multifunction_targets"
            undefined["mfp_detection.recall"] = "no_multifunction_targets; zero_division=0"
        if not mfp_pred.any():
            undefined["mfp_detection.precision"] = "no_multifunction_predictions; zero_division=0"
    result["undefined_metrics"] = {k: v for k, v in undefined.items() if v}
    result["zero_division"] = 0
    return result


def evaluate_groups(targets, probabilities, groups, threshold, train_support, function_projection, predicted=None):
    """CPU summaries of the same predictions; no additional model forward."""
    output = {}
    for name, mask in groups.items():
        rows = int(mask.sum())
        if not rows:
            output[name] = {"rows": 0, "metrics": None, "reason": "no_samples"}
            continue
        score = evaluate_metrics(targets[mask], probabilities[mask], threshold, train_support, function_projection,
                                 None if predicted is None else predicted[mask])
        # Global per-label arrays live once in history; group summaries retain
        # scalars, MFP counts, and explicit undefined-statistic explanations.
        score = {k: v for k, v in score.items() if not isinstance(v, list)}
        output[name] = {"rows": rows, "metrics": score}
    return output


def format_epoch(model, task, seed, maximum_epochs, record, patience):
    """Separated epoch console block; JSONL remains the complete metric record."""
    def number(value):
        return "NA" if value is None else f"{value:.4f}"

    def line(name, report, loss):
        if "metric_suite" in record:
            from .metric_protocol import format_primary
            return f"{name:5} | Loss={number(loss)} | {format_primary(record['metric_suite'], report)}"
        fields = [("Acc", "accuracy"), ("Prec", "precision"), ("Cov", "coverage"),
                  ("AT", "absolute_true"), ("AF", "absolute_false"),
                  ("MacroAP", "macro_ap"), ("MicroAP", "micro_ap"),
                  ("MacroF1", "macro_f1"), ("MicroF1", "micro_f1")]
        text = " ".join(f"{label}={number(report[key])}" for label, key in fields)
        return f"{name:5} | Loss={number(loss)} | {text} MFP-F1={number(report.get('mfp_detection', {}).get('f1'))}"

    validation = record["validation"]
    peak = record["peak_cuda_memory_bytes"]
    memory = "NA" if peak is None else f"{peak/2**30:.2f}GiB"
    best_name = "SelectedEpochMacroAP" if record.get("metric_suite") == "traditional5" else "Best"
    return "\n".join([
        "",
        "="*96,
        f"Epoch {record['epoch']}/{maximum_epochs} | Model={model} Task={task} Seed={seed} | MonitorThreshold={record['monitor_threshold']:.2f}",
        line("Train", record["train"], record["losses"]["total"]),
        line("Val", validation, record["validation_loss"]),
        f"Extra | HeadAP={number(validation.get('head_macro_ap'))} MediumAP={number(validation.get('medium_macro_ap'))} "
        f"TailAP={number(validation.get('tail_macro_ap'))} | LR={record['lr']:.3g} Time={record['seconds']:.1f}s "
        f"Elapsed={record['elapsed_seconds']/60:.1f}min Peak={memory} "
        f"{best_name}={number(record['best_macro_ap'])}@{record['best_epoch']} Patience={record['early_stopping_count']}/{patience} "
        f"AMP-skipped={record['skipped_amp_updates']}",
    ])


def fit_label_thresholds(targets, probabilities):
    """Optional Validation experiment; never invoked on Test."""
    global_threshold = fit_global_threshold(targets, probabilities)
    grid = np.round(np.arange(.05, .96, .01), 2)
    return [float(max(grid, key=lambda t: (f1_score(targets[:, c], probabilities[:, c]>=t, zero_division=0),
                                           -abs(t-global_threshold), -t))) if targets[:, c].any() else global_threshold
            for c in range(targets.shape[1])]


def fit_global_threshold(targets, probabilities):
    """Call on Validation only. Maximize sample Jaccard; ties prefer near 0.5."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    return float(max(grid, key=lambda t: (
        jaccard_score(targets, probabilities >= t, average="samples", zero_division=0),
        -abs(t - 0.5), -t)))
