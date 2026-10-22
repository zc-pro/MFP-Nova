"""Explicit evaluation/selection profiles; all thresholds are fitted on Val only."""
import numpy as np

from .metrics import fit_global_threshold, fit_label_thresholds


SUITES = ("traditional5", "task_specific")
TRADITIONAL = ("accuracy", "precision", "coverage", "absolute_true", "absolute_false")
TASK_SPECIFIC = ("macro_ap", "micro_ap", "macro_f1", "micro_f1", "sample_jaccard",
                 "head_macro_ap", "medium_macro_ap", "tail_macro_ap", "mfp_detection",
                 "mfp_function_prediction")


def primary_metrics(report, suite):
    fields = TRADITIONAL if suite == "traditional5" else TASK_SPECIFIC
    return {name: report[name] for name in fields if name in report}


def protocol_description(suite, strategy):
    if suite not in SUITES:
        raise ValueError(f"metric_suite must be one of {SUITES}")
    return {
        "metric_suite": suite,
        "selection": (["val_fitted_accuracy", "val_fitted_absolute_true", "earlier_epoch"]
                      if suite == "traditional5" else ["val_macro_ap", "val_micro_ap", "earlier_epoch"]),
        "threshold_strategy": strategy,
        "threshold_objective": "sample_jaccard" if strategy == "global" else "per_label_f1",
        "threshold_fit_split": "val",
        "selection_threshold": "fitted_each_epoch" if suite == "traditional5" else "not_applicable_to_AP",
        "monitor_threshold": .5,
        "absolute_false_direction": "minimize",
    }


def selection(y, probabilities, report, suite, strategy):
    """Reuse the epoch's predictions; never run another neural forward pass."""
    if suite == "task_specific":
        return (report["macro_ap"], report["micro_ap"]), None, primary_metrics(report, suite)
    threshold = (fit_label_thresholds(y, probabilities) if strategy == "per_label"
                 else fit_global_threshold(y, probabilities))
    actual, predicted = y.astype(bool), probabilities >= threshold
    intersection = (actual & predicted).sum(1)
    union = (actual | predicted).sum(1)
    ratio = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    fitted = {
        "accuracy": float(ratio(intersection, union).mean()),
        "precision": float(ratio(intersection, predicted.sum(1)).mean()),
        "coverage": float(ratio(intersection, actual.sum(1)).mean()),
        "absolute_true": float((actual == predicted).all(1).mean()),
        "absolute_false": float((actual != predicted).mean()),
    }
    return (fitted["accuracy"], fitted["absolute_true"]), threshold, fitted


def format_primary(suite, report):
    def value(number):
        return "NA" if number is None else f"{number:.4f}"
    if suite == "traditional5":
        names = zip(("Acc", "Prec", "Cov", "AT", "AF(min)"), TRADITIONAL)
    else:
        names = (("MacroAP", "macro_ap"), ("MicroAP", "micro_ap"),
                 ("MacroF1", "macro_f1"), ("MicroF1", "micro_f1"), ("Jaccard", "sample_jaccard"))
    text = " ".join(f"{label}={value(report.get(key))}" for label, key in names)
    if suite == "task_specific":
        text += f" MFP-F1={value(report.get('mfp_detection', {}).get('f1'))}"
    return text
