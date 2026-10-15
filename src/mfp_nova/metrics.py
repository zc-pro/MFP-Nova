import numpy as np
from sklearn.metrics import average_precision_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score


def evaluate_metrics(targets, probabilities, threshold=0.5, train_support=None, function_projection=None):
    active = targets.sum(0) > 0
    ap = [float(average_precision_score(targets[:, c], probabilities[:, c])) if active[c] else None
          for c in range(targets.shape[1])]
    predicted = probabilities >= threshold
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
    if train_support is not None:
        support = np.asarray(train_support)
        for name, mask in [("head", support>=1000), ("medium", (support>=200)&(support<1000)), ("tail", support<200)]:
            values = [a for a, selected in zip(ap, mask) if selected and a is not None]
            result[f"{name}_macro_ap"] = float(np.mean(values)) if values else None
            result[f"{name}_ap_labels"] = len(values)
    if function_projection is not None:
        from .multimodal import independent_functions
        truth = independent_functions(actual, function_projection)
        guessed = independent_functions(predicted, function_projection)
        mfp_true, mfp_pred = truth.sum(1)>=2, guessed.sum(1)>=2
        result["mfp_detection"] = {
            "precision": float(precision_score(mfp_true, mfp_pred, zero_division=0)),
            "recall": float(recall_score(mfp_true, mfp_pred, zero_division=0)),
            "f1": float(f1_score(mfp_true, mfp_pred, zero_division=0)),
            "support": int(mfp_true.sum()), "predicted": int(mfp_pred.sum())}
        result["mfp_function_prediction"] = {
            "precision": float(precision_score(truth[mfp_true], guessed[mfp_true], average="samples", zero_division=0)),
            "recall": float(recall_score(truth[mfp_true], guessed[mfp_true], average="samples", zero_division=0)),
            "f1": float(f1_score(truth[mfp_true], guessed[mfp_true], average="samples", zero_division=0))} if mfp_true.any() else None
    return result


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
