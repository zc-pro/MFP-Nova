import numpy as np
from sklearn.metrics import average_precision_score, f1_score, jaccard_score, precision_score, recall_score


def evaluate_metrics(targets, probabilities, threshold=0.5):
    active = targets.sum(0) > 0
    ap = [float(average_precision_score(targets[:, c], probabilities[:, c])) if active[c] else None
          for c in range(targets.shape[1])]
    predicted = probabilities >= threshold
    return {
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


def fit_global_threshold(targets, probabilities):
    """Call on Validation only. Maximize sample Jaccard; ties prefer near 0.5."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    return float(max(grid, key=lambda t: (
        jaccard_score(targets, probabilities >= t, average="samples", zero_division=0),
        -abs(t - 0.5), -t)))
