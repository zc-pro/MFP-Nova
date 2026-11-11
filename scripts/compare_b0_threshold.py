"""Apply the traditional5 Jaccard threshold to an existing B0 Validation export."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import jaccard_score

from mfp_nova.metrics import fit_global_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="B0 run directory containing validation_predictions.npz")
    args = parser.parse_args()

    with np.load(args.run / "validation_predictions.npz", allow_pickle=False) as validation:
        targets = validation["project_targets"]
        probabilities = validation["decision_probabilities"]
        saved = validation["predicted"]
    threshold = fit_global_threshold(targets, probabilities)
    predicted = probabilities >= threshold
    saved_jaccard = float(jaccard_score(targets, saved, average="samples", zero_division=0))
    refit_jaccard = float(jaccard_score(targets, predicted, average="samples", zero_division=0))
    print(json.dumps({"run": str(args.run), "split": "validation", "rows": len(targets),
                      "saved_jaccard": saved_jaccard, "traditional5_threshold": threshold,
                      "traditional5_jaccard": refit_jaccard,
                      "delta_jaccard_pp": 100 * (refit_jaccard - saved_jaccard),
                      "changed_prediction_rows": int((predicted != saved).any(1).sum())}, indent=2))


if __name__ == "__main__":
    main()
