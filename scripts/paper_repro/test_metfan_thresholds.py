import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from scripts.paper_repro.metfan_therapep import diagnose, exact_f1_thresholds


class MetfanThresholdTests(unittest.TestCase):
    def test_exact_search_matches_distinct_scores_and_handles_constant_columns(self):
        targets = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1]])
        probabilities = np.array([[0.0001, 0.1, 0.9], [0.0003, 0.2, 0.8],
                                  [0.0002, 0.1, 0.7], [0.0003, 0.2, 0.8]])
        thresholds = exact_f1_thresholds(targets, probabilities)
        self.assertEqual(thresholds[0], 0.0003)
        np.testing.assert_array_equal(thresholds[1:], [0.5, 0.5])
        best = max(f1_score(targets[:, 0], probabilities[:, 0] >= t)
                   for t in np.unique(probabilities[:, 0]))
        self.assertEqual(f1_score(targets[:, 0], probabilities[:, 0] >= thresholds[0]), best)

    def test_equal_f1_uses_highest_threshold(self):
        targets = np.array([[1], [0], [0], [1]])
        probabilities = np.array([[0.1], [0.2], [0.3], [0.4]])
        self.assertEqual(exact_f1_thresholds(targets, probabilities)[0], 0.4)

    def test_diagnose_preserves_native_files_and_fits_only_validation(self):
        val_y = np.array([[0, 0], [1, 0], [0, 0], [1, 0]])
        val_p = np.array([[0.0001, 0.1], [0.0003, 0.2], [0.0002, 0.1], [0.0003, 0.2]])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "task.json").write_text(json.dumps({"task": "therapep48"}))
            result = output / "training/seed42/0/textcnn_esm2_mean_prot_t5_mean_optimized"
            result.mkdir(parents=True)
            for split in ("val", "test"):
                np.savez(result / f"{split}_predictions.npz", targets=val_y, probabilities=val_p,
                         thresholds=[0.0, 0.5], label_names=["rare", "absent"])
            native_path = result / "comparison.json"
            native_path.write_text('{"original": true}')
            diagnose(output, 42)
            report = json.loads((result / "threshold_analysis.json").read_text())
            strategies = report["strategies"]
            self.assertEqual(strategies["native_val_grid"]["test"]["all_positive_labels"], ["rare"])
            self.assertEqual(strategies["val_exact_f1"]["test"]["metrics"]["micro_f1"], 1.0)
            # Altering Test must not alter the fitted thresholds.
            np.savez(result / "test_predictions.npz", targets=1-val_y, probabilities=1-val_p,
                     thresholds=[0.0, 0.5], label_names=["rare", "absent"])
            diagnose(output, 42)
            second = json.loads((result / "threshold_analysis.json").read_text())
            self.assertEqual(strategies["val_exact_f1"]["thresholds"],
                             second["strategies"]["val_exact_f1"]["thresholds"])
            self.assertEqual(native_path.read_text(), '{"original": true}')
            with np.load(result / "val_predictions.npz") as archive:
                np.testing.assert_array_equal(archive["thresholds"], [0.0, 0.5])


if __name__ == "__main__":
    unittest.main()
