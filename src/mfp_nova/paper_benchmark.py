"""Data and evaluation shared by the clustered paper baselines."""
from pathlib import Path

import numpy as np
import torch

from .data import PeptideData
from .metric_protocol import format_primary, primary_metrics, protocol_description, selection
from .metrics import evaluate_metrics
from .multimodal import function_projection
from .therapep import ROOTS as THERAPEP_ROOTS, read_data as read_therapep


CLUSTERED = ("major15", "fine39")
ROOTS = {**THERAPEP_ROOTS, **dict.fromkeys(CLUSTERED, "data/therapeutic_2025")}
TRACKS = tuple(ROOTS)
SPLITS = ("train", "val", "test")


def read_data(root, task):
    if task not in CLUSTERED:
        return read_therapep(root, task)
    data = PeptideData(root, task)
    indices = [data.indices(split) for split in SPLITS]
    rows = np.concatenate(indices)
    # The authors' sequence alphabets do not distinguish lowercase D residues.
    return {"rows": data.rows[rows], "ids": data.ids[rows], "labels": data.labels,
            "sequences": np.char.upper(data.sequences[rows]), "y": data.y[rows],
            "splits": np.repeat(SPLITS, [len(index) for index in indices])}


def traditional5_selection(targets, probabilities):
    return selection(targets, probabilities, None, "traditional5", "global")


class BestTraditional5:
    def __init__(self, path):
        self.path = Path(path)
        self.key = None
        self.epoch = None
        self.threshold = None

    def update(self, epoch, model, targets, probabilities):
        key, threshold, metrics = traditional5_selection(targets, probabilities)
        print(f"Val epoch={epoch} threshold={threshold:.2f} "
              f"{format_primary('traditional5', metrics)}", flush=True)
        if self.key is not None and key <= self.key:
            return False
        torch.save(model.state_dict(), self.path)
        self.key, self.epoch, self.threshold = key, epoch, float(threshold)
        return True

    def restore(self, model):
        model.load_state_dict(torch.load(self.path, map_location="cpu"))


def traditional5_report(targets, probabilities, threshold, epoch, task, labels, train_support):
    metrics = evaluate_metrics(targets, probabilities, threshold=threshold, train_support=train_support,
                               function_projection=function_projection(labels, task))
    primary = primary_metrics(metrics, "traditional5")
    print(f"Test epoch={epoch} threshold={threshold:.2f} "
          f"{format_primary('traditional5', primary)}", flush=True)
    return {"best_epoch": epoch, "threshold": float(threshold),
            "metric_protocol": protocol_description("traditional5", "global"),
            "traditional5": primary, "full_test": metrics}


def positive_scope(task, targets, probabilities, predicted):
    if task in CLUSTERED:
        return {}
    positive = targets[:, :48].any(1)
    return {"positive_test_rows": int(positive.sum()),
            "positive_function48": evaluate_metrics(targets[positive, :48], probabilities[positive, :48],
                                                     predicted=predicted[positive, :48]),
            "background_any_function_fpr": (float(predicted[~positive, :48].any(1).mean())
                                            if (~positive).any() else None)}
