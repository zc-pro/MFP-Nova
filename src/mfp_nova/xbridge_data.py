"""Train-only, cluster-balanced positive pairing for XBridge."""

import math

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class BridgePeptides(Dataset):
    def __init__(self, dataset, clusters):
        self.dataset = dataset
        self.clusters = torch.from_numpy(np.unique(clusters, return_inverse=True)[1])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "xbridge_cluster": self.clusters[index]}


class PositivePairBatches(Sampler):
    """Add a positive pair to each shuffled batch: label, cluster, then row."""

    def __init__(self, targets, clusters, batch_size, generator):
        if batch_size < 3:
            raise ValueError("Pair sampling needs a batch size of at least three")
        self.size = len(targets)
        self.generator = generator
        self.pools = []
        for positive in targets.T:
            groups = {}
            for index in np.flatnonzero(positive):
                groups.setdefault(clusters[index], []).append(int(index))
            if len(groups) > 1:
                self.pools.append(list(groups.values()))
        self.stride = batch_size - 2 if self.pools else batch_size

    def __len__(self):
        return math.ceil(self.size / self.stride)

    def __iter__(self):
        seed = int(torch.randint(2**32, (), generator=self.generator))
        rng = np.random.default_rng(seed)
        rows = rng.permutation(self.size)
        for start in range(0, self.size, self.stride):
            batch = rows[start:start + self.stride].tolist()
            if self.pools:
                groups = self.pools[rng.integers(len(self.pools))]
                for group in rng.choice(len(groups), size=2, replace=False):
                    index = int(rng.choice(groups[group]))
                    if index not in batch:
                        batch.append(index)
            yield batch


def recovery_diagnostics(targets, scores, projection):
    from .multimodal import independent_functions

    observed = targets.astype(bool)
    hit = observed & (scores >= .5)
    axes = independent_functions(observed, projection)
    recovered = axes & ((hit @ np.asarray(projection)) > 0)
    counts = axes.sum(1)
    recovered_count = recovered.sum(1)
    ranked = np.argsort(-scores, axis=1, kind="stable")[:, :5]
    ranked_hits = np.take_along_axis(observed, ranked, axis=1)
    result = {"zero_hit_rows": int((~hit.any(1)).sum()), "rows": len(hit),
              "top1_hit_rows": int(ranked_hits[:, 0].sum()), "top5_hit_rows": int(ranked_hits.any(1).sum())}
    for number in (2, 3):
        eligible = counts >= number
        result[f"at_least_{number}_axes"] = {
            "rows": int(eligible.sum()),
            "recovered_at_least": int((eligible & (recovered_count >= number)).sum()),
            "all_recovered": int((eligible & (recovered_count == counts)).sum()),
        }
    return result


def evidence_report(targets, probabilities, outputs, statistics):
    from .metrics import evaluate_metrics

    projection = statistics["function_projection"]
    report = {"final": recovery_diagnostics(targets, probabilities, projection)}
    for key, logits in outputs.items():
        scores = torch.from_numpy(logits).sigmoid().numpy()
        report[key] = {**evaluate_metrics(targets, scores, .5, statistics["labels"]["support"], projection),
                       **recovery_diagnostics(targets, scores, projection)}
    if "reference_logits" in outputs:
        reference = outputs["reference_logits"] >= 0
        predicted = probabilities >= .5
        targets = targets.astype(bool)
        before, after = (targets & reference).any(1), (targets & predicted).any(1)
        added, dropped = predicted & ~reference, reference & ~predicted
        report["vs_reference"] = {
            "rescued_zero_hit_rows": int((~before & after).sum()),
            "new_zero_hit_rows": int((before & ~after).sum()),
            "added_recorded": int((added & targets).sum()), "dropped_recorded": int((dropped & targets).sum()),
            "added_unrecorded": int((added & ~targets).sum()), "dropped_unrecorded": int((dropped & ~targets).sum()),
        }
    return report
