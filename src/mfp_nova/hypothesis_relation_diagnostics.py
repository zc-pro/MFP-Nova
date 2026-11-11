"""Held-out diagnostics for cross-background and pairwise predictions."""

import numpy as np

from .models.hypothesis_relations import matched_pairs


def _four_states(labels, y, observed, groups, predicted, pairs, probabilities=None):
    states = ("00", "10", "01", "11")
    whole_empty = ~predicted.any(1)
    report = []
    for index, (a, b) in enumerate(pairs):
        both_known = observed[:, a] & observed[:, b]
        truth = y[:, a].astype(int) + 2 * y[:, b]
        decisions = predicted[:, a].astype(int) + 2 * predicted[:, b]
        counts = {}
        for state, name in enumerate(states):
            rows = both_known & (truth == state)
            n = int(rows.sum())
            predicted_states = np.bincount(decisions[rows], minlength=4)
            counts[name] = {"rows": n, "families": int(len(np.unique(groups[rows]))),
                            "predicted_states": dict(zip(states, predicted_states.tolist())),
                            "whole_prediction_empty": float(whole_empty[rows].mean()) if n else None,
                            "predicted_a_rate": float(predicted[rows, a].mean()) if n else None,
                            "predicted_b_rate": float(predicted[rows, b].mean()) if n else None,
                            "predicted_both_rate": float(predicted_states[3] / n) if n else None}
            if probabilities is not None:
                p = probabilities[rows, index, state]
                counts[name]["nll"] = float(-np.log(p.clip(min=1e-30)).mean()) if n else None
        report.append({"labels": [labels[a], labels[b]], "states": counts})
    return report


def _ranking_pairs(model, data, rows, ranking):
    backgrounds = model.assets.get("ranking_backgrounds", [])
    targets, observed = data.targets[rows], data.observed[rows]
    groups = data.clusters[rows]
    pairs = matched_pairs(targets, observed, groups, data.labels, data.lengths[rows], data.sources[rows],
                          data.ids[rows], backgrounds=backgrounds)
    positive, negative, label = pairs.T
    margin = ranking[positive, label] - ranking[negative, label]
    per_label = []
    for c, name in enumerate(data.labels):
        eligible = int((targets[:, c].astype(bool) & observed[:, c]).sum())
        selected = label == c
        n = int(selected.sum())
        per_label.append({"label": name, "eligible_positive_rows": eligible, "pairs": n,
                          "positive_families": int(len(np.unique(groups[positive[selected]]))),
                          "negative_families": int(len(np.unique(groups[negative[selected]]))),
                          "wrong_order_rate": float((margin[selected] < 0).mean()) if n else None,
                          "tie_rate": float((margin[selected] == 0).mean()) if n else None})
    return {"pairs": len(pairs), "wrong_order_rate": float((margin < 0).mean()) if len(pairs) else None,
            "tie_rate": float((margin == 0).mean()) if len(pairs) else None, "by_label": per_label}


def relation_diagnostics(model, data, rows, ranking, predicted, pair_probabilities=None):
    if model.name == "h4":
        return {"matched_ranking": _ranking_pairs(model, data, rows, np.asarray(ranking))}
    if model.name == "h1":
        pairs = [(companion, c) for c, companion in enumerate(model.assets.get("companions", [])) if companion >= 0]
    elif model.name == "h2":
        pairs = model.assets.get("edges", [])
    elif model.name == "h5":
        pairs = model.assets.get("pairs", [])
    else:
        return {}
    predicted = np.asarray(predicted, dtype=bool)
    probabilities = None if pair_probabilities is None else np.asarray(pair_probabilities)
    return {"four_state_diagnostics": _four_states(data.labels, data.targets[rows], data.observed[rows],
                                                    data.clusters[rows], predicted, pairs, probabilities)}
