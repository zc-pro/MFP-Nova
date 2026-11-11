"""Calibrate on held-out families and score raw-sequence hypothesis models."""

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .hypothesis_relation_diagnostics import relation_diagnostics
from .metrics import evaluate_metrics
from .models.hypothesis_backbone import pad_tokens, tokenize
from .models.hypothesis_evidence import evidence_diagnostics
from .models.hypothesis_set import (count_top_k, fit_count_temperature, gfm_decode,
                                    independent_cardinality)
from .multimodal import function_projection


LOGIT_GRID = np.r_[-np.inf, np.arange(-4, 4.01, .25), np.inf]


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def _fit_threshold(scores, truth, observed, reference):
    truth = truth & observed
    positives = truth.sum(axis=0)

    def objective(value):
        predicted = (scores >= value) & observed
        hits = (predicted & truth).sum(axis=0)
        f1 = _ratio(2 * hits, positives + predicted.sum(axis=0))
        return float(np.mean(f1)), -abs(value - reference) if value != reference else 0., value

    return float(max(LOGIT_GRID, key=objective))


def _ordered_batches(rows, items):
    position = np.concatenate(items["position"])
    order = np.argsort(position)
    restore = order[np.searchsorted(position[order], rows)]
    return {key: np.concatenate(value)[restore] for key, value in items.items() if key != "position"}


@torch.inference_mode()
def calibrate(session, model, rows):
    """Fit thresholds and H3 count temperature on the calibration side only."""
    data = session.data
    result = {"thresholds": [0.] * len(data.labels), "global_threshold": 0., "temperature": 1.,
              "threshold_source": "no_calibration_rows", "calibration_rows": len(rows),
              "count_calibration_rows": 0}
    if not len(rows):
        return result

    model.eval()
    has_counts = model.name == "h3" and model.head is not None
    items = defaultdict(list)
    for indices, tokens, mask in session.loader(rows):
        tokens, mask = tokens.to(session.device), mask.to(session.device)
        if has_counts:
            encoded = model.backbone(tokens, mask)
            counts, _ = model.head(encoded.evidence, encoded.summary, encoded.logits)
            items["count_logits"].append(counts.float().cpu().numpy())
            scores = encoded.logits
        else:
            scores = model(tokens, mask)["decision_scores"]
        items["scores"].append(scores.float().cpu().numpy())
        items["position"].append(indices)

    values = _ordered_batches(rows, items)
    scores = values["scores"]
    truth = data.targets[rows].astype(bool)
    observed = data.observed[rows]
    groups = data.clusters[rows]
    positive, negative = truth & observed, ~truth & observed
    if (positive.any(axis=0) & negative.any(axis=0)).any():
        global_threshold = _fit_threshold(scores, truth, observed, 0.)
        thresholds = [global_threshold] * truth.shape[1]
        for label in range(truth.shape[1]):
            pos_groups = np.unique(groups[positive[:, label]]).size
            neg_groups = np.unique(groups[negative[:, label]]).size
            if min(pos_groups, neg_groups) >= 5:
                thresholds[label] = _fit_threshold(scores[:, label], truth[:, label],
                                                   observed[:, label], global_threshold)
        result.update(thresholds=thresholds, global_threshold=global_threshold,
                      threshold_source="reliable_positions")
    elif session.settings["unlabeled_weight"] > 0:
        project_truth = data.y[rows].astype(bool)
        global_threshold = _fit_threshold(scores, project_truth, np.ones_like(project_truth), 0.)
        result.update(thresholds=[global_threshold] * truth.shape[1], global_threshold=global_threshold,
                      threshold_source="project_annotation_proxy")
    else:
        result["threshold_source"] = "no_reliable_contrast"

    if has_counts:
        result["count_calibration_rows"] = int(observed.all(1).sum())
        result["temperature"] = fit_count_temperature(values["count_logits"], truth, observed)
    return result


def _sample_metrics(truth, predicted):
    hits = (truth & predicted).sum(1)
    true_count, predicted_count = truth.sum(1), predicted.sum(1)
    return {"accuracy": float(_ratio(hits, true_count + predicted_count - hits).mean()),
            "precision": float(_ratio(hits, predicted_count).mean()),
            "coverage": float(_ratio(hits, true_count).mean()),
            "sample_f1": float(_ratio(2 * hits, true_count + predicted_count).mean())}


def _observed_metrics(truth, observed, scores, predicted, train_support):
    truth = truth & observed
    predicted = predicted & observed
    measured = observed.any(1)
    contrasted = truth.any(0) & (observed & ~truth).any(0)
    aps = [float(average_precision_score(truth[observed[:, c], c], scores[observed[:, c], c]))
           if contrasted[c] else None for c in range(truth.shape[1])]
    available_ap = [score for score in aps if score is not None]
    tail_ap = [score for score, support in zip(aps, train_support) if score is not None and support < 200]
    report = {"observed_rows": int(measured.sum()), "all_unknown_rows": int((~measured).sum()),
              "complete_rows": int(observed.all(1).sum()), "observed_positions": int(observed.sum()),
              "reliable_negative_positions": int((observed & ~truth).sum()),
              "labels_with_reliable_contrast": int(contrasted.sum()), "label_ap": aps,
              "macro_ap": float(np.mean(available_ap)) if available_ap else None,
              "tail_macro_ap": float(np.mean(tail_ap)) if tail_ap else None}
    if measured.any():
        report.update(_sample_metrics(truth[measured], predicted[measured]))
        if not report["reliable_negative_positions"]:
            # Positive-only coverage is measurable. Unknown predictions cannot
            # establish precision or set agreement.
            report.update(accuracy=None, precision=None, sample_f1=None)
    else:
        report.update(accuracy=None, precision=None, coverage=None, sample_f1=None)
    return report


def _cardinality_report(truth, observed, predicted, groups):
    complete = observed.all(1)
    known_positive = (truth & observed).any(1)
    unknown = ~observed.any(1)
    predicted_empty = ~predicted.any(1)
    result = {"complete_rows": int(complete.sum()), "complete_fraction": float(complete.mean()),
              "predicted_empty_fraction": float(predicted_empty.mean()),
              "all_unknown_rows": int(unknown.sum()),
              "all_unknown_predicted_empty_rate": float(predicted_empty[unknown].mean()) if unknown.any() else None,
              "known_positive_rows": int(known_positive.sum()),
              "known_positive_predicted_empty_rate": (float(predicted_empty[known_positive].mean())
                                                      if known_positive.any() else None)}
    if not complete.any():
        return result
    truth, predicted, groups = truth[complete], predicted[complete], groups[complete]
    true_count, predicted_count = truth.sum(1), predicted.sum(1)
    delta = predicted_count - true_count
    count_pairs, frequency = np.unique(np.stack((true_count, predicted_count), axis=1), axis=0, return_counts=True)
    result.update(count_mae=float(np.abs(delta).mean()), count_bias=float(delta.mean()),
                  confusion=np.column_stack((count_pairs, frequency)).tolist())
    strata = {}
    for label, lower, upper in (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5),
                                ("6-9", 6, 9), ("10+", 10, truth.shape[1])):
        selected = (true_count >= lower) & (true_count <= upper)
        item = {"rows": int(selected.sum()), "families": int(np.unique(groups[selected]).size)}
        if selected.any():
            known, chosen = truth[selected], predicted[selected]
            item.update(_sample_metrics(known, chosen),
                        predicted_empty_rate=float((predicted_count[selected] == 0).mean()),
                        missed_labels=float((known & ~chosen).sum(1).mean()),
                        extra_labels=float((~known & chosen).sum(1).mean()))
        strata[label] = item
    result["strata"] = strata
    return result


def _evidence_report(diagnostics):
    report = {"targets": len(diagnostics["label_indices"])}
    if report["targets"]:
        report["random_degenerate_fraction"] = float(diagnostics["random_degenerate"].mean())
        for key in ("keep_effect", "delete_effect", "random_keep_effect", "random_delete_effect"):
            effects = diagnostics[key].reshape(-1, 2).mean(axis=0)
            report[key] = {"target_probability_delta": float(effects[0]),
                           "mean_absolute_other_probability_delta": float(effects[1])}
    return report


@torch.inference_mode()
def evaluate_rows(session, model, rows, calibration, *, diagnostics=False):
    """Score untouched rows and return row-aligned arrays for optional export."""
    if not len(rows):
        raise ValueError("The evaluation split has no sequence rows")
    data = session.data
    model.eval()
    threshold = torch.as_tensor(calibration["thresholds"], device=session.device)
    temperature = calibration["temperature"]
    items, evidence = defaultdict(list), defaultdict(list)
    generator = torch.Generator(device=session.device).manual_seed(session.settings["seed"])
    for indices, tokens, mask in session.loader(rows):
        tokens, mask = tokens.to(session.device), mask.to(session.device)
        output = model(tokens, mask, threshold=threshold, temperature=temperature)
        rank, decision = output["rank_scores"], output["decision_scores"]
        threshold_scores = rank if model.name == "h3" else decision
        output["global_predicted"] = threshold_scores >= calibration["global_threshold"]
        if "count_probabilities" in output:
            output["b0_threshold_predicted"] = rank >= threshold
            output["count_top_k_predicted"] = count_top_k(output["joint_probabilities"].sum(-1), output["count_probabilities"])
            independent_counts, independent_joint = independent_cardinality(rank)
            output["independent_gfm_predicted"] = gfm_decode(independent_counts, independent_joint)
        for key, value in output.items():
            if key != "joint_probabilities":
                items[key].append(value.cpu().numpy())
        items["position"].append(indices)
        if diagnostics and model.name == "h6":
            effects = evidence_diagnostics(model.backbone, tokens, mask,
                                           torch.as_tensor(data.targets[indices], device=session.device),
                                           torch.as_tensor(data.observed[indices], device=session.device),
                                           generator=generator)
            positions = indices[effects.pop("row_indices").cpu().numpy()]
            evidence["row_id"].append(data.rows[positions])
            for key, value in effects.items():
                evidence[key].append(value.cpu().numpy())

    values = _ordered_batches(rows, items)
    target, observed = data.targets[rows].astype(bool), data.observed[rows]
    raw_truth = data.y[rows]
    train_support = data.y[session.train_indices].sum(0)
    projection = function_projection(data.labels, data.track)
    rank = values["rank_scores"]
    probabilities = torch.from_numpy(rank).sigmoid().numpy()

    predicted = values["predicted"]
    report = {"rows": len(rows), "labels": data.labels,
              "metrics": evaluate_metrics(raw_truth, rank, train_support=train_support,
                                          function_projection=projection, predicted=predicted),
              "observed_metrics": _observed_metrics(target, observed, rank, predicted, train_support),
              "cardinality": _cardinality_report(target, observed, predicted, data.clusters[rows]),
              "global_threshold_metrics": _sample_metrics(raw_truth, values["global_predicted"])}
    if "count_probabilities" in values:
        complete = observed.all(1)
        if complete.any():
            count_probs = values["count_probabilities"][complete]
            true_sizes = target[complete].sum(1)
            report["cardinality"]["count_nll"] = float(-np.log(count_probs[np.arange(len(true_sizes)), true_sizes]
                                                                 .clip(min=1e-30)).mean())
        controls = {}
        for key in ("b0_threshold", "count_top_k", "independent_gfm"):
            decision = values[f"{key}_predicted"]
            controls[key] = {"metrics": _sample_metrics(raw_truth, decision),
                             "cardinality": _cardinality_report(target, observed, decision, data.clusters[rows])}
        report["h3_controls"] = controls
    report.update(relation_diagnostics(model, data, rows, rank, predicted, values.get("pair_probabilities")))

    values.update(row_id=data.rows[rows], ID=data.ids[rows].astype(str), label_names=np.asarray(data.labels),
                  targets=target, observed=observed, project_targets=raw_truth, probabilities=probabilities,
                  decision_probabilities=torch.from_numpy(values["decision_scores"]).sigmoid().numpy())
    if evidence:
        effects = {key: np.concatenate(value) for key, value in evidence.items()}
        report["h6_evidence"] = _evidence_report(effects)
        values.update({f"evidence_{key}": value for key, value in effects.items()})
    return report, values


@torch.inference_mode()
def predict_sequences(model, calibration, sequences, device="cpu", batch_size=32):
    """Apply the deployed rule to raw sequences in the input order."""
    model.to(device).eval()
    threshold = torch.as_tensor(calibration["thresholds"], device=device)
    labels = model.labels
    records = []
    for start in range(0, len(sequences), batch_size):
        selected = sequences[start:start + batch_size]
        tokens, mask = pad_tokens([tokenize(seq) for seq in selected])
        output = model(tokens.to(device), mask.to(device), threshold=threshold, temperature=calibration["temperature"])
        rank, decision = output["rank_scores"].tolist(), output["decision_scores"].tolist()
        probability, predicted = output["decision_scores"].sigmoid().tolist(), output["predicted"].tolist()
        for i, seq in enumerate(selected):
            record = {"sequence": seq, "predicted_labels": [name for name, yes in zip(labels, predicted[i]) if yes],
                      "rank_scores": dict(zip(labels, rank[i])), "decision_scores": dict(zip(labels, decision[i])),
                      "probabilities": dict(zip(labels, probability[i])), "evidence_scope": "sequence_only"}
            if "count_probabilities" in output:
                record["count_probabilities"] = output["count_probabilities"][i].tolist()
            if "pair_probabilities" in output:
                record["pair_probabilities"] = [
                    {"labels": [labels[a], labels[b]], "probabilities_00_10_01_11": values}
                    for (a, b), values in zip(model.assets["pairs"], output["pair_probabilities"][i].tolist())]
            records.append(record)
    return records
