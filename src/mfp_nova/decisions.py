"""Validation-only decision studies on saved probabilities, without model training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .data import PeptideData
from .metrics import evaluate_metrics
from .multimodal import function_projection, independent_functions


def function_decisions(probabilities, projection, label_threshold, secondary_threshold=None):
    """Retain the strongest active function; demand extra evidence for others.

    Probabilities are untouched. Generic antimicrobial annotations are retained
    with a retained specific antimicrobial function, never counted twice.
    """
    predicted = probabilities >= label_threshold
    if secondary_threshold is None or secondary_threshold <= label_threshold:
        return predicted
    projection = np.asarray(projection, dtype=bool)
    active = independent_functions(predicted, projection)
    scores = np.stack([probabilities[:, projection[:, j]].max(1) if projection[:, j].any()
                       else np.zeros(len(probabilities)) for j in range(projection.shape[1])], 1)
    strongest = np.where(active, scores, -1).argmax(1)
    keep = active & (scores >= secondary_threshold)
    keep[np.arange(len(keep)), strongest] |= active.any(1)
    keep[:, 4] |= keep[:, :4].any(1)
    return predicted & (keep.astype(int) @ projection.T > 0)


def apply_decision(probabilities, labels, task, decision):
    if decision['task'] != task or decision['labels'] != list(labels):
        raise ValueError('Decision task/label order differs from model')
    return function_decisions(np.asarray(probabilities), function_projection(labels, task),
                              decision['label_threshold'], decision.get('secondary_threshold'))


def decision_scores(targets, predicted, projection):
    """Cheap exact decision metrics for a threshold sweep; ranking is unchanged."""
    y = targets.astype(bool)
    tp = (y & predicted).sum(1)
    actual, guessed = y.sum(1), predicted.sum(1)
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    truth_mfp = independent_functions(y, projection).sum(1) >= 2
    guess_mfp = independent_functions(predicted, projection).sum(1) >= 2
    mtp = int((truth_mfp & guess_mfp).sum())
    mactual, mguessed = int(truth_mfp.sum()), int(guess_mfp.sum())
    return {'accuracy': float(divide(tp, actual+guessed-tp).mean()),
            'precision': float(divide(tp, guessed).mean()), 'coverage': float(divide(tp, actual).mean()),
            'absolute_true': float((y == predicted).all(1).mean()), 'absolute_false': float((y != predicted).mean()),
            'micro_f1': float(2*tp.sum()/max(1, actual.sum()+guessed.sum())),
            'mfp_precision': mtp/max(1, mguessed), 'mfp_recall': mtp/max(1, mactual),
            'mfp_f1': 2*mtp/max(1, mactual+mguessed), 'mfp_predicted': mguessed,
            'mfp_support': mactual}


def sweep(targets, probabilities, projection, baseline_threshold):
    candidates = [('baseline', float(baseline_threshold), None)]
    candidates += [('global', float(t), None) for t in np.round(np.arange(.05, .96, .01), 2)]
    candidates += [('secondary_function', float(baseline_threshold), float(t))
                   for t in np.round(np.arange(.05, .96, .01), 2) if t > baseline_threshold]
    return [{'rule': name, 'label_threshold': threshold, 'secondary_threshold': secondary,
             **decision_scores(targets, function_decisions(probabilities, projection, threshold, secondary), projection)}
            for name, threshold, secondary in candidates]


def select_candidate(candidates, jaccard_budget):
    baseline = candidates[0]
    eligible = [r for r in candidates if r['accuracy'] >= baseline['accuracy']-jaccard_budget]
    # Explicit tradeoff budget, never a new weighted checkpoint selection score.
    return max(eligible, key=lambda r: (r['mfp_f1'], r['accuracy'], r['micro_f1'], r['rule']=='baseline'))


def pareto_candidates(candidates):
    keys = ('accuracy', 'micro_f1', 'mfp_f1')
    return [r for r in candidates if not any(all(s[k] >= r[k] for k in keys) and
            any(s[k] > r[k] for k in keys) for s in candidates)]


def study(predictions, data_root, task, output, jaccard_budget=.005):
    data = PeptideData(data_root, task)
    indices = data.indices('val')
    with np.load(predictions, allow_pickle=False) as z:
        y, p = z['targets'], z['probabilities']
        threshold = float(z['threshold'])
        if z['label_names'].tolist() != data.labels or not np.array_equal(z['row_id'], data.rows[indices]):
            raise ValueError('Decision fitting requires the complete aligned Validation rows/labels; Test is not allowed')
        if not np.array_equal(y, data.y[indices]):
            raise ValueError('Prediction targets differ from current Validation annotations')
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite aligned probabilities in [0, 1]')
    if not 0 <= jaccard_budget <= 1:
        raise ValueError('Jaccard budget must be an absolute difference in [0, 1]')
    projection = function_projection(data.labels, task)
    support = data.y[data.indices('train')].sum(0)
    candidates = sweep(y, p, projection, threshold)
    chosen = select_candidate(candidates, jaccard_budget)
    decision = {'task': task, 'labels': data.labels, 'fit_split': 'val',
                'rule': chosen['rule'], 'label_threshold': chosen['label_threshold'],
                'secondary_threshold': chosen['secondary_threshold'],
                'selection': 'max_mfp_f1_with_explicit_jaccard_budget', 'jaccard_budget': jaccard_budget}
    # Cross-fit only the additional decision search, never retrain/select a model.
    cross_pred = np.zeros_like(y, dtype=bool)
    folds = []
    for fit, held in GroupKFold(n_splits=2).split(p, groups=data.clusters[indices]):
        candidate = select_candidate(sweep(y[fit], p[fit], projection, threshold), jaccard_budget)
        cross_pred[held] = function_decisions(p[held], projection, candidate['label_threshold'], candidate['secondary_threshold'])
        folds.append({'fit_rows': len(fit), 'held_rows': len(held), 'candidate': candidate,
                      'held': decision_scores(y[held], cross_pred[held], projection)})
    baseline = evaluate_metrics(y, p, threshold, support, projection)
    fitted = evaluate_metrics(y, p, threshold, support, projection, apply_decision(p, data.labels, task, decision))
    cross = evaluate_metrics(y, p, threshold, support, projection, cross_pred)
    report = {'task': task, 'rows': len(y), 'predictions': str(predictions), 'fit_split': 'val',
              'ranking_scope': 'unchanged_raw_probabilities', 'baseline': baseline,
              'fitted_validation': fitted, 'decision_crossfit_validation': cross,
              'crossfit_scope': 'cluster-disjoint decision folds within Val; checkpoint and original threshold already selected on full Val; not independent generalization',
              'decision': decision, 'folds': folds, 'pareto': pareto_candidates(candidates),
              'candidate_status': 'requires Train/Validation assessment; no automatic adoption'}
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in [('results.json', report), ('decision.json', decision)]:
        (destination/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (destination/'candidates.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(candidates[0])); writer.writeheader(); writer.writerows(candidates)
    for name, score in [('baseline', baseline), ('fitted_val', fitted), ('decision_crossfit_val', cross)]:
        print(f"{name}: MacroAP={score['macro_ap']:.6f} MFP-F1={score['mfp_detection']['f1']:.6f} "
              f"Acc={score['accuracy']:.6f} Prec={score['precision']:.6f} Cov={score['coverage']:.6f} "
              f"AT={score['absolute_true']:.6f} AF={score['absolute_false']:.6f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--task', required=True, choices=('major15', 'fine39'))
    parser.add_argument('--data-root', default='data/therapeutic_2025')
    parser.add_argument('--output', required=True)
    parser.add_argument('--jaccard-budget', type=float, default=.005, help='Absolute permitted Val Jaccard decrease; 0.005 is 0.5 percentage points')
    args = parser.parse_args()
    study(args.predictions, args.data_root, args.task, args.output, args.jaccard_budget)


if __name__ == '__main__':
    main()
