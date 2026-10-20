"""Validation-only decision studies on saved probabilities, without model training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .data import PeptideData
from .metrics import evaluate_metrics, fit_label_thresholds
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
            'macro_f1': float(divide(2*(y & predicted).sum(0), y.sum(0)+predicted.sum(0)).mean()),
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


def load_validation_predictions(path, data):
    """Align both axes by identity and reject incomplete or non-Val cohorts."""
    indices = data.indices('val')
    expected_rows = data.rows[indices]
    with np.load(path, allow_pickle=False) as source:
        if 'split' in source and str(source['split'].item()) != 'val':
            raise ValueError('Only Validation predictions are allowed')
        if 'task' in source and str(source['task'].item()) != data.track:
            raise ValueError('Prediction task differs from requested task')
        rows, labels = source['row_id'].tolist(), source['label_names'].tolist()
        if len(rows) != len(expected_rows) or set(rows) != set(expected_rows.tolist()):
            raise ValueError('Predictions require the complete Validation row_id set, without duplicates')
        if len(labels) != len(data.labels) or set(labels) != set(data.labels):
            raise ValueError('Prediction label set differs from current task')
        row_lookup = {row: i for i, row in enumerate(rows)}
        label_lookup = {label: i for i, label in enumerate(labels)}
        row_order = [row_lookup[row] for row in expected_rows.tolist()]
        label_order = [label_lookup[label] for label in data.labels]
        if source['probabilities'].shape != (len(rows), len(labels)) or source['targets'].shape != (len(rows), len(labels)):
            raise ValueError('Targets/probabilities must match row_id and label_names dimensions')
        y = source['targets'][np.ix_(row_order, label_order)]
        p = source['probabilities'][np.ix_(row_order, label_order)]
        threshold = np.asarray(source['threshold'], dtype=float)
        if threshold.ndim == 1 and len(threshold) == len(labels):
            threshold = threshold[label_order]
        elif threshold.ndim != 0:
            raise ValueError('Saved threshold must be a scalar or one value per label')
    if not np.array_equal(y, data.y[indices]):
        raise ValueError('Prediction targets differ from Validation annotations')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite and in [0, 1]')
    if not np.isfinite(threshold).all() or np.any((threshold < 0) | (threshold > 1)):
        raise ValueError('Invalid saved threshold')
    return {'targets': y, 'probabilities': p, 'threshold': threshold,
            'rows_reordered': rows != expected_rows.tolist(), 'labels_reordered': labels != data.labels}


def compare_predictions(reference, candidate, data_root, task, output):
    """Paired Val diagnostics; threshold experiments do not change the main protocol."""
    data = PeptideData(data_root, task)
    indices, train = data.indices('val'), data.indices('train')
    support = data.y[train].sum(0)
    projection = function_projection(data.labels, task)
    inputs = {name: load_validation_predictions(path, data)
              for name, path in [('reference', reference), ('candidate', candidate)]}
    y = inputs['reference']['targets']
    p = {name: value['probabilities'] for name, value in inputs.items()}
    saved = {name: p[name] >= value['threshold'] for name, value in inputs.items()}
    selected = {'saved': saved}
    for name, threshold in [('shared_0.50', .5), ('shared_reference_threshold', inputs['reference']['threshold']),
                            ('shared_candidate_threshold', inputs['candidate']['threshold'])]:
        selected[name] = {key: values >= threshold for key, values in p.items()}
    fitted = {name: fit_label_thresholds(y, values) for name, values in p.items()}
    selected['per_label_val_fit'] = {name: values >= fitted[name] for name, values in p.items()}
    cross = {name: np.zeros_like(y, dtype=bool) for name in p}
    folds = []
    clusters = data.clusters[indices]
    for fit, held in GroupKFold(n_splits=2).split(y, groups=clusters):
        thresholds = {name: fit_label_thresholds(y[fit], values[fit]) for name, values in p.items()}
        for name, values in p.items():
            cross[name][held] = values[held] >= thresholds[name]
        folds.append({'fit_rows': len(fit), 'held_rows': len(held),
                      'fit_clusters': np.unique(clusters[fit]).tolist(), 'held_clusters': np.unique(clusters[held]).tolist(),
                      'fit_positive_support': y[fit].sum(0).astype(int).tolist(),
                      'held_positive_support': y[held].sum(0).astype(int).tolist(), 'thresholds': thresholds})
    selected['per_label_crossfit'] = cross
    experiments = {}
    for name, predictions in selected.items():
        scores = {key: decision_scores(y, value, projection) for key, value in predictions.items()}
        scores['delta'] = {k: scores['candidate'][k]-scores['reference'][k] for k in scores['reference']}
        experiments[name] = scores
    primary = {name: evaluate_metrics(y, value['probabilities'], value['threshold'], support, projection)
               for name, value in inputs.items()}
    rows = []
    for j, label in enumerate(data.labels):
        row = {'label': label, 'train_support': int(support[j]), 'val_support': int(y[:, j].sum()),
               'stratum': 'head' if support[j] >= 1000 else 'medium' if support[j] >= 200 else 'tail'}
        for name, values in p.items():
            predicted = saved[name][:, j]
            tp, fp, fn = int((predicted & (y[:, j] == 1)).sum()), int((predicted & (y[:, j] == 0)).sum()), int((~predicted & (y[:, j] == 1)).sum())
            row.update({name+'_ap': primary[name]['label_ap'][j], name+'_f1': primary[name]['label_f1'][j],
                        name+'_tp': tp, name+'_fp_observed': fp, name+'_fn_observed': fn,
                        name+'_predicted': int(predicted.sum())})
            threshold = np.broadcast_to(inputs[name]['threshold'], (len(data.labels),))[j]
            row[name+'_near_threshold'] = int((np.abs(values[:, j]-threshold) <= .05).sum())
            for group, mask in [('positive', y[:, j] == 1), ('unannotated', y[:, j] == 0)]:
                quantiles = np.quantile(values[mask, j], [.1, .5, .9]).tolist() if mask.any() else [None]*3
                for quantile, value in zip(('p10', 'median', 'p90'), quantiles):
                    row[f'{name}_{group}_{quantile}'] = value
            cp = cross[name][:, j]
            ctp = int((cp & (y[:, j] == 1)).sum())
            row[name+'_crossfit_f1'] = 2*ctp/max(1, int(cp.sum()+y[:, j].sum()))
        rows.append(row)
    with np.load(Path(data_root)/'arrays/structure_primary.npz', allow_pickle=False) as source:
        if not np.array_equal(source['row_id'], data.rows):
            raise ValueError('Structure row_id order differs from labels')
        present = source['structure_available'][indices].astype(bool)
    with np.load(Path(data_root)/'arrays/sequences.npz', allow_pickle=False) as source:
        d = (data.d_mask[indices] & (source['canonical_tokens'][indices] != 6)).any(1)
    group_masks = {'structure_present': present, 'structure_missing': ~present, 'D_annotation': d, 'L_only': ~d}
    grouped = {name: {'rows': int(mask.sum()), **{key: decision_scores(y[mask], value[mask], projection)
                    for key, value in saved.items()}} if mask.any() else {'rows': 0}
               for name, mask in group_masks.items()}
    report = {'task': task, 'rows': len(y), 'labels': data.labels, 'fit_split': 'val',
              'reference': str(reference), 'candidate': str(candidate), 'annotation_based_metrics': True,
              'alignment': {name: {k: value[k] for k in ('rows_reordered', 'labels_reordered')} for name, value in inputs.items()},
              'saved_thresholds': {name: value['threshold'].tolist() for name, value in inputs.items()},
              'saved_metrics': primary, 'threshold_experiments': experiments,
              'per_label_fitted_thresholds': fitted, 'decision_folds': folds, 'groups_at_saved_threshold': grouped,
              'label_diagnostics': rows,
              'decision_crossfit_scope': 'Cluster-disjoint threshold fitting within Val; checkpoints already selected on full Val; not independent generalization',
              'per_label_fit_scope': 'Fitted and scored on the same Val; optimistic diagnostic, not a replacement result',
              'calibration_scope': 'Score distributions for observed positives versus unannotated cells; not true-positive posterior calibration',
              'candidate_status': 'Diagnostic only; no automatic adoption or checkpoint changes'}
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination/'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (destination/'label_diagnostics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    md = ['# TriRelay Val概率配对诊断', '', f'任务：{task}；完整Val：{len(y)}条。reference为EMA-only，candidate为EMA+一致性。', '',
          '以下均相对已有不完整标注；不将0视为实验确认阴性。主结果仍是各自checkpoint保存阈值，其余均为独立诊断，不修改权重或阈值。', '',
          '|模式|Reference Macro-F1|Candidate Macro-F1|Reference Jaccard|Candidate Jaccard|Reference MFP-F1|Candidate MFP-F1|',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name, values in experiments.items():
        md.append('|'+name+'|'+'|'.join(f'{values[variant][metric]:.6f}' for metric in ('macro_f1', 'accuracy', 'mfp_f1') for variant in ('reference', 'candidate'))+'|')
    md += ['', '`shared_*`对两组使用同一阈值，仅用于区分分数尺度与决策差异。',
           '`per_label_val_fit`在同一Val拟合并打分，存在乐观性；`per_label_crossfit`在Val内部按簇两折交叉拟合，不重训模型、不修改原数据划分。checkpoint仍已用完整Val选择，因此不是独立验证或Test成绩。',
           '详见results.json的每标签原始AP、各阈值模式及分组，label_diagnostics.csv提供TP/观测FP/FN、分数分位数、近阈值数量和交叉拟合F1。',
           '该诊断不能证明只改阈值就能修复模型，也不能从未标注分数直接认定概率校准优劣。']
    (destination/'report.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    for name, value in experiments.items():
        print(f"{name}: MacroF1 delta={value['delta']['macro_f1']:+.6f} "
              f"Jaccard delta={value['delta']['accuracy']:+.6f} MFP-F1 delta={value['delta']['mfp_f1']:+.6f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--reference', help='EMA-only Val NPZ for paired comparison; --predictions is the consistency candidate')
    parser.add_argument('--task', required=True, choices=('major15', 'fine39'))
    parser.add_argument('--data-root', default='data/therapeutic_2025')
    parser.add_argument('--output', required=True)
    parser.add_argument('--jaccard-budget', type=float, default=.005, help='Absolute permitted Val Jaccard decrease; 0.005 is 0.5 percentage points')
    args = parser.parse_args()
    if args.reference:
        compare_predictions(args.reference, args.predictions, args.data_root, args.task, args.output)
    else:
        study(args.predictions, args.data_root, args.task, args.output, args.jaccard_budget)


if __name__ == '__main__':
    main()
