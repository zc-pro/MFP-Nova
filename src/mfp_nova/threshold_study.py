"""CPU-only Validation threshold study, from a result ZIP or one prediction NPZ."""
import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .thresholds import SetScores, cluster_folds, fit_thresholds


def diagnostic_scores(y, predicted, labels):
    scores = SetScores(y, predicted).score()
    def functions(values):
        values = values.copy()
        generic = labels.index('antimicrobial')
        specific = [labels.index(c) for c in ('antibacterial', 'antifungal', 'antiparasitic', 'antiviral')]
        values[:, generic] &= ~values[:, specific].any(1)
        return values
    truth, guessed = functions(y), functions(predicted)
    actual_mfp, predicted_mfp = truth.sum(1) >= 2, guessed.sum(1) >= 2
    tp = int((actual_mfp & predicted_mfp).sum())
    fp, fn = (~y & predicted).sum(1), (y & ~predicted).sum(1)
    scores.update(mfp_precision=tp/max(1, int(predicted_mfp.sum())),
                  mfp_recall=tp/max(1, int(actual_mfp.sum())),
                  mfp_f1=2*tp/max(1, int(predicted_mfp.sum()+actual_mfp.sum())),
                  two_true_functions_recovered=int(((truth & guessed).sum(1) >= 2).sum()),
                  fp_only=int(((fp > 0) & (fn == 0)).sum()),
                  fn_only=int(((fn > 0) & (fp == 0)).sum()),
                  fp_and_fn=int(((fp > 0) & (fn > 0)).sum()),
                  empty_predictions=int((~predicted.any(1)).sum()),
                  mean_predicted_labels=float(predicted.sum(1).mean()),
                  multifunction_at=float((y[actual_mfp] == predicted[actual_mfp]).all(1).mean())
                  if actual_mfp.any() else None)
    return scores


def aligned_predictions(source, data_root):
    """Match the released original Val cohort, never fit on Test rows."""
    if str(source['split'].item()) != 'val' or str(source['task'].item()) != 'major15':
        raise ValueError('This study requires major15 Validation predictions')
    root = Path(data_root)
    with np.load(root/'arrays/sequences.npz', allow_pickle=False) as sequences:
        available = sequences['canonical_available']
        sequence_rows = sequences['row_id']
    with np.load(root/'tracks/major15/dataset.npz', allow_pickle=False) as data:
        if not np.array_equal(sequence_rows, data['row_id']):
            raise ValueError('Sequence/label row_id order differs')
        mask = data['val_original_mask'] & available
        rows, labels = data['row_id'][mask], data['label_names'].tolist()
        y, clusters = data['Y'][mask].astype(bool), data['cluster_id'][mask]
    provided_rows, provided_labels = source['row_id'].tolist(), source['label_names'].tolist()
    if (len(provided_rows) != len(rows) or set(provided_rows) != set(rows.tolist())
            or len(provided_labels) != len(labels) or set(provided_labels) != set(labels)):
        raise ValueError('Predictions must contain the complete original Val rows and labels exactly once')
    row_lookup = {r: j for j, r in enumerate(provided_rows)}
    col_lookup = {c: j for j, c in enumerate(provided_labels)}
    order = np.ix_([row_lookup[r] for r in rows], [col_lookup[c] for c in labels])
    p = source['probabilities'][order]
    if not np.array_equal(source['targets'][order], y):
        raise ValueError('Prediction targets differ from the released Val annotations')
    if not np.array_equal(source['cluster_id'][[row_lookup[r] for r in rows]], clusters):
        raise ValueError('Prediction sequence clusters differ from the released split')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite probabilities in [0, 1]')
    threshold = np.asarray(source['threshold'])
    if threshold.ndim == 1:
        threshold = threshold[[col_lookup[c] for c in labels]]
    return y, p, clusters, rows, labels, threshold


def study_one(source, data_root, destination, name, kappa, radius, protect_functions=False):
    y, p, clusters, rows, labels, saved = aligned_predictions(source, data_root)
    function_spec = (labels.index('antimicrobial'), [labels.index(c) for c in
                     ('antibacterial', 'antifungal', 'antiparasitic', 'antiviral')]) if protect_functions else None
    joint_name = 'joint_functions' if protect_functions else 'joint'
    methods, fit_info = fit_thresholds(y, p, clusters, kappa, radius, function_spec)
    full_scores = {method: diagnostic_scores(y, p >= t, labels) for method, t in methods.items()}
    cross = {method: np.zeros_like(y) for method in methods}
    folds = []
    for fit, held in cluster_folds(clusters):
        fitted, info = fit_thresholds(y[fit], p[fit], clusters[fit], kappa, radius, function_spec)
        held_scores = {}
        for method, t in fitted.items():
            cross[method][held] = p[held] >= t
            held_scores[method] = diagnostic_scores(y[held], cross[method][held], labels)
        folds.append({'fit_rows': len(fit), 'held_rows': len(held),
                      'fit_positive_support': y[fit].sum(0).tolist(),
                      'thresholds': {m: t.tolist() for m, t in fitted.items()},
                      'fit_information': info, 'held_metrics': held_scores})
    cross_scores = {m: diagnostic_scores(y, b, labels) for m, b in cross.items()}
    baseline = cross_scores['global']
    deltas = {m: {k: s[k]-baseline[k] for k in
              ('accuracy', 'absolute_true', 'absolute_false', 'macro_f1', 'mfp_f1')}
              for m, s in cross_scores.items()}
    report = {'prediction': name, 'task': 'major15', 'rows': len(y), 'labels': labels,
              'epoch': int(source['epoch']), 'weight_source': str(source['weight_source'].item()),
              'annotation_based_metrics': True, 'raw_ranking_scores': 'unchanged; no AP gain from thresholding',
              'settings': {'positive_cluster_kappa': kappa, 'maximum_radius': radius,
                           'protect_functions': protect_functions,
                           'search_passes': 3, 'pair_candidates': 6,
                           'pair_grid_locations': 7, 'hyperparameter_tuning': 'none'},
              'joint_objective': 'Acc, then AT, then Macro-F1, then smaller residual; fit-set Acc/AT/Macro-F1 >= global, AF <= global'
                  + ('; MFP-F1 and two-observed-function recovery >= global' if protect_functions else ''),
              'scope': 'Exploratory cluster-disjoint decision cross-fit within Val. Checkpoint already selected on full Val. Not independent Test performance. Full-Val fit scores are optimistic.'
                  + (' Function protection proposed after observing earlier cross-fit results.' if protect_functions else ''),
              'saved_threshold': saved.tolist(), 'saved_metrics': diagnostic_scores(y, p >= saved, labels),
              'full_val_thresholds': {m: t.tolist() for m, t in methods.items()},
              'full_val_fit_information': fit_info, 'full_val_metrics': full_scores,
              'crossfit_metrics': cross_scores, 'crossfit_delta_from_global': deltas, 'folds': folds}
    (destination/f'{name}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    decision = {'task': 'major15', 'labels': labels, 'fit_split': 'val',
                'rule': 'cluster_shrunk_set_joint_functions' if protect_functions else 'cluster_shrunk_set_joint',
                'label_threshold': methods[joint_name].tolist(), 'secondary_threshold': None,
                'status': 'diagnostic candidate; not automatically adopted', 'settings': report['settings']}
    (destination/f'{name}_decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(destination/f'{name}_crossfit.npz', row_id=rows,
                        label_names=np.asarray(labels), targets=y, **cross)
    records = []
    for scope, scores in [('full_val_fit', full_scores), ('cluster_crossfit', cross_scores)]:
        for method, values in scores.items():
            records.append({'prediction': name, 'scope': scope, 'method': method, **values})
            print(f"{name} {scope} {method}: " + ' '.join(f'{k}={values[k]:.4f}' for k in
                  ('accuracy', 'absolute_true', 'macro_f1', 'absolute_false', 'mfp_f1')), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True, help='Result ZIP containing best*_validation.npz, or one NPZ')
    parser.add_argument('--data-root', default='data/therapeutic_2025')
    parser.add_argument('--output', required=True)
    parser.add_argument('--kappa', type=float, default=20.)
    parser.add_argument('--radius', type=float, default=.10)
    parser.add_argument('--protect-functions', action='store_true',
                        help='Also protect fit-set MFP-F1 and recovery of two observed independent functions')
    args = parser.parse_args()
    if args.kappa <= 0 or not 0 < args.radius <= .5:
        parser.error('Require kappa > 0 and 0 < radius <= 0.5')
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    path = Path(args.predictions)
    if path.suffix.lower() == '.zip':
        with ZipFile(path) as archive:
            members = [n for n in archive.namelist() if Path(n).name in ('best_validation.npz', 'best_ap_validation.npz')]
            if not members:
                raise ValueError('Result archive contains no best Validation prediction NPZ')
            for name in sorted(members):
                with np.load(BytesIO(archive.read(name)), allow_pickle=False) as source:
                    records += study_one(source, args.data_root, destination, Path(name).stem, args.kappa, args.radius, args.protect_functions)
    else:
        with np.load(path, allow_pickle=False) as source:
            records += study_one(source, args.data_root, destination, path.stem, args.kappa, args.radius, args.protect_functions)
    with (destination/'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(f'Results: {destination.resolve()}', flush=True)


if __name__ == '__main__':
    main()
