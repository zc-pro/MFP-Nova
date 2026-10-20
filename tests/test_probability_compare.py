import json

import numpy as np
import pytest

from mfp_nova.data import PeptideData
from mfp_nova.decisions import compare_predictions, decision_scores, load_validation_predictions
from mfp_nova.metrics import evaluate_metrics
from mfp_nova.multimodal import function_projection
from test_research import multimodal_fixture


def pair(tmp_path):
    root, _ = multimodal_fixture(tmp_path)
    data = PeptideData(root, 'major15')
    index = data.indices('val')
    y = data.y[index]
    p = np.where(y, .8, .1).astype(np.float32)
    p[1, 6] = .6
    fields = {'row_id': data.rows[index], 'label_names': np.asarray(data.labels), 'targets': y,
              'probabilities': p, 'threshold': .5, 'split': 'val', 'task': 'major15'}
    np.savez(tmp_path/'reference.npz', **fields)
    reverse = np.arange(len(data.labels))[::-1]
    candidate = {**fields, 'row_id': fields['row_id'][::-1], 'label_names': fields['label_names'][reverse],
                 'targets': y[::-1][:, reverse], 'probabilities': p[::-1][:, reverse], 'threshold': .75}
    np.savez(tmp_path/'candidate.npz', **candidate)
    return root, data, fields


def test_paired_alignment_saved_and_common_thresholds(tmp_path):
    root, data, fields = pair(tmp_path)
    aligned = load_validation_predictions(tmp_path/'candidate.npz', data)
    np.testing.assert_array_equal(aligned['probabilities'], fields['probabilities'])
    assert aligned['rows_reordered'] and aligned['labels_reordered']
    report = compare_predictions(tmp_path/'reference.npz', tmp_path/'candidate.npz', root, 'major15', tmp_path/'out')
    assert report['saved_metrics']['reference']['macro_ap'] == report['saved_metrics']['candidate']['macro_ap']
    experiments = report['threshold_experiments']
    assert experiments['saved']['delta']['accuracy'] > 0  # identical scores, different saved thresholds
    for key in ('shared_0.50', 'shared_reference_threshold', 'shared_candidate_threshold', 'per_label_crossfit'):
        assert all(v == 0 for v in experiments[key]['delta'].values())
    assert (tmp_path/'out/label_diagnostics.csv').exists()
    assert not (tmp_path/'out/decision.json').exists()
    saved = json.loads((tmp_path/'out/results.json').read_text(encoding='utf-8'))
    assert saved['fit_split'] == 'val'
    assert saved['groups_at_saved_threshold']['D_annotation']['rows'] == 1
    assert saved['groups_at_saved_threshold']['structure_missing']['rows'] == 1
    quick = decision_scores(fields['targets'], fields['probabilities'] >= .5, function_projection(data.labels, 'major15'))
    full = evaluate_metrics(fields['targets'], fields['probabilities'])
    assert quick['macro_f1'] == pytest.approx(full['macro_f1'])


def test_crossfit_fits_only_on_cluster_disjoint_parts(tmp_path, monkeypatch):
    from mfp_nova import decisions
    root, _, _ = pair(tmp_path)
    fit_sizes = []

    def fit(y, p):
        fit_sizes.append(len(y))
        return [.5]*y.shape[1]

    monkeypatch.setattr(decisions, 'fit_label_thresholds', fit)
    report = compare_predictions(tmp_path/'reference.npz', tmp_path/'candidate.npz', root, 'major15', tmp_path/'out')
    assert sorted(fit_sizes) == [1, 1, 1, 1, 2, 2]
    for fold in report['decision_folds']:
        assert set(fold['fit_clusters']).isdisjoint(fold['held_clusters'])
        assert fold['fit_rows'] == fold['held_rows'] == 1
        assert len(fold['fit_positive_support']) == 15


def test_label_threshold_vector_is_reordered_with_labels(tmp_path):
    _, data, fields = pair(tmp_path)
    thresholds = np.linspace(.2, .9, len(data.labels))
    columns = np.arange(len(data.labels))[::-1]
    np.savez(tmp_path/'permuted.npz', **{**fields, 'label_names': fields['label_names'][columns],
             'targets': fields['targets'][:, columns], 'probabilities': fields['probabilities'][:, columns],
             'threshold': thresholds[columns]})
    loaded = load_validation_predictions(tmp_path/'permuted.npz', data)
    np.testing.assert_array_equal(loaded['threshold'], thresholds)


@pytest.mark.parametrize('bad', ['split', 'task', 'row_id', 'labels', 'targets', 'probabilities'])
def test_wrong_cohort_annotations_and_probabilities_are_rejected(tmp_path, bad):
    _, data, fields = pair(tmp_path)
    fields = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in fields.items()}
    if bad == 'split':
        fields['split'] = 'test'
    elif bad == 'task':
        fields['task'] = 'fine39'
    elif bad == 'row_id':
        fields['row_id'][1] = fields['row_id'][0]
    elif bad == 'labels':
        fields['label_names'][1] = fields['label_names'][0]
    elif bad == 'targets':
        fields['targets'][0, 0] = 0
    else:
        fields['probabilities'][0, 0] = np.nan
    np.savez(tmp_path/'bad.npz', **fields)
    with pytest.raises(ValueError):
        load_validation_predictions(tmp_path/'bad.npz', data)
