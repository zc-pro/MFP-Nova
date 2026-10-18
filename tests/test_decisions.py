import json

import numpy as np
import pytest

from mfp_nova.decisions import apply_decision, decision_scores, function_decisions, study
from mfp_nova.metrics import evaluate_metrics
from mfp_nova.multimodal import function_projection, independent_functions
from mfp_nova.research import epoch_loss_settings
from test_research import multimodal_fixture


def test_secondary_functions_preserve_subtypes_and_generic_semantics():
    labels = ['anti_gram+', 'anti_gram_', 'antimicrobial__unknown', 'antifungal__unknown', 'anticancer__unknown']
    projection = function_projection(labels, 'fine39')
    p = np.array([[.9, .8, .99, .6, .1], [.6, .7, .99, .1, .95], [.1, .1, .1, .1, .1]])
    original = p.copy()
    result = function_decisions(p, projection, .5, .85)
    # Two bacterial subtypes are one independent function; generic antimicrobial
    # evidence does not retain a second function when specific evidence is removed.
    assert result.tolist() == [[True, True, True, False, False],
                               [False, False, False, False, True], [False]*5]
    assert independent_functions(result, projection).sum(1).tolist() == [1, 1, 0]
    np.testing.assert_array_equal(p, original)
    np.testing.assert_array_equal(function_decisions(p, projection, .5, .5), p >= .5)


def test_decisions_change_binary_metrics_not_ranking():
    labels = ['antibacterial', 'antifungal', 'antimicrobial']
    projection = function_projection(labels, 'major15')
    y = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    p = np.array([[.9, .6, .1], [.9, .85, .1], [.1, .1, .9]])
    selected = function_decisions(p, projection, .5, .8)
    baseline = evaluate_metrics(y, p, .5, function_projection=projection)
    improved = evaluate_metrics(y, p, .5, function_projection=projection, predicted=selected)
    quick = decision_scores(y, selected, projection)
    for name in ('macro_ap', 'micro_ap', 'macro_auroc', 'micro_auroc'):
        assert baseline[name] == improved[name]
    assert improved['mfp_detection']['f1'] > baseline['mfp_detection']['f1']
    for name in ('accuracy', 'precision', 'coverage', 'absolute_true', 'absolute_false', 'micro_f1'):
        assert quick[name] == pytest.approx(improved[name])
    assert quick['mfp_f1'] == improved['mfp_detection']['f1']
    with pytest.raises(ValueError, match='shape'):
        evaluate_metrics(y, p, predicted=np.zeros((3, 1), bool))


def test_validation_study_and_test_row_rejection(tmp_path):
    from mfp_nova.data import PeptideData
    root, _ = multimodal_fixture(tmp_path)
    data = PeptideData(root, 'major15')
    idx = data.indices('val')
    y = data.y[idx]
    p = np.where(y, .9, .1).astype(float)
    p[1, 6] = .7
    path = tmp_path/'val.npz'
    np.savez(path, row_id=data.rows[idx], label_names=data.labels, targets=y, probabilities=p, threshold=.5)
    report = study(path, root, 'major15', tmp_path/'study')
    assert report['fit_split'] == 'val' and len(report['folds']) == 2
    assert report['baseline']['macro_ap'] == report['fitted_validation']['macro_ap']
    assert report['baseline']['macro_ap'] == report['decision_crossfit_validation']['macro_ap']
    decision = json.loads((tmp_path/'study/decision.json').read_text())
    assert apply_decision(p, data.labels, 'major15', decision).shape == p.shape
    with pytest.raises(ValueError, match='label order'):
        apply_decision(p, data.labels[::-1], 'major15', decision)
    np.savez(tmp_path/'test.npz', row_id=data.rows[data.indices('test')], label_names=data.labels,
             targets=data.y[data.indices('test')], probabilities=p, threshold=.5)
    with pytest.raises(ValueError, match='Validation'):
        study(tmp_path/'test.npz', root, 'major15', tmp_path/'forbidden')


def test_reconstruction_decay_preserves_legacy_and_other_objectives():
    config = {'loss': {'reconstruction': .05, 'direct': .2, 'unlabeled_weight': .3},
              'train': {'aux_warmup_epochs': 5}}
    assert epoch_loss_settings(config, 1) == {'reconstruction': .05*.2, 'direct': .2*.2, 'unlabeled_weight': .3}
    assert epoch_loss_settings(config, 20) == config['loss']
    config['train'].update(reconstruction_decay=[5, 20], reconstruction_floor=.2)
    values = [epoch_loss_settings(config, epoch)['reconstruction'] for epoch in range(5, 22)]
    assert values[0] == .05 and values[-1] == pytest.approx(.01)
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert epoch_loss_settings(config, 20)['direct'] == .2
    assert config['loss']['reconstruction'] == .05
    config['train']['reconstruction_decay'] = [5, 5]
    with pytest.raises(ValueError, match='decay'):
        epoch_loss_settings(config, 6)


def test_decay_candidate_is_separate_from_fixed_five_seed_runs(tmp_path):
    import copy
    import yaml
    from mfp_nova.experiments import PROJECT, compatibility_issues, discover_runs
    config = yaml.safe_load((PROJECT/'configs/trirelay.yaml').read_text(encoding='utf-8'))
    config['train'].update(batch_size=64, precision='fp32')
    metadata = {'model': 'trirelay', 'task': 'major15', 'seed': 42, 'config': config,
                'encoder': {'name': 'facebook/esm2_t33_650M_UR50D'}}
    assert compatibility_issues(metadata, None, tmp_path) == []
    candidate = copy.deepcopy(metadata)
    candidate['config']['train']['reconstruction_decay'] = [5, 20]
    assert any('reconstruction_decay' in item for item in compatibility_issues(candidate, None, tmp_path))
    directory = tmp_path/'trirelay_major15_reconstruction_decay_seed42'
    directory.mkdir()
    (directory/'metadata.json').write_text(json.dumps(candidate), encoding='utf-8')
    assert discover_runs(tmp_path) == []


def test_training_decay_resume_and_decision_inference(tmp_path):
    import torch
    from mfp_nova import research
    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    config = {'model_name': 'trirelay', 'model': {'dim': 32, 'heads': 4, 'dropout': 0},
              'data': {'root': str(root), 'track': 'major15', 'features': str(features)},
              'encoder': {'name': 'fixture', 'revision': 'main', 'max_length': 80},
              'loss': {'mask_ratio': .3, 'modality_dropout': .1, 'reconstruction': .05},
              'train': {'device': 'cpu', 'seed': 42, 'epochs': 1, 'batch_size': 2,
                        'learning_rate': .001, 'weight_decay': 0, 'patience': 0, 'num_workers': 0,
                        'aux_warmup_epochs': 0, 'reconstruction_decay': [0, 2], 'reconstruction_floor': .2,
                        'output': str(tmp_path)}}
    research.train(config, 'first')
    history = json.loads((tmp_path/'first/history.jsonl').read_text().splitlines()[0])
    assert history['loss_settings']['reconstruction'] == pytest.approx(.03)
    config['train']['epochs'] = 2
    research.train(config, 'continued', resume=tmp_path/'first/last.pt')
    history = json.loads((tmp_path/'continued/history.jsonl').read_text().splitlines()[0])
    assert history['epoch'] == 2 and history['loss_settings']['reconstruction'] == pytest.approx(.01)
    checkpoint = tmp_path/'continued/best.pt'
    base = research.evaluate(checkpoint, root, features, 'val', 'cpu', tmp_path/'base.json', tmp_path/'val.npz')
    fitted = study(tmp_path/'val.npz', root, 'major15', tmp_path/'decision')
    measured = research.evaluate(checkpoint, root, features, 'val', 'cpu', tmp_path/'with_decision.json',
                                 decision_path=tmp_path/'decision/decision.json')
    assert measured['metrics'] == fitted['fitted_validation']
    assert base['metrics']['macro_ap'] == measured['metrics']['macro_ap']
