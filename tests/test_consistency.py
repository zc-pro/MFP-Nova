import copy
import json

import numpy as np
import pytest
import torch

from mfp_nova.consistency import bernoulli_consistency, consistency_weight
from mfp_nova import research
from test_research import multimodal_fixture


def test_consistency_is_stable_detached_and_respects_positive_and_scope_masks():
    student = torch.tensor([[2., -3., -80., 80.], [0., 1., 80., -80.]], requires_grad=True)
    teacher = torch.tensor([[-2., 3., 80., -80.], [1., -1., -80., 80.]], requires_grad=True)
    target = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    mask = torch.tensor([[1, 1, 0, 1], [1, 1, 1, 1]], dtype=torch.bool)
    loss, terms = bernoulli_consistency(student, teacher, target, mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()
    assert teacher.grad is None
    assert student.grad[0, 0] == student.grad[1, 1] == student.grad[0, 2] == 0
    assert student.grad[0, 1] < 0 and student.grad[0, 3] > 0
    assert terms['consistency_high'] > 0 and terms['consistency_low'] > 0
    identical, _ = bernoulli_consistency(student.detach(), student.detach(), target)
    assert identical.item() == 0
    all_observed, _ = bernoulli_consistency(student, teacher, torch.ones_like(target))
    assert all_observed.item() == 0


def test_low_score_cells_have_a_separate_bounded_budget():
    student = torch.tensor([[0., 0.]], requires_grad=True)
    teacher = torch.tensor([[2., -2.]])
    target = torch.zeros_like(student)
    loss, terms = bernoulli_consistency(student, teacher, target)
    expanded, _ = bernoulli_consistency(torch.cat([student[:, :1], student[:, 1:].repeat(1, 100)], 1),
                                       torch.cat([teacher[:, :1], teacher[:, 1:].repeat(1, 100)], 1),
                                       torch.zeros(1, 101))
    torch.testing.assert_close(loss, expanded)
    torch.testing.assert_close(loss, terms['consistency_high']+.1*terms['consistency_low'])


def test_consistency_schedule_leaves_ema_initialization_epoch_unchanged():
    settings = {'consistency_weight': .1, 'consistency_start_epoch': 7, 'consistency_ramp_epochs': 5}
    assert [consistency_weight(settings, e) for e in (1, 6, 7, 9, 11, 20)] == pytest.approx([0, 0, .02, .06, .1, .1])


def fixture_config(tmp_path, task='major15'):
    root, features = multimodal_fixture(tmp_path)
    if task == 'fine39':
        labels = ['analgesic', 'anti_gram+', 'anti_gram_', 'antiangiogenesis', 'antibacterial__unknown',
                  'antibreastcancer', 'anticancer__unknown', 'anticervicalcancer', 'anticoloncancer',
                  'antifungal__unknown', 'antihcv', 'antihiv', 'antihsv', 'antihypertensive__unknown',
                  'antiinflammatory', 'antilungcancer', 'antimicrobial__unknown', 'antioxidant__unknown',
                  'antiparasitic__unknown', 'antiplasmodial', 'antiprostatecancer', 'antisars', 'antiskincancer',
                  'antitrypanosomic', 'antiviral__unknown', 'blood_brain_barrier_penetrating',
                  'cell_communication__unknown', 'cell_penetrating', 'glucose_metabolism',
                  'growth_regeneration__unknown', 'growth_regulatory', 'immunoactive__unknown', 'lipid_metabolism',
                  'metabolic_regulatory__unknown', 'neuropeptide__unknown', 'opioid', 'quorum_sensing',
                  'thrombolytic__unknown', 'tumor_homing']
        with np.load(root/'tracks/major15/dataset.npz') as source:
            arrays = dict(source)
        arrays['Y'] = np.zeros((8, 39), np.uint8)
        arrays['Y'][::2, [1, 4]] = 1
        arrays['Y'][1::2, [0, 34]] = 1
        arrays['label_names'] = labels
        (root/'tracks/fine39').mkdir()
        np.savez(root/'tracks/fine39/dataset.npz', **arrays)
    return {'model_name': 'trirelay', 'model': {'dim': 32, 'heads': 4, 'dropout': .1},
            'data': {'root': str(root), 'track': task, 'features': str(features)},
            'encoder': {'name': 'fixture', 'revision': 'main', 'max_length': 80},
            'loss': {'mask_ratio': .3, 'modality_dropout': .2, 'reconstruction': .05},
            'train': {'device': 'cpu', 'seed': 42, 'epochs': 3, 'batch_size': 2, 'learning_rate': .001,
                      'weight_decay': 0, 'patience': 0, 'num_workers': 0, 'warmup_epochs': 0,
                      'aux_warmup_epochs': 0, 'output': str(tmp_path), 'ema_decay': .9, 'ema_start_epoch': 1,
                      'consistency_weight': .1, 'consistency_ramp_epochs': 2}}


@pytest.mark.parametrize('task', ['major15', 'fine39'])
def test_consistency_training_both_tasks_and_independent_val(tmp_path, task):
    torch.set_num_threads(1)
    config = fixture_config(tmp_path, task)
    result = research.train(copy.deepcopy(config), 'candidate')
    history = [json.loads(line) for line in (tmp_path/'candidate/history.jsonl').read_text().splitlines()]
    assert [h['loss_settings']['consistency'] for h in history] == pytest.approx([0, .05, .1])
    assert history[0]['losses']['consistency'] == 0
    assert all(h['losses']['consistency'] > 0 for h in history[1:])
    saved = torch.load(tmp_path/'candidate/last.pt', weights_only=True)
    assert saved['config']['train']['consistency_start_epoch'] == 2
    assert saved['ema_state']['n_averaged'].item() == 6
    assert all(torch.isfinite(v).all() for v in saved['model_state'].values())
    measured = research.evaluate(tmp_path/'candidate/best.pt', config['data']['root'], config['data']['features'],
                                 'val', 'cpu', tmp_path/'evaluation.json', batch_size=2)
    assert measured['metrics'] == result['validation']
    assert len(result['labels']) == (15 if task == 'major15' else 39)


def test_consistency_resume_preserves_training_and_cannot_retrofit_old_ema(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = fixture_config(tmp_path)
    research.train(copy.deepcopy(config), 'complete')
    real_format = research.format_epoch

    def interrupt(*args):
        if args[4]['epoch'] == 2:
            raise RuntimeError('saved epoch interruption')
        return real_format(*args)

    with monkeypatch.context() as patch:
        patch.setattr(research, 'format_epoch', interrupt)
        with pytest.raises(RuntimeError, match='saved epoch'):
            research.train(copy.deepcopy(config), 'partial')
    research.train(copy.deepcopy(config), 'resumed', resume=tmp_path/'partial/last.pt')
    full = torch.load(tmp_path/'complete/last.pt', weights_only=True)
    resumed = torch.load(tmp_path/'resumed/last.pt', weights_only=True)
    for key in ('model_state', 'ema_state', 'best_model_state', 'online_best_state'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)
    plain = copy.deepcopy(config)
    plain['train'].pop('consistency_weight')
    plain['train'].pop('consistency_ramp_epochs')
    plain['train']['epochs'] = 1
    research.train(plain, 'old_ema')
    with pytest.raises(ValueError, match='Consistency changes'):
        research.train(copy.deepcopy(config), 'invalid', resume=tmp_path/'old_ema/last.pt')


@pytest.mark.parametrize('change', [{'model_name': 'geofactor'}, {'ema_decay': None}, {'consistency_weight': -.1},
                                  {'consistency_start_epoch': 1}, {'consistency_ramp_epochs': 0}])
def test_consistency_rejects_incompatible_experiments(tmp_path, change):
    config = fixture_config(tmp_path)
    if 'model_name' in change:
        config.update(change)
    else:
        config['train'].update(change)
    with pytest.raises(ValueError):
        research.train(config, 'invalid')
