import copy
import json

import pytest
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from mfp_nova.research import train, update_ema, evaluate
from test_research import multimodal_fixture


def test_ema_successful_updates_and_buffer_semantics():
    model = nn.Linear(1, 1, bias=False)
    model.register_buffer('prior', torch.tensor([3]))
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(.5))
    with torch.no_grad():
        model.weight.fill_(2)
    update_ema(ema, model, 1, 2, False)
    assert ema.n_averaged == 0
    update_ema(ema, model, 2, 2, True)
    assert ema.n_averaged == 0
    update_ema(ema, model, 2, 2, False)
    with torch.no_grad():
        model.weight.fill_(4)
        model.prior.fill_(7)
    update_ema(ema, model, 2, 2, False)
    assert ema.n_averaged == 2
    assert ema.module.weight.item() == 3
    assert ema.module.prior.item() == 7


def test_ema_preserves_online_training_and_resumes_exactly(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    config = {'model_name': 'trirelay', 'model': {'dim': 32, 'heads': 4, 'dropout': .1},
              'data': {'root': str(root), 'track': 'major15', 'features': str(features)},
              'encoder': {'name': 'fixture', 'revision': 'main', 'max_length': 80},
              'loss': {'mask_ratio': .15, 'modality_dropout': .1, 'reconstruction': .05},
              'train': {'device': 'cpu', 'seed': 42, 'epochs': 3, 'batch_size': 1,
                        'learning_rate': .001, 'weight_decay': 0, 'patience': 0, 'num_workers': 0,
                        'warmup_epochs': 0, 'aux_warmup_epochs': 0, 'accumulation_steps': 2,
                        'output': str(tmp_path)}}
    train(copy.deepcopy(config), 'plain')
    config['train'].update(ema_decay=.9, ema_start_epoch=2)
    result = train(copy.deepcopy(config), 'ema')
    plain = torch.load(tmp_path/'plain/last.pt', weights_only=True)
    full = torch.load(tmp_path/'ema/last.pt', weights_only=True)
    for name, value in full['model_state'].items():
        torch.testing.assert_close(value, plain['model_state'][name], rtol=0, atol=0)
    assert full['ema_state']['n_averaged'].item() == 4  # two successful updates per epoch after warmup
    assert any(not torch.equal(full['model_state'][k], full['ema_state']['module.'+k]) for k in full['model_state'])
    history = [json.loads(line) for line in (tmp_path/'ema/history.jsonl').read_text().splitlines()]
    assert [h['validation_weights'] for h in history] == ['online_warmup', 'ema', 'ema']
    assert history[0]['validation'] == history[0]['online_validation']
    from mfp_nova import research
    real_format = research.format_epoch
    def interrupt_after_saved_epoch(*args):
        if args[4]['epoch'] == 2:
            raise RuntimeError('simulated epoch-boundary interruption')
        return real_format(*args)
    with monkeypatch.context() as patch:
        patch.setattr(research, 'format_epoch', interrupt_after_saved_epoch)
        with pytest.raises(RuntimeError, match='simulated epoch-boundary'):
            train(copy.deepcopy(config), 'partial')
    resumed = train(copy.deepcopy(config), 'resumed', resume=tmp_path/'partial/last.pt')
    saved = torch.load(tmp_path/'resumed/last.pt', weights_only=True)
    for key in ('model_state', 'ema_state', 'best_model_state', 'online_best_state'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, saved[key][name], rtol=0, atol=0)
    assert result['validation'] == resumed['validation']
    assert result['online_reference']['validation'] == resumed['online_reference']['validation']
    for file, expected in [('best.pt', result['validation']), ('online_best.pt', result['online_reference']['validation'])]:
        measured = evaluate(tmp_path/'ema'/file, root, features, 'val', 'cpu', tmp_path/(file+'.json'), batch_size=1)
        assert measured['metrics'] == expected
    terminal = train(copy.deepcopy(config), 'terminal', resume=tmp_path/'resumed/last.pt')
    assert terminal['global_step'] == result['global_step']
    assert terminal['validation'] == result['validation']
    assert terminal['online_reference']['validation'] == result['online_reference']['validation']
    with pytest.raises(ValueError, match='new training run'):
        train(copy.deepcopy(config), 'invalid_resume', resume=tmp_path/'plain/last.pt')


def test_ema_is_separate_from_original_five_seed_protocol(tmp_path):
    import yaml
    from mfp_nova.experiments import PROJECT, discover_runs
    config = yaml.safe_load((PROJECT/'configs/trirelay.yaml').read_text(encoding='utf-8'))
    config['train'].update(batch_size=64, precision='fp32', ema_decay=.999, ema_start_epoch=6)
    directory = tmp_path/'trirelay_major15_ema_seed42'
    directory.mkdir()
    metadata = {'model': 'trirelay', 'task': 'major15', 'seed': 42, 'config': config,
                'encoder': {'name': 'facebook/esm2_t33_650M_UR50D'}}
    (directory/'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    assert discover_runs(tmp_path) == []
