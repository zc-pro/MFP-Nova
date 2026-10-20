import copy
import json

import numpy as np
import pytest
import torch

from mfp_nova import research
from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from mfp_nova.trirelay import TriRelay
from test_consistency import fixture_config


def auxiliary_config(tmp_path, task):
    # Build both real-label schemas; the fixture ESM is synthetic, not performance evidence.
    config = fixture_config(tmp_path, 'fine39')
    config['data']['track'] = task
    for key in ('consistency_weight', 'consistency_ramp_epochs'):
        config['train'].pop(key)
    config['train'].update(function_aux_weight=.1, function_hierarchy_weight=.02,
                           aux_warmup_epochs=2)
    return config


def read_track(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def test_auxiliary_pairing_preserves_primary_cohort_and_fits_train_only(tmp_path):
    config = auxiliary_config(tmp_path, 'major15')
    root = config['data']['root']
    from pathlib import Path
    path = Path(root)/'tracks/fine39/dataset.npz'
    arrays = read_track(path)
    arrays['train_original_mask'][1] = False
    arrays['row_in_scope'][1] = False
    np.savez(path, **arrays)
    source = MultimodalData(root, 'major15')
    source.enable_function_auxiliary()
    train = MultimodalPeptides(source, config['data']['features'], 'train')
    val = MultimodalPeptides(source, config['data']['features'], 'val')
    assert len(train) == 4
    assert source.statistics['function_auxiliary']['train_rows'] == 3
    assert not train[1]['aux_mask'] and train[1]['aux_y'].count_nonzero() == 0
    assert all(not val[i]['aux_mask'] for i in range(len(val)))
    assert all(val[i]['aux_y'].count_nonzero() == 0 for i in range(len(val)))
    prior = copy.deepcopy(source.statistics['function_auxiliary'])
    arrays['Y'][4:] = 1-arrays['Y'][4:]
    np.savez(path, **arrays)
    changed = MultimodalData(root, 'major15')
    changed.enable_function_auxiliary()
    assert changed.statistics['function_auxiliary'] == prior


@pytest.mark.parametrize('task', ['major15', 'fine39'])
def test_full_trirelay_auxiliary_shapes_masks_and_gradients(tmp_path, task):
    torch.set_num_threads(1)
    config = auxiliary_config(tmp_path, task)
    source = MultimodalData(config['data']['root'], task)
    source.enable_function_auxiliary()
    dataset = MultimodalPeptides(source, config['data']['features'], 'train')
    batch = collate_multimodal([dataset[i] for i in range(3)])
    model = TriRelay(12, len(source.data.labels), len(source.statistics['chemical_vocab'])+2,
                     source.statistics['labels'], dim=32, heads=4, dropout=0,
                     function_aux=True, function_aux_statistics=source.statistics['function_auxiliary'])
    model.train()
    total, terms, output = model.compute_loss(batch, {'function_aux':.1, 'function_hierarchy':.02}, return_output=True)
    assert output['logits'].shape == (3, 15 if task == 'major15' else 39)
    assert output['function_aux'].shape == (3, 39 if task == 'major15' else 15)
    total.backward()
    assert torch.isfinite(total)
    assert model.function_auxiliary.query.grad.abs().sum() > 0
    assert model.function_auxiliary.read.attention.in_proj_weight.grad.abs().sum() > 0
    assert model.feedback[0].attention.in_proj_weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    # Auxiliary objectives alone must propagate into the complete evidence pipeline.
    model.zero_grad(set_to_none=True)
    out = model(batch)
    aux = model.function_auxiliary.loss(out['logits'], out['function_aux'], batch, {})
    sum(aux.values()).backward()
    assert model.sequence.plm.weight.grad.abs().sum() > 0
    assert model.relations.factor[0][0].weight.grad.abs().sum() > 0
    absent = dict(batch, aux_mask=torch.zeros_like(batch['aux_mask']))
    primary = out['logits'].detach().requires_grad_()
    secondary = out['function_aux'].detach().requires_grad_()
    no_aux = sum(model.function_auxiliary.loss(primary, secondary, absent, {}).values())
    no_aux.backward()
    assert no_aux == 0
    assert primary.grad.count_nonzero() == secondary.grad.count_nonzero() == 0
    # Inference never requires targets from either task.
    model.eval()
    inputs = {k:v for k,v in batch.items() if k not in ('y','aux_y','aux_mask')}
    with torch.no_grad():
        assert torch.isfinite(model(inputs)['function_aux']).all()


def test_hierarchy_is_presence_not_exclusivity_or_function_count(tmp_path):
    config = auxiliary_config(tmp_path, 'fine39')
    source = MultimodalData(config['data']['root'], 'fine39')
    source.enable_function_auxiliary()
    from mfp_nova.function_auxiliary import FunctionAuxiliary
    head = FunctionAuxiliary(source.data.labels, source.statistics['function_auxiliary'], 32, 4, 0)
    # Two antibacterial subtypes and a generic antimicrobial label can coexist.
    fine = torch.full((1,39), -10.)
    fine[:,[1,2,16]] = 8
    major = torch.full((1,15), -10.)
    major[:,[0,4]] = 10
    batch = {'aux_mask':torch.tensor([True]), 'aux_y':torch.zeros(1,15)}
    assert head.loss(fine, major, batch, {})['function_hierarchy'] == 0
    major[:,0] = -8
    fine.requires_grad_(); major.requires_grad_()
    hierarchy = head.loss(fine, major, batch, {})['function_hierarchy']
    hierarchy.backward()
    assert hierarchy > 0 and major.grad[0,0] < 0 and fine.grad[0,1] > 0


@pytest.mark.parametrize('task', ['major15', 'fine39'])
def test_auxiliary_training_resume_and_independent_validation(tmp_path, task, monkeypatch):
    torch.set_num_threads(1)
    config = auxiliary_config(tmp_path, task)
    complete = research.train(copy.deepcopy(config), 'complete')
    original_format = research.format_epoch

    def stop(*args):
        if args[4]['epoch'] == 2:
            raise RuntimeError('epoch saved')
        return original_format(*args)

    with monkeypatch.context() as patch:
        patch.setattr(research, 'format_epoch', stop)
        with pytest.raises(RuntimeError, match='epoch saved'):
            research.train(copy.deepcopy(config), 'partial')
    resumed = research.train(copy.deepcopy(config), 'resumed', resume=tmp_path/'partial/last.pt')
    full = torch.load(tmp_path/'complete/last.pt', weights_only=True)
    continuation = torch.load(tmp_path/'resumed/last.pt', weights_only=True)
    for key in ('model_state', 'ema_state', 'best_model_state'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, continuation[key][name], rtol=0, atol=0)
    assert complete['validation'] == resumed['validation']
    measured = research.evaluate(tmp_path/'complete/best.pt', config['data']['root'], config['data']['features'],
                                 'val', 'cpu', tmp_path/'evaluation.json', batch_size=2)
    assert measured['metrics'] == complete['validation']
    history = [json.loads(line) for line in (tmp_path/'complete/history.jsonl').read_text().splitlines()]
    assert [h['loss_settings']['function_aux'] for h in history] == [.05,.1,.1]
    assert all(h['losses']['function_aux'] > 0 for h in history)
    assert all('consistency' not in h['losses'] for h in history)
    invalid = copy.deepcopy(config)
    invalid['train']['function_aux_weight'] = .2
    with pytest.raises(ValueError, match='changes the experiment'):
        research.train(invalid, 'invalid', resume=tmp_path/'partial/last.pt')


def test_auxiliary_cli_switch_and_legacy_model_rejection(monkeypatch):
    import sys
    from mfp_nova.cli import main
    captured = []
    monkeypatch.setattr(research, 'train', lambda config, *a, **k: captured.append(config))
    monkeypatch.setattr(sys, 'argv', ['mfp-nova','train','--model','trirelay','--task','fine39',
                                    '--seed','42','--function-aux-weight','0.1','--function-hierarchy-weight','0.02'])
    main()
    assert captured[0]['train']['function_aux_weight'] == .1
    assert captured[0]['train']['function_hierarchy_weight'] == .02
    assert captured[0]['data']['track'] == 'fine39'
    monkeypatch.setattr(sys, 'argv', ['mfp-nova','train','--function-aux-weight','0.1'])
    with pytest.raises(ValueError, match='TriRelay'):
        main()
