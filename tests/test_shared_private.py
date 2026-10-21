import copy
import json

import pytest
import torch

from mfp_nova import research
from mfp_nova.cli import main
from test_consistency import fixture_config
from test_research import sample_statistics, synthetic_batch


@pytest.mark.parametrize('labels', [15, 39])
def test_shared_private_evidence_masks_and_gradients(labels):
    batch = synthetic_batch(labels)
    args = {'input_dim': 12, 'num_labels': labels, 'vocab_size': 10,
            'statistics': sample_statistics(labels), 'dim': 32, 'heads': 4, 'dropout': 0}
    model = research.make_model('trirelay', dict(args, shared_private=True)).eval()
    output = model(batch)
    assert output['logits'].shape == (3, labels)
    assert output['private_branch'].shape == (3, 3, labels)
    assert output['private_gate'].shape == (3, labels, 3)
    assert output['private_gate'][1, :, 1].count_nonzero() == 0
    assert output['fusion'][1, :, 1].count_nonzero() == 0

    corrupt = {k: v.clone() for k, v in batch.items()}
    corrupt['esm'][~batch['mask']] = 100
    corrupt['geometry'][~batch['struct_mask']] = 100
    corrupt['coords'][~batch['struct_mask']] = 100
    torch.testing.assert_close(model(corrupt)['logits'], output['logits'])
    missing = dict(batch, struct_mask=torch.zeros_like(batch['struct_mask']),
                   confidence_known=torch.zeros_like(batch['confidence_known']),
                   phys_mask=torch.zeros_like(batch['phys_mask']))
    missing_output = model(missing)
    assert torch.isfinite(missing_output['logits']).all()
    assert missing_output['private_gate'][:, :, 1:].count_nonzero() == 0
    assert missing_output['fusion'][:, :, 1:].count_nonzero() == 0
    assert not torch.allclose(model(dict(batch, d=1-batch['d']))['logits'], output['logits'])

    # The bypass is not rewritten by feedback and can change final predictions.
    private_reads = []
    handle = model.evidence_partition.readers[0].register_forward_pre_hook(
        lambda module, inputs: private_reads.append(inputs[1].detach().clone()))
    model(batch)
    handle.remove()
    assert len(private_reads) == 2
    torch.testing.assert_close(private_reads[0], private_reads[1], rtol=0, atol=0)
    handle = model.evidence_partition.readers[0].register_forward_hook(
        lambda module, inputs, value: value*0)
    no_private_sequence = model(batch)
    handle.remove()
    assert not torch.allclose(no_private_sequence['logits'], output['logits'])

    settings = {'modality_dropout': 0, 'mask_ratio': 0, 'unlabeled_weight': .3}
    loss, terms, measured = model.compute_loss(batch, settings, return_output=True)
    from mfp_nova.research_layers import observed_label_loss
    branches = [observed_label_loss(measured[key][measured['available'][:, m], m],
                                   batch['y'][measured['available'][:, m]], args['statistics'], .3)
                for key in ('branch', 'private_branch') for m in range(3)]
    torch.testing.assert_close(terms['branch'], torch.stack(branches).mean())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for prefix in ('evidence_partition.shared', 'evidence_partition.private',
                   'evidence_partition.readers', 'evidence_partition.gates',
                   'interactions', 'feedback', 'refine', 'relations'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for name, p in model.named_parameters() if name.startswith(prefix)), prefix


def test_disabled_partition_preserves_old_state_and_forward():
    batch = synthetic_batch()
    args = {'input_dim': 12, 'num_labels': 15, 'vocab_size': 10, 'statistics': sample_statistics(15),
            'dim': 32, 'heads': 4, 'dropout': 0}
    torch.manual_seed(42)
    original = research.make_model('trirelay', args).eval()
    torch.manual_seed(42)
    disabled = research.make_model('trirelay', dict(args, shared_private=False)).eval()
    disabled.load_state_dict(original.state_dict(), strict=True)
    assert not hasattr(disabled, 'evidence_partition')
    torch.testing.assert_close(original(batch)['logits'], disabled(batch)['logits'], rtol=0, atol=0)
    torch.manual_seed(42)
    enabled = research.make_model('trirelay', dict(args, shared_private=True))
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, enabled.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize('task', ['major15', 'fine39'])
def test_shared_private_ema_training_resume_and_independent_val(tmp_path, monkeypatch, task):
    torch.set_num_threads(1)
    config = fixture_config(tmp_path, task)
    config['model']['shared_private'] = True
    config['train']['consistency_weight'] = 0
    config['train'].pop('consistency_ramp_epochs')
    config['train']['epochs'] = 2
    result = research.train(copy.deepcopy(config), 'complete')
    saved = torch.load(tmp_path/'complete/last.pt', weights_only=True)
    assert saved['model_args']['shared_private'] is True
    assert saved['ema_state']['n_averaged'].item() > 0
    measured = research.evaluate(tmp_path/'complete/best.pt', config['data']['root'], config['data']['features'],
                                 'val', 'cpu', tmp_path/'val.json', predictions=tmp_path/'val_predictions.npz', batch_size=2)
    assert measured['metrics'] == result['validation']
    history = [json.loads(line) for line in (tmp_path/'complete/history.jsonl').read_text().splitlines()]
    assert len(history) == 2 and all(h['losses']['branch'] > 0 for h in history)
    assert len(result['labels']) == (15 if task == 'major15' else 39)
    if task == 'fine39':
        return

    real_format = research.format_epoch

    def interrupt(*args):
        if args[4]['epoch'] == 1:
            raise RuntimeError('saved epoch interruption')
        return real_format(*args)

    with monkeypatch.context() as patch:
        patch.setattr(research, 'format_epoch', interrupt)
        with pytest.raises(RuntimeError, match='saved epoch interruption'):
            research.train(copy.deepcopy(config), 'partial')
    resume_config = copy.deepcopy(config)
    resume_config['model'].pop('shared_private')  # Recover architecture from its own checkpoint.
    research.train(resume_config, 'resumed', resume=tmp_path/'partial/last.pt')
    resumed = torch.load(tmp_path/'resumed/last.pt', weights_only=True)
    for key in ('model_state', 'ema_state', 'best_model_state', 'online_best_state'):
        for name, value in saved[key].items():
            torch.testing.assert_close(value, resumed[key][name], rtol=0, atol=0)
    different = copy.deepcopy(config)
    different['model']['shared_private'] = False
    with pytest.raises(ValueError, match='Shared/private architecture'):
        research.train(different, 'invalid', resume=tmp_path/'complete/last.pt')


@pytest.mark.parametrize('flag,enabled', [('--shared-private', True), ('--no-shared-private', False)])
def test_shared_private_cli(flag, enabled, monkeypatch):
    captured = {}
    monkeypatch.setattr(research, 'train', lambda config, name, resume=None: captured.update(config=config))
    main(['train', '--model', 'trirelay', '--task', 'fine39', '--seed', '42', flag])
    assert captured['config']['model']['shared_private'] is enabled


@pytest.mark.parametrize('model', ['sequence', 'geofactor', 'protoroute'])
def test_shared_private_is_trirelay_specific(model):
    with pytest.raises(ValueError, match='TriRelay model option'):
        main(['train', '--model', model, '--shared-private'])
