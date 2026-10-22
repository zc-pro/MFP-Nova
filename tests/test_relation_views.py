import copy
import json

import pytest
import torch

from mfp_nova import research
from mfp_nova.cli import main
from mfp_nova.consistency import bernoulli_consistency
from test_consistency import fixture_config
from test_research import sample_statistics, synthetic_batch


@pytest.mark.parametrize('labels', [15, 39])
def test_relation_view_complete_model_and_gradients(labels):
    batch = synthetic_batch(labels)
    model = research.make_model('trirelay', dict(input_dim=12, num_labels=labels, vocab_size=10,
        statistics=sample_statistics(labels), dim=32, heads=4, dropout=0,
        shared_private=True, evidence_fusion='normalized_competitive'))
    model.eval()
    baseline = model(batch)['logits']
    torch.testing.assert_close(model(batch, relation_view_dropout=0)['logits'], baseline, rtol=0, atol=0)
    out = model(batch, relation_view_dropout=.5)
    torch.testing.assert_close(out['logits'], baseline, rtol=0, atol=0)
    assert not torch.allclose(out['relation_view_logits'], out['logits'])
    corrupt = {k:v.clone() for k,v in batch.items()}
    corrupt['esm'][~batch['mask']] = 100
    corrupt['geometry'][~batch['struct_mask']] = 100
    torch.testing.assert_close(model(corrupt)['logits'], baseline)
    assert model(batch)['fusion'][1, :, 1].count_nonzero() == 0
    assert not torch.allclose(model(dict(batch, d=1-batch['d']))['logits'], baseline)
    # Drop every off-diagonal edge and factor: self edges keep attention finite.
    assert torch.isfinite(model.relations(torch.randn(3, labels, 32), True, 1.)).all()
    calls=[]
    hook=model.sequence.register_forward_hook(lambda *args:calls.append(1))
    model.train()
    loss, terms, output = model.compute_loss(batch, dict(relation_view=.1, relation_dropout=.5,
        branch_supervision='shared_private', modality_dropout=0, mask_ratio=0), return_output=True)
    hook.remove()
    assert len(calls)==1
    assert terms['relation_stability'] > 0 and terms['relation_anchor'] >= 0
    gradient=torch.autograd.grad(terms['relation_view'], model.relations.latent_membership, retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    loss.backward()
    for prefix in ('sequence','geometry','properties','interactions','feedback','relations','evidence_partition'):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for n,p in model.named_parameters() if n.startswith(prefix)), prefix
    # Existing positives cannot be contradicted by weak teacher targets.
    logits=torch.randn_like(batch['y'],requires_grad=True)
    agreement,_=bernoulli_consistency(logits, torch.full_like(logits,-20), batch['y'])
    grad=torch.autograd.grad(agreement,logits)[0]
    assert grad[batch['y'].bool()].count_nonzero()==0


@pytest.mark.parametrize('task', ['major15', 'fine39'])
def test_relation_view_training_evaluation_and_resume(tmp_path, monkeypatch, task):
    torch.set_num_threads(1)
    cfg=fixture_config(tmp_path, task)
    cfg['model'].update(shared_private=True,evidence_fusion='normalized_competitive')
    cfg['train'].update(epochs=2, consistency_weight=0, relation_view_weight=.1,
                        relation_dropout=.2, branch_supervision='shared_private')
    cfg['train'].pop('consistency_ramp_epochs')
    run=tmp_path/'complete'
    result=research.train(copy.deepcopy(cfg),'complete')
    history=[json.loads(s) for s in (run/'history.jsonl').read_text().splitlines()]
    assert all(h['losses']['relation_stability']>0 for h in history)
    assert history[0]['loss_settings']['relation_view']==pytest.approx(.02)
    measured=research.evaluate(run/'best.pt',cfg['data']['root'],cfg['data']['features'],
                              'val','cpu',tmp_path/'val.json',batch_size=2)
    assert measured['metrics']==result['validation']
    assert len(result['labels'])==(15 if task=='major15' else 39)
    if task=='fine39':
        return
    def interrupt(*args):
        if args[4]['epoch']==1:
            raise RuntimeError('saved epoch')
    with monkeypatch.context() as patch:
        patch.setattr(research,'format_epoch',interrupt)
        with pytest.raises(RuntimeError,match='saved epoch'):
            research.train(copy.deepcopy(cfg),'partial')
    research.train(copy.deepcopy(cfg),'resumed',resume=tmp_path/'partial/last.pt')
    full=torch.load(run/'last.pt',weights_only=True)
    resumed=torch.load(tmp_path/'resumed/last.pt',weights_only=True)
    for name in full['model_state']:
        torch.testing.assert_close(full['model_state'][name],resumed['model_state'][name],rtol=0,atol=0)
    changed=copy.deepcopy(cfg);changed['train']['relation_view_weight']=.2
    with pytest.raises(ValueError,match='Relation-view supervision changes'):
        research.train(changed,'invalid',resume=run/'last.pt')


def test_relation_view_cli_schedule(monkeypatch):
    captured={}
    monkeypatch.setattr(research,'train',lambda config,*args,**kwargs:captured.update(config=config))
    main(['train','--model','trirelay','--relation-view-weight','.1','--relation-dropout','.2'])
    cfg=captured['config']
    assert cfg['train']['relation_view_weight']==.1
    assert [research.epoch_loss_settings(cfg,e)['relation_view'] for e in (1,5,6,10)]==pytest.approx([0,0,.02,.1])
    with pytest.raises(ValueError,match='TriRelay training option'):
        main(['train','--model','geofactor','--relation-view-weight','.1'])
