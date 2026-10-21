import copy
import json

import pytest
import torch

from mfp_nova import research
from mfp_nova.cli import main
from mfp_nova.research_layers import observed_label_loss
from test_consistency import fixture_config
from test_research import sample_statistics, synthetic_batch


@pytest.mark.parametrize('labels',[15,39])
def test_unmixed_first_read_supervision_budget_isolation_and_gradients(labels):
    batch=synthetic_batch(labels)
    model=research.make_model('trirelay',{'input_dim':12,'num_labels':labels,'vocab_size':10,
        'statistics':sample_statistics(labels),'dim':32,'heads':4,'dropout':0,'shared_private':True}).eval()
    parameter_count=sum(p.numel() for p in model.parameters())
    ordinary=model(batch)
    separate=model(batch,unmixed_branches=True)
    torch.testing.assert_close(ordinary['logits'],separate['logits'],rtol=0,atol=0)
    assert separate['shared_branch'].shape==separate['unmixed_private_branch'].shape==(3,3,labels)
    assert sum(p.numel() for p in model.parameters())==parameter_count
    # A pure shared read must not depend on private projections/readers/gates.
    with torch.no_grad():
        for p in model.evidence_partition.private.parameters():
            p.add_(.3)
    changed=model(batch,unmixed_branches=True)
    torch.testing.assert_close(changed['shared_branch'],separate['shared_branch'],rtol=0,atol=0)
    assert not torch.equal(changed['unmixed_private_branch'],separate['unmixed_private_branch'])
    settings={'branch_supervision':'shared_private','mask_ratio':0,'modality_dropout':0,
              'direct':.2,'branch':.1,'reconstruction':.05}
    total,terms,out=model.compute_loss(batch,settings,return_output=True)
    for name,key in [('shared','shared_branch'),('private','unmixed_private_branch')]:
        expected=torch.stack([observed_label_loss(out[key][out['available'][:,m],m],
            batch['y'][out['available'][:,m]],model.statistics,.3) for m in range(3)]).mean()
        torch.testing.assert_close(terms['branch_'+name],expected)
    torch.testing.assert_close(terms['branch'],.5*(terms['branch_shared']+terms['branch_private']))
    torch.testing.assert_close(total,terms['classification']+.2*terms['direct']+
                               .05*terms['branch_shared']+.05*terms['branch_private']+.05*terms['reconstruction'])
    private_parameters=list(model.evidence_partition.private.parameters())+list(model.evidence_partition.readers.parameters())+list(model.evidence_partition.gates.parameters())
    dependencies=torch.autograd.grad(terms['branch_shared'],private_parameters,allow_unused=True,retain_graph=True)
    assert all(gradient is None for gradient in dependencies)
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for prefix in ('evidence_partition.shared','evidence_partition.private','evidence_partition.gates','feedback','relations'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for name,p in model.named_parameters() if name.startswith(prefix))
    missing=dict(batch,struct_mask=torch.zeros_like(batch['struct_mask']),phys_mask=torch.zeros_like(batch['phys_mask']))
    assert torch.isfinite(model.compute_loss(missing,settings)[0])
    legacy={**settings,'branch_supervision':'mixed_private'}
    implicit=dict(legacy);implicit.pop('branch_supervision')
    a=model.compute_loss(batch,legacy)[0]
    b=model.compute_loss(batch,implicit)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('task',['major15','fine39'])
def test_symmetric_training_ema_resume_and_independent_val(tmp_path,task,monkeypatch):
    torch.set_num_threads(1)
    config=fixture_config(tmp_path,task)
    config['model']['shared_private']=True
    config['loss'].update(branch=.1,direct=.2)
    config['train'].update(epochs=2,aux_warmup_epochs=2,consistency_weight=0,branch_supervision='shared_private')
    config['train'].pop('consistency_ramp_epochs')
    result=research.train(copy.deepcopy(config),'complete')
    hist=[json.loads(line) for line in (tmp_path/'complete/history.jsonl').read_text().splitlines()]
    for epoch,h in enumerate(hist,1):
        assert h['loss_settings']['branch_supervision']=='shared_private'
        assert h['loss_settings']['branch']==pytest.approx(.1*epoch/2)
        assert h['losses']['branch']==pytest.approx(.5*(h['losses']['branch_shared']+h['losses']['branch_private']))
    report=research.evaluate(tmp_path/'complete/best.pt',config['data']['root'],config['data']['features'],
                             'val','cpu',tmp_path/'val.json',batch_size=2)
    assert report['metrics']==result['validation']
    if task=='fine39':
        return
    original_format=research.format_epoch
    def interrupt(*args):
        if args[4]['epoch']==1:
            raise RuntimeError('saved epoch')
        return original_format(*args)
    with monkeypatch.context() as patch:
        patch.setattr(research,'format_epoch',interrupt)
        with pytest.raises(RuntimeError,match='saved epoch'):
            research.train(copy.deepcopy(config),'partial')
    resume=copy.deepcopy(config)
    resume['train'].pop('branch_supervision')
    research.train(resume,'resumed',resume=tmp_path/'partial/last.pt')
    full=torch.load(tmp_path/'complete/last.pt',weights_only=True)
    resumed=torch.load(tmp_path/'resumed/last.pt',weights_only=True)
    for key in ('model_state','ema_state','best_model_state','online_best_state'):
        for name,value in full[key].items():
            torch.testing.assert_close(value,resumed[key][name],rtol=0,atol=0)
    changed=copy.deepcopy(config)
    changed['train']['branch_supervision']='mixed_private'
    with pytest.raises(ValueError,match='Branch supervision changes'):
        research.train(changed,'invalid',resume=tmp_path/'complete/last.pt')


def test_branch_supervision_cli_and_required_architecture(tmp_path,monkeypatch):
    captured={}
    with monkeypatch.context() as patch:
        patch.setattr(research,'train',lambda config,*args,**kwargs:captured.update(config=config))
        main(['train','--model','trirelay','--task','fine39','--seed','42','--shared-private',
              '--branch-supervision','shared_private'])
    assert captured['config']['train']['branch_supervision']=='shared_private'
    assert captured['config']['model']['shared_private'] is True
    config=captured['config']
    config['model']['shared_private']=False
    with pytest.raises(ValueError,match='requires shared-private TriRelay'):
        research.train(config,'invalid')
    with pytest.raises(ValueError,match='TriRelay training option'):
        main(['train','--model','geofactor','--branch-supervision','shared_private'])
