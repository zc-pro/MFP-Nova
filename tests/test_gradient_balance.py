import copy
import json

import pytest
import torch
from torch import nn

from mfp_nova.gradient_balance import backward_reconstruction
from mfp_nova import research
from test_consistency import fixture_config


@pytest.mark.parametrize('aux', [[-1.,2.],[10.,10.],[.2,.1]])
def test_shared_projection_bound_and_private_gradients(aux):
    shared=nn.Parameter(torch.zeros(2))
    main_private=nn.Parameter(torch.zeros(1))
    aux_private=nn.Parameter(torch.zeros(1))
    main=shared[0]+3*main_private.sum()
    auxiliary=(shared*torch.tensor(aux)).sum()+4*aux_private.sum()
    record=backward_reconstruction(main,auxiliary,[shared,main_private,aux_private])
    primary=torch.tensor([1.,0.])
    applied=shared.grad-primary
    assert torch.dot(applied,primary)>=-1e-6
    assert applied.norm()<=primary.norm()+1e-6
    assert main_private.grad.item()==3 and aux_private.grad.item()==4
    assert record['conflict']==(aux[0]<0)
    assert record['applied_aux_main_ratio']<=1
    if aux==[.2,.1]:
        torch.testing.assert_close(shared.grad,primary+torch.tensor(aux))


def test_zero_main_gradient_and_scaled_accumulation():
    p=nn.Parameter(torch.ones(2))
    record=backward_reconstruction(p.sum()*0,p.sum(),[p])
    assert p.grad.count_nonzero()==0
    assert record['cosine'] is None and record['applied_aux_main_ratio']==0
    p.grad=None
    for _ in range(2):
        backward_reconstruction(p[0]*64/2,(-p[0]+2*p[1])*64/2,[p])
    torch.testing.assert_close(p.grad,torch.tensor([64.,64.]))


def test_nonfinite_scaled_gradients_are_not_silently_zeroed():
    p=nn.Parameter(torch.ones(2))
    record=backward_reconstruction(p.sum(),p.sum()*float('inf'),[p])
    assert not record['finite'] and record['cosine'] is None
    assert not torch.isfinite(p.grad).all()


def gradient_config(tmp_path,task):
    config=fixture_config(tmp_path,task)
    config['train'].pop('consistency_weight')
    config['train'].pop('consistency_ramp_epochs')
    config['train'].update(reconstruction_gradient='project',reconstruction_grad_ratio=1.,accumulation_steps=2)
    return config


@pytest.mark.parametrize('task',['major15','fine39'])
def test_projection_training_resume_and_independent_val(tmp_path,task,monkeypatch):
    torch.set_num_threads(1)
    config=gradient_config(tmp_path,task)
    result=research.train(copy.deepcopy(config),'complete')
    real_format=research.format_epoch

    def stop(*args):
        if args[4]['epoch']==2:
            raise RuntimeError('saved epoch')
        return real_format(*args)

    with monkeypatch.context() as patch:
        patch.setattr(research,'format_epoch',stop)
        with pytest.raises(RuntimeError,match='saved epoch'):
            research.train(copy.deepcopy(config),'partial')
    # Omitted new arguments must recover the original optimization protocol.
    resume_config=copy.deepcopy(config)
    resume_config['train'].pop('reconstruction_gradient')
    resume_config['train'].pop('reconstruction_grad_ratio')
    resumed=research.train(resume_config,'resumed',resume=tmp_path/'partial/last.pt')
    full=torch.load(tmp_path/'complete/last.pt',weights_only=True)
    continuation=torch.load(tmp_path/'resumed/last.pt',weights_only=True)
    for key in ('model_state','ema_state','best_model_state','online_best_state'):
        for name,value in full[key].items():
            torch.testing.assert_close(value,continuation[key][name],rtol=0,atol=0)
    assert result['validation']==resumed['validation']
    score=research.evaluate(tmp_path/'complete/best.pt',config['data']['root'],config['data']['features'],
                            'val','cpu',tmp_path/'val.json',batch_size=2)
    assert score['metrics']==result['validation']
    histories=[json.loads(l) for l in (tmp_path/'complete/history.jsonl').read_text().splitlines()]
    for h in histories:
        g=h['reconstruction_gradient']
        assert g['batches']==2 and g['finite_batches']==2
        assert g['mean_applied_aux_main_ratio']<=1+1e-6
        assert all(torch.isfinite(torch.tensor(v)) for v in h['losses'].values())
    changed=copy.deepcopy(config)
    changed['train']['reconstruction_gradient']='sum'
    with pytest.raises(ValueError,match='changes the experiment'):
        research.train(changed,'changed',resume=tmp_path/'partial/last.pt')
    assert not any(k.startswith('function_auxiliary.') for k in full['model_state'])


def test_projection_cli_and_protocol_separation(tmp_path,monkeypatch):
    import sys
    import yaml
    from mfp_nova.cli import main
    from mfp_nova.experiments import PROJECT,discover_runs
    captured=[]
    monkeypatch.setattr(research,'train',lambda config,*a,**k:captured.append(config))
    monkeypatch.setattr(sys,'argv',['mfp-nova','train','--model','trirelay','--task','fine39',
        '--reconstruction-gradient','project','--reconstruction-grad-ratio','1'])
    main()
    assert captured[0]['train']['reconstruction_gradient']=='project'
    config=yaml.safe_load((PROJECT/'configs/trirelay.yaml').read_text(encoding='utf-8'))
    config['train'].update(batch_size=64,precision='fp32',reconstruction_gradient='project')
    directory=tmp_path/'trirelay_major15_gradient_seed42'
    directory.mkdir()
    metadata={'model':'trirelay','task':'major15','seed':42,'config':config,
              'encoder':{'name':'facebook/esm2_t33_650M_UR50D'}}
    (directory/'metadata.json').write_text(json.dumps(metadata))
    assert discover_runs(tmp_path)==[]
