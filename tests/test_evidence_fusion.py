import copy
import json

import pytest
import torch

from mfp_nova import research
from mfp_nova.branch_evidence import export
from mfp_nova.cli import main
from test_consistency import fixture_config
from test_research import synthetic_batch, sample_statistics


@pytest.mark.parametrize('labels',[15,39])
@pytest.mark.parametrize('fusion',['normalized_competitive','normalized_residual'])
def test_normalized_fusion_in_both_reads_bounds_logits_and_keeps_gradients(labels,fusion,monkeypatch):
    batch=synthetic_batch(labels)
    args={'input_dim':12,'num_labels':labels,'vocab_size':10,'statistics':sample_statistics(labels),
          'dim':32,'heads':4,'dropout':0,'shared_private':True}
    baseline=research.make_model('trirelay',args).eval()
    model=research.make_model('trirelay',{**args,'evidence_fusion':fusion}).eval()
    model.load_state_dict(baseline.state_dict(),strict=True)
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in baseline.parameters())
    reads=[]
    read=model.evidence_partition.read
    def capture(*args,**kwargs):
        result=read(*args,**kwargs)
        reads.append(result)
        return result
    monkeypatch.setattr(model.evidence_partition,'read',capture)
    out=model(batch,first_readouts=True)
    assert len(reads)==2
    for _,mixed,fusion,private,gate,shared in reads:
        for value in (shared,private):
            assert (value.float().square().mean(-1)<=1+1e-5).all()
        competitive=fusion=='normalized_competitive'
        assert (mixed.float().square().mean(-1)<=(1 if competitive else 4)+1e-5).all()
        torch.testing.assert_close(mixed,((1-gate[...,None]) if competitive else 1)*shared+gate[...,None]*private)
        assert gate[1,:,1].count_nonzero()==fusion[1,:,1].count_nonzero()==0
    gate=out['first_private_gate'].transpose(1,2)
    expected=((1-gate)*out['shared_branch']+gate*out['unmixed_private_branch'] if fusion=='normalized_competitive' else
              out['shared_branch']+gate*(out['unmixed_private_branch']-model.classifier.bias))
    torch.testing.assert_close(out['first_mixed_branch'],expected,rtol=1e-6,atol=1e-6)
    if fusion=='normalized_competitive':
        lower=torch.minimum(out['shared_branch'],out['unmixed_private_branch'])
        upper=torch.maximum(out['shared_branch'],out['unmixed_private_branch'])
        assert (out['first_mixed_branch']>=lower-1e-6).all()
        assert (out['first_mixed_branch']<=upper+1e-6).all()
    settings={'branch_supervision':'shared_private','mask_ratio':0,'modality_dropout':0,
              'branch':.1,'direct':.2,'reconstruction':.05}
    total,terms=model.compute_loss(batch,settings)
    torch.testing.assert_close(total,terms['classification']+.2*terms['direct']+
                               .05*terms['branch_shared']+.05*terms['branch_private']+.05*terms['reconstruction'])
    total.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for prefix in ('sequence','geometry','properties','evidence_partition.shared','evidence_partition.private',
                   'evidence_partition.gates','feedback','relations'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for name,p in model.named_parameters() if name.startswith(prefix))
    missing=dict(batch,struct_mask=torch.zeros_like(batch['struct_mask']),phys_mask=torch.zeros_like(batch['phys_mask']))
    assert torch.isfinite(model.compute_loss(missing,settings)[0])
    reads.clear()
    model.private_evidence_enabled=False
    disabled=model(batch,first_readouts=True)
    for _,mixed,_,_,gate,shared in reads:
        assert gate.count_nonzero()==0
        torch.testing.assert_close(mixed,shared,rtol=0,atol=0)
    assert torch.isfinite(disabled['logits']).all()


def test_implicit_residual_is_exactly_unchanged_and_no_extra_state():
    batch=synthetic_batch()
    args={'input_dim':12,'num_labels':15,'vocab_size':10,'statistics':sample_statistics(15),
          'dim':32,'heads':4,'dropout':0,'shared_private':True}
    implicit=research.make_model('trirelay',args).eval()
    explicit=research.make_model('trirelay',{**args,'evidence_fusion':'residual'}).eval()
    explicit.load_state_dict(implicit.state_dict(),strict=True)
    with torch.inference_mode():
        a,b=implicit(batch,first_readouts=True),explicit(batch,first_readouts=True)
    for name in a:
        torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    gate=a['first_private_gate'].transpose(1,2)
    torch.testing.assert_close(a['first_mixed_branch'],a['shared_branch']+
                              gate*(a['unmixed_private_branch']-implicit.classifier.bias.detach()),rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('task',['major15','fine39'])
@pytest.mark.parametrize('fusion',['normalized_competitive','normalized_residual'])
def test_normalized_training_checkpoint_resume_and_readout_export(tmp_path,task,fusion,monkeypatch):
    torch.set_num_threads(1)
    config=fixture_config(tmp_path,task)
    config['model'].update(shared_private=True,evidence_fusion=fusion)
    config['loss'].update(direct=.2,branch=.1)
    config['train'].update(epochs=2,consistency_weight=0,branch_supervision='shared_private')
    config['train'].pop('consistency_ramp_epochs')
    result=research.train(copy.deepcopy(config),'complete')
    run=tmp_path/'complete'
    model,saved=research.load_model(run/'best.pt')
    assert model.evidence_partition.fusion==fusion
    assert saved['model_args']['evidence_fusion']==fusion
    meta=json.loads((run/'metadata.json').read_text())
    assert ('convex' if fusion=='normalized_competitive' else 'only mixing changes') in meta['evidence_fusion_scope']
    summary=export([run],tmp_path/'readouts',data_root=config['data']['root'],batch_size=2)
    observed=summary['comparisons'][0]
    assert observed['evidence_fusion']==fusion
    assert all(v==0 for v in observed['final_reproduction_deltas'].values() if v is not None)
    assert ('(1-g)' in observed['head_definitions']['mixed_*']) == (fusion=='normalized_competitive')
    assert 'normalized_private' in observed['head_definitions']['mixed_*']
    ablated=research.evaluate(run/'best.pt',config['data']['root'],config['data']['features'],
                             'val','cpu',tmp_path/'off.json',without_private_evidence=True)
    assert ('shared coefficient=1' if fusion=='normalized_competitive' else 'coefficient remains 1') in ablated['inference_ablation']['scope']
    if task=='fine39':
        return
    changed=copy.deepcopy(config)
    changed['model']['evidence_fusion']='normalized_residual' if fusion=='normalized_competitive' else 'normalized_competitive'
    with pytest.raises(ValueError,match='Evidence fusion differs'):
        research.train(changed,'invalid',resume=run/'last.pt')
    def interrupt(*args):
        if args[4]['epoch']==1:
            raise RuntimeError('saved epoch')
    with monkeypatch.context() as patch:
        patch.setattr(research,'format_epoch',interrupt)
        with pytest.raises(RuntimeError,match='saved epoch'):
            research.train(copy.deepcopy(config),'partial')
    resume=copy.deepcopy(config);resume['model'].pop('evidence_fusion')
    research.train(resume,'resumed',resume=tmp_path/'partial/last.pt')
    full=torch.load(run/'last.pt',weights_only=True)
    resumed=torch.load(tmp_path/'resumed/last.pt',weights_only=True)
    assert resumed['model_args']['evidence_fusion']==fusion
    for key in ('model_state','ema_state','best_model_state','online_best_state'):
        for name,value in full[key].items():
            torch.testing.assert_close(value,resumed[key][name],rtol=0,atol=0)
    assert result['threshold_fit_split']=='val'


@pytest.mark.parametrize('fusion',['normalized_competitive','normalized_residual'])
def test_evidence_fusion_cli_and_required_architecture(monkeypatch,fusion):
    captured={}
    monkeypatch.setattr(research,'train',lambda config,*args,**kwargs:captured.update(config=config))
    main(['train','--model','trirelay','--task','fine39','--shared-private',
          '--branch-supervision','shared_private','--evidence-fusion',fusion])
    assert captured['config']['model']['evidence_fusion']==fusion
    for name in ('geofactor','sequence'):
        with pytest.raises(ValueError,match='TriRelay model option'):
            main(['train','--model',name,'--evidence-fusion',fusion])
    with pytest.raises(ValueError,match='requires shared-private'):
        research.make_model('trirelay',{'input_dim':12,'num_labels':15,'vocab_size':10,
            'statistics':sample_statistics(15),'evidence_fusion':fusion})


def test_normalized_residual_changes_only_combination_before_feedback():
    batch=synthetic_batch()
    model=research.make_model('trirelay',{'input_dim':12,'num_labels':15,'vocab_size':10,
        'statistics':sample_statistics(15),'dim':32,'heads':4,'dropout':0,'shared_private':True,
        'evidence_fusion':'normalized_competitive'}).eval()
    with torch.inference_mode():
        competitive=model(batch,first_readouts=True)
        model.evidence_partition.fusion='normalized_residual'
        additive=model(batch,first_readouts=True)
        for key in ('shared_branch','unmixed_private_branch','first_private_gate'):
            torch.testing.assert_close(additive[key],competitive[key],rtol=0,atol=0)
        assert not torch.equal(additive['first_mixed_branch'],competitive['first_mixed_branch'])
        assert not torch.equal(additive['logits'],competitive['logits'])
