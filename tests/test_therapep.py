import copy
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mfp_nova import research
from mfp_nova.data import PeptideData
from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from mfp_nova.research_layers import unlabeled_weights
from mfp_nova.therapep import (DATASET, LABELS, configure_task, download_data,
                               published_metrics, read_data)


@pytest.fixture
def published_fixture(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path/'published'
    root.mkdir()
    positive = [0]*49
    positive[18] = positive[48] = 1
    other = [0]*49
    other[36] = other[37] = other[48] = 1
    for split, rows in {'train': [('ACDE', positive), ('FGHIK', other), ('RRKK', [0]*49), ('LLLL', [0]*49)],
                        'val': [('ACDE', positive), ('RRKK', [0]*49)],
                        'test': [('WWKK', positive), ('VVVV', [0]*49)]}.items():
        with (root/f'{split}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['Sequence','Label encoding','is_therapeutic','source','Length','negative_type'])
            writer.writeheader()
            for seq,y in rows:
                writer.writerow({'Sequence':seq,'Label encoding':json.dumps(y),'is_therapeutic':str(bool(y[-1])),
                                 'source':'TheraPepDB' if y[-1] else 'synthetic_uniform',
                                 'Length':len(seq),'negative_type':'' if y[-1] else 'uniform'})
    return root


def cache_fixture(root, output, width, name):
    data=PeptideData(root,'therapep49')
    rng=np.random.default_rng(42)
    for split in ('train','val','test'):
        # Physical order is deliberately reversed; row IDs determine alignment.
        indices=data.indices(split)[::-1]
        directory=output/split
        directory.mkdir(parents=True)
        lengths=np.array([len(data.sequences[i]) for i in indices])
        np.save(directory/'row_ids.npy',data.rows[indices])
        np.save(directory/'ids.npy',data.ids[indices])
        np.save(directory/'sequences.npy',data.sequences[indices])
        np.save(directory/'offsets.npy',np.r_[0,np.cumsum(lengths)])
        np.save(directory/'residues.npy',rng.normal(size=(int(lengths.sum()),width)).astype(np.float16))
        (directory/'metadata.json').write_text(json.dumps({
            'name':name,'requested_revision':'main','resolved_revision':'fixture','max_length':80,
            'dataset':DATASET,'input_dim':width,'extraction_dtype':'float32','storage_dtype':'float16'}))


def test_published_membership_labels_and_duplicates(published_fixture):
    data=PeptideData(published_fixture,'therapep49')
    assert data.labels==LABELS and data.y.shape==(8,49)
    assert data.labels[36:38]==['Growth_Regeneration','Growth_regeneration']
    assert [len(data.indices(s)) for s in ('train','val','test')]==[4,2,2]
    assert data.ids[:2].tolist()==['therapep:train:0','therapep:train:1']
    assert data.clusters[0]==data.clusters[4]  # Preserve author membership, even for shared sequences.
    assert data.y[1,36:38].tolist()==[1,1]    # Do not merge aliases or propagate/rewrite labels.
    with pytest.raises(ValueError,match='no augmentation'):
        data.indices('train',augment=True)


def test_assigned_background_and_global_flag_are_not_weak_unlabeled_terms():
    y=torch.zeros(2,49)
    y[0,-1]=1
    weights=unlabeled_weights(y,{'labels':LABELS,'therapep_background_index':48},.3)
    torch.testing.assert_close(weights[0,:48],torch.full((48,),.3))
    assert weights[0,-1]==1 and (weights[1]==1).all()
    original=unlabeled_weights(y,{'labels':LABELS},.3)
    assert (original==.3).all()


def test_configuration_preserves_existing_tasks_and_custom_paths():
    config=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/theraset.yaml').read_text())
    original=copy.deepcopy(config)
    configure_task(config,'major15')
    assert config==original
    configure_task(config,'therapep49')
    assert config['data']['root']=='data/therapep_ai'
    assert config['data']['features']=='features/therapep_ai/esm2_650m'
    assert config['data']['peptide_features']=='features/therapep_ai/esm2_peptide_150m'
    assert config['train']['metric_suite']=='micro_f1'
    assert not config['model']['property_evidence'] and not config['model']['structure_evidence']
    config['data'].update(root='custom/data',features='custom/features')
    configure_task(config,'therapep49')
    assert config['data']['root']=='custom/data' and config['data']['features']=='custom/features'


def test_train_only_statistics_and_uniform_modality_scope(published_fixture):
    source=MultimodalData(published_fixture,'therapep49')
    assert source.statistics['train_rows']==4
    assert source.statistics['labels']['support'][18]==1
    assert source.statistics['labels']['support'][-1]==2
    assert source.statistics['function_projection'] is None
    assert not any(source.statistics['phys']['active'])
    for i in (0,2):
        sample=source.sample(i,torch.randn(4,12))
        assert not sample['phys_mask'].any() and not sample['struct_mask'].any()
        assert not sample['struct_global_mask'].any() and not sample['residue_chem'].any()
        assert sample['quality'][1:].count_nonzero()==0


def test_published_metric_views_keep_function_and_flag_scores_separate():
    y=np.zeros((2,49));y[0,18]=y[0,-1]=1
    p=np.full((2,49),.1);p[0,18]=p[0,-1]=.9
    p[1,0]=.9
    report=published_metrics(y,p,.5)
    assert report['positive_all49']['micro_f1']==1
    assert report['positive_functions48']['micro_f1']==1
    assert report['therapeutic_flag']['f1']==1
    assert report['synthetic_background']['any_function_fpr']==1
    assert report['synthetic_background']['therapeutic_flag_fpr']==0
    per_label=published_metrics(y,p,[.5]*49)
    assert per_label==report
    assert report['all49_labelwise_accuracy']==pytest.approx(97/98)


def test_download_reuses_csvs_and_preserves_original_bytes(published_fixture,tmp_path,monkeypatch):
    from mfp_nova import therapep
    calls=[]
    def open_url(url):
        split=url.rsplit('/',1)[-1]
        calls.append(split)
        return io.BytesIO((published_fixture/split).read_bytes())
    monkeypatch.setattr(therapep.urllib.request,'urlopen',open_url)
    target=tmp_path/'download'
    report=download_data(target)
    assert report['splits']['train']['rows']==4
    assert not list(target.glob('*.part'))
    assert (target/'train.csv').read_bytes()==(published_fixture/'train.csv').read_bytes()
    download_data(target)
    assert calls==['train.csv','val.csv','test.csv']


def test_cli_task_defaults_and_explicit_overrides(monkeypatch):
    from mfp_nova.cli import main
    captured=[]
    monkeypatch.setattr(research,'train',lambda config,name,resume=None:captured.append((config,name,resume)))
    main(['train','--model','theraset','--task','therapep49','--data-root','custom/data','--feature-root','custom/cache'])
    config,name,_=captured[0]
    assert name=='theraset_therapep49_seed42'
    assert config['data']['track']=='therapep49'
    assert config['data']['root']=='custom/data' and config['data']['features']=='custom/cache'
    assert config['train']['metric_suite']=='micro_f1'


def test_feature_extraction_uses_published_members_and_separate_cache(published_fixture,tmp_path,monkeypatch):
    from mfp_nova import encoder
    class FakeEncoder:
        def __init__(self,*args):
            self.metadata={'name':'fixture','input_dim':12,'requested_revision':'main','resolved_revision':'fixture',
                           'max_length':80,'storage_dtype':'float16','extraction_dtype':'float32'}
        def encode(self,sequences):
            mask=torch.arange(max(map(len,sequences)))[None]<torch.tensor(list(map(len,sequences)))[:,None]
            return torch.ones(len(sequences),mask.shape[1],12),mask
    monkeypatch.setattr(encoder,'FrozenEncoder',FakeEncoder)
    output=tmp_path/'extracted'
    config={'data':{'root':str(published_fixture),'track':'therapep49','features':str(output),'augment':False},
            'encoder':{'name':'fixture','revision':'main','max_length':80,'batch_size':2}}
    encoder.prepare_features(config,['train','val'],'cpu')
    assert np.load(output/'train/row_ids.npy').tolist()==[0,1,2,3]
    assert np.load(output/'val/row_ids.npy').tolist()==[4,5]
    assert not (output/'test').exists()
    assert json.loads((output/'train/metadata.json').read_text())['dataset']==DATASET


def test_training_resume_evaluation_and_prediction(published_fixture,tmp_path,monkeypatch):
    features,peptide=tmp_path/'features',tmp_path/'peptide'
    cache_fixture(published_fixture,features,12,'fixture')
    cache_fixture(published_fixture,peptide,6,'peptide_fixture')
    config={'model_name':'theraset','data':{'root':str(published_fixture),'track':'therapep49',
            'features':str(features),'peptide_features':str(peptide),'augment':False},
            'encoder':{'name':'fixture','revision':'main','max_length':80},
            'peptide_encoder':{'name':'peptide_fixture','revision':'main','max_length':80},
            'model':{'dim':32,'heads':4,'dropout':.1,'cnn_channels':8,'pair_dim':8,'decoder_depth':1,
                     'property_evidence':True,'structure_evidence':True,'chemical_evidence':True},
            'loss':{'rdrop':.1,'direct':.2,'evidence':.1},
            'train':{'device':'cpu','seed':42,'epochs':1,'batch_size':2,'accumulation_steps':1,'num_workers':0,
                     'learning_rate':.001,'weight_decay':0,'patience':0,'warmup_epochs':0,
                     'rdrop_start_epoch':1,'rdrop_ramp_epochs':1,'ema_decay':.9,'ema_start_epoch':1,
                     'save_selection_candidates':True,'export_validation_predictions':True,'output':str(tmp_path)}}
    configure_task(config,'therapep49')
    result=research.train(config,'first')
    assert result['threshold']==.5 and result['threshold_fit_split'] is None
    assert result['therapep_validation']['synthetic_background']['rows']==1
    assert 'mfp_detection' not in result['validation']
    assert 'alternate_selection' not in result
    checkpoint=tmp_path/'first/best.pt'
    report=research.evaluate(checkpoint,published_fixture,features,'val','cpu',tmp_path/'eval.json')
    assert report['metrics']==result['validation']
    assert report['therapep']['positive_all49']['label_positive_support'][-1]==1
    saved=torch.load(checkpoint,weights_only=True)
    assert saved['statistics']['train_rows']==4
    assert saved['model_args']['num_labels']==49
    source=MultimodalData(published_fixture,'therapep49',saved['statistics'])
    dataset=MultimodalPeptides(source,features,'val',peptide_feature_root=peptide)
    batch=collate_multimodal([dataset[0],dataset[1]])
    np.savez(tmp_path/'inputs.npz',**{k:v.numpy() for k,v in batch.items() if k!='y'})
    assert len(research.predict_inputs(checkpoint,tmp_path/'inputs.npz'))==2
    class FakeEncoder:
        def __init__(self,name,*args): self.width=6 if name=='peptide_fixture' else 12
        def encode(self,sequences):
            mask=torch.arange(max(map(len,sequences)))[None]<torch.tensor(list(map(len,sequences)))[:,None]
            return torch.ones(len(sequences),mask.shape[1],self.width),mask
    monkeypatch.setattr('mfp_nova.encoder.FrozenEncoder',FakeEncoder)
    assert len(research.predict_sequences(checkpoint,['ACDE','RRKK']))==2
    config['train']['epochs']=2
    continued=research.train(config,'continued',resume=tmp_path/'first/last.pt')
    assert continued['epochs_completed']==2


def test_malformed_label_vector_fails_instead_of_silent_zero_labels(published_fixture):
    path=published_fixture/'train.csv'
    text=path.read_text()
    text=text.replace(json.dumps([0]*49),json.dumps([0]*48),1)
    path.write_text(text)
    with pytest.raises(ValueError,match='49-bit'):
        read_data(published_fixture)


def test_multiseed_explicit_track_does_not_expand_default_plan(tmp_path):
    from mfp_nova import experiments
    assert experiments.TASKS==('major15','fine39')
    plan=experiments.make_plan(tmp_path,models=('theraset',),tasks=('therapep49',),seeds=(42,),running={})
    assert plan['total']==1
    job=plan['jobs'][0]
    assert job['task']=='therapep49' and job['status']=='not_started'
    assert job['train_settings']['metric_suite']=='micro_f1'
    assert 'micro_f1' in job['run_name']


@pytest.mark.parametrize('name',['trirelay','geofactor','protoroute','helixset'])
def test_other_shared_models_accept_published_sequence_only_batches(published_fixture,name):
    source=MultimodalData(published_fixture,'therapep49')
    batch=collate_multimodal([source.sample(i,torch.randn(len(source.data.sequences[i]),12)) for i in (0,2)])
    model=research.make_model(name,{'input_dim':12,'num_labels':49,'vocab_size':2,
                                   'statistics':source.statistics['labels'],'dim':32,'heads':4,'dropout':0})
    loss,_=model.compute_loss(batch,{'mask_ratio':0,'modality_dropout':0})
    loss.backward()
    assert torch.isfinite(loss)
    assert model(batch)['logits'].shape==(2,49)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_gpu_check_defaults_select_the_published_cache_before_inspection(tmp_path,monkeypatch):
    from mfp_nova import runtime_check
    calls=[]
    def inspect(root,device):
        calls.append((root,device))
        return {'gpu':{'other_compute_processes':[['uuid','1','python','1000']]}}
    monkeypatch.setattr(runtime_check,'inspect',inspect)
    with pytest.raises(RuntimeError,match='other compute jobs'):
        runtime_check.run(['theraset'],['therapep49'],'features/esm2_650m','cuda:0',tmp_path/'probe')
    assert calls==[('features/therapep_ai/esm2_650m','cuda:0')]
