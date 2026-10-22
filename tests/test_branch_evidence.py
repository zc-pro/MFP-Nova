import json
import zipfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from mfp_nova import research
from mfp_nova.branch_evidence import HEADS, collect_readouts, export, summarize_heads
from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from test_consistency import fixture_config
from test_research import sample_statistics, synthetic_batch


@pytest.mark.parametrize('labels', [15,39])
def test_first_readouts_preserve_prediction_and_capture_pre_feedback_evidence(labels):
    batch = synthetic_batch(labels)
    model = research.make_model('trirelay', {'input_dim':12,'num_labels':labels,'vocab_size':10,
        'statistics':sample_statistics(labels),'dim':32,'heads':4,'dropout':0,'shared_private':True}).eval()
    original_keys = set(model.state_dict())
    with torch.inference_mode():
        ordinary = model(batch)
        observed = model(batch, first_readouts=True)
        for key in ordinary:
            torch.testing.assert_close(ordinary[key],observed[key],rtol=0,atol=0)
        assert observed['first_mixed_branch'].shape==(3,3,labels)
        assert observed['first_private_gate'].shape==observed['first_fusion'].shape==(3,labels,3)
        torch.testing.assert_close(observed['first_mixed_branch'], observed['shared_branch']+
            observed['first_private_gate'].transpose(1,2)*(observed['unmixed_private_branch']-model.classifier.bias),
            rtol=1e-6,atol=1e-6)
        assert observed['first_private_gate'][1,:,1].count_nonzero()==0
        assert observed['first_fusion'][1,:,1].count_nonzero()==0
        # Later feedback changes final predictions, never the saved first read.
        for p in model.feedback.parameters():
            p.add_(.2)
        changed = model(batch, first_readouts=True)
        for key in ('shared_branch','unmixed_private_branch','first_mixed_branch','first_fusion','first_private_gate','direct'):
            torch.testing.assert_close(observed[key],changed[key],rtol=0,atol=0)
        assert not torch.equal(observed['logits'],changed['logits'])
        assert set(model.state_dict())==original_keys


@pytest.mark.parametrize('task', ['major15','fine39'])
def test_complete_val_export_masks_missing_modalities_and_never_fits(tmp_path,task):
    torch.set_num_threads(1)
    config = fixture_config(tmp_path,task)
    config['model']['shared_private'] = True
    source = MultimodalData(config['data']['root'],task)
    dataset = MultimodalPeptides(source,config['data']['features'],'val')
    args = {'input_dim':dataset.meta['input_dim'],'num_labels':len(source.data.labels),
            'vocab_size':len(source.statistics['chemical_vocab'])+2,'statistics':source.statistics['labels'], **config['model']}
    model = research.make_model('trirelay',args).eval()
    loader = DataLoader(dataset,batch_size=2,collate_fn=collate_multimodal)
    y, full = research.predict_loader(model,loader,'cpu')
    collected = collect_readouts(model,loader,'cpu')
    np.testing.assert_array_equal(collected['probabilities'][:,-1],full)
    np.testing.assert_array_equal(collected['row_id'],source.data.rows[dataset.indices])
    threshold = .61
    heads, _, gates = summarize_heads(collected,source.statistics,threshold)
    assert heads['shared_geometry']['rows']==1
    assert heads['shared_sequence']['rows']==2
    assert heads['final']['common_available_rows']==1
    assert gates['geometry']['rows']==1
    # All geometry absent: metrics are unavailable, never a bias-only score.
    absent = dict(collected)
    absent['head_available'] = collected['head_available'].copy()
    absent['head_available'][:,[1,4,7]]=False
    unavailable, _, _ = summarize_heads(absent,source.statistics,threshold)
    assert unavailable['shared_geometry']['metrics'] is None
    assert unavailable['final']['common_available_metrics'] is None
    run = tmp_path/task
    run.mkdir()
    checkpoint = {'model_name':'trirelay','model_args':args,'model_state':model.state_dict(),'config':config,
                  'track':task,'statistics':source.statistics,'encoder':dataset.meta,'threshold':threshold,
                  'best_epoch':0,'weight_source':'online'}
    torch.save(checkpoint,run/'best.pt')
    (run/'results.json').write_text(json.dumps({'task':task,'threshold':threshold,
        'validation':research.metrics(y,full,source.statistics,threshold)}),encoding='utf-8')
    (run/'metadata.json').write_text(json.dumps({'git_commit':'synthetic_untrained_fixture','config':config}),encoding='utf-8')
    (run/'history.jsonl').write_text('',encoding='utf-8')
    result = export([run],tmp_path/'observed',device='cpu',batch_size=2,data_root=config['data']['root'])
    report = result['comparisons'][0]
    assert report['split']=='val' and report['rows']==2
    assert report['weight_updates'] is report['threshold_fit'] is report['inference_intervention'] is False
    assert report['saved_threshold_matches_results']
    assert all(v==0 for v in report['final_reproduction_deltas'].values() if v is not None)
    assert list(report['heads'])==list(HEADS)
    with np.load(tmp_path/'observed'/task/'readouts.npz',allow_pickle=False) as z:
        assert z['probabilities'].shape==(2,11,len(source.data.labels))
        assert z['head_available'].shape==(2,11)
        assert z['head_available'][:,1].tolist()==[True,False]
        assert z['threshold']==threshold
        assert z['head_names'].tolist()==list(HEADS)
        np.testing.assert_array_equal(z['targets'],y)
    with zipfile.ZipFile(tmp_path/'observed.zip') as z:
        assert len(z.namelist())==6
        assert f'{task}/metadata.json' in z.namelist()
        assert not any(n.endswith('.pt') for n in z.namelist())
    after = torch.load(run/'best.pt',weights_only=True)
    for key,value in checkpoint['model_state'].items():
        torch.testing.assert_close(value,after['model_state'][key],rtol=0,atol=0)


def test_first_read_export_rejects_a_non_partitioned_model(tmp_path):
    run=tmp_path/'unsupported';run.mkdir()
    (run/'results.json').write_text('{}')
    args={'input_dim':12,'num_labels':15,'vocab_size':10,'statistics':sample_statistics(15),'dim':32,'heads':4}
    model=research.make_model('trirelay',args)
    torch.save({'model_name':'trirelay','model_args':args,'model_state':model.state_dict()},run/'best.pt')
    with pytest.raises(ValueError,match='shared-private TriRelay'):
        export([run],tmp_path/'invalid')
