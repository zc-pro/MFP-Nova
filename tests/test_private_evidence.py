import copy
import json
import zipfile

import numpy as np
import pytest
import torch

from mfp_nova import research
from mfp_nova.cli import main
from mfp_nova.private_evidence import export
from test_consistency import fixture_config
from test_research import synthetic_batch, sample_statistics


@pytest.mark.parametrize('labels', [15,39])
def test_private_intervention_removes_only_residual_in_both_reads(labels):
    batch = synthetic_batch(labels)
    model = research.make_model('trirelay', {'input_dim':12,'num_labels':labels,'vocab_size':10,
                               'statistics':sample_statistics(labels),'dim':32,'heads':4,
                               'dropout':0,'shared_private':True}).eval()
    saved = {key:value.clone() for key,value in model.state_dict().items()}
    reads = []
    # Instrument the gate modules because read() is called directly.
    handles = [module.register_forward_hook(lambda module, inputs, value: reads.append(value.detach()))
               for module in model.evidence_partition.gates]
    with torch.inference_mode():
        full = model(batch)
        reads.clear()
        model.private_evidence_enabled = False
        ablated = model(batch)
        assert len(reads) == 6  # All three modalities in both reads remain computed.
        assert ablated['private_gate'].count_nonzero() == 0
        assert ablated['fusion'][1,:,1].count_nonzero() == 0
        assert not torch.allclose(full['logits'], ablated['logits'])
        assert torch.isfinite(ablated['logits']).all()
        model.private_evidence_enabled = True
        # An independent intervention on the reader outputs must give the same logits.
        zeros = [module.register_forward_hook(lambda module, inputs, value: value*0)
                 for module in model.evidence_partition.readers]
        independently_ablated = model(batch)
        for hook in zeros:
            hook.remove()
        torch.testing.assert_close(independently_ablated['logits'], ablated['logits'], rtol=0, atol=0)
        torch.testing.assert_close(model(batch)['logits'], full['logits'], rtol=0, atol=0)
    for hook in handles:
        hook.remove()
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,saved[key],rtol=0,atol=0)


def test_two_task_val_export_preserves_weights_thresholds_and_packages_predictions(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = fixture_config(tmp_path,'fine39')
    config['model']['shared_private'] = True
    config['train'].update(epochs=1,consistency_weight=0)
    config['train'].pop('consistency_ramp_epochs')
    runs = []
    states = []
    for task in ('major15','fine39'):
        current = copy.deepcopy(config)
        current['data']['track'] = task
        research.train(current,task)
        run = tmp_path/task
        runs.append(run)
        states.append(torch.load(run/'best.pt',weights_only=True))
    summary = export(runs,tmp_path/'paired',device='cpu',batch_size=2,data_root=config['data']['root'])
    assert summary['split']=='val' and summary['threshold_fit'] is False and summary['weight_updates'] is False
    assert len(summary['comparisons']) == 2
    for run,original,measured in zip(runs,states,summary['comparisons']):
        assert measured['saved_threshold_matches_results']
        assert all(value==0 for value in measured['full_reproduction_deltas'].values() if value is not None)
        full = json.loads((tmp_path/'paired'/run.name/'full.json').read_text())
        disabled = json.loads((tmp_path/'paired'/run.name/'without_private.json').read_text())
        assert full['threshold']==disabled['threshold']==original['threshold']
        assert full['inference_ablation']['private_evidence_enabled'] is True
        assert disabled['inference_ablation']['private_evidence_enabled'] is False
        assert disabled['inference_ablation']['threshold_refit'] is False
        with np.load(tmp_path/'paired'/run.name/'full.npz',allow_pickle=False) as a, np.load(tmp_path/'paired'/run.name/'without_private.npz',allow_pickle=False) as b:
            np.testing.assert_array_equal(a['row_id'],b['row_id'])
            np.testing.assert_array_equal(a['label_names'],b['label_names'])
            np.testing.assert_array_equal(a['targets'],b['targets'])
            assert not np.array_equal(a['probabilities'],b['probabilities'])
            assert json.loads(b['inference_ablation_json'].item())['private_evidence_enabled'] is False
            assert a['split'].item()==b['split'].item()=='val'
        after = torch.load(run/'best.pt',weights_only=True)
        for key,value in original['model_state'].items():
            torch.testing.assert_close(after['model_state'][key],value,rtol=0,atol=0)
    with zipfile.ZipFile(tmp_path/'paired.zip') as archive:
        assert len(archive.namelist())==15
        assert 'summary.json' in archive.namelist()
        assert all(name.endswith(('.json','.jsonl','.npz')) for name in archive.namelist())

    captured = {}
    monkeypatch.setattr(research,'evaluate',lambda *args,**kwargs: captured.update(kwargs))
    main(['evaluate','--checkpoint',str(runs[0]/'best.pt'),'--split','val',
          '--output',str(tmp_path/'cli.json'),'--without-private-evidence'])
    assert captured['without_private_evidence'] is True


def test_unsupported_checkpoint_cannot_be_mislabeled_as_private_ablation(tmp_path):
    model = research.make_model('trirelay', {'input_dim':12,'num_labels':15,'vocab_size':10,
                               'statistics':sample_statistics(15),'dim':32,'heads':4,'dropout':0})
    checkpoint = {'model_name':'trirelay', 'model_args':{'input_dim':12,'num_labels':15,'vocab_size':10,
                    'statistics':sample_statistics(15),'dim':32,'heads':4,'dropout':0},'model_state':model.state_dict()}
    torch.save(checkpoint,tmp_path/'old.pt')
    with pytest.raises(ValueError,match='shared-private TriRelay'):
        research.evaluate(tmp_path/'old.pt','unused','unused','val','cpu',tmp_path/'invalid.json',without_private_evidence=True)
