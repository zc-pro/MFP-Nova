import copy
import json
from argparse import Namespace
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

from mfp_nova import research
from mfp_nova.theraset_ablation import (BATCHES, VARIANTS, command, find_run,
                                       job_config, jobs, pack, run)
from test_research import multimodal_fixture


def test_matrix_and_task_protocols():
    matrix = list(jobs())
    assert len(matrix) == 54
    assert len([j for j in matrix if j[0] == 'therapep49']) == 14
    for task, variant, suite in matrix:
        config = job_config(task, suite, variant)
        assert config['train']['seed'] == 42
        assert config['train']['batch_size'] == BATCHES[task]
        assert config['train']['accumulation_steps'] == 1
        assert config['train']['metric_suite'] == suite
        for switch in VARIANTS[variant]:
            assert config['model'][switch] is False
        if task == 'therapep49':
            assert config['data']['root'] == 'data/therapep_ai'
            assert config['data']['features'] == 'features/therapep_ai/esm2_650m'
            assert config['loss']['modality_dropout'] == 0
            assert not any(config['model'][k] for k in ('chemical_evidence', 'property_evidence', 'structure_evidence'))


def test_cli_passes_pause_controls_and_keeps_explicit_therapep_suite(monkeypatch):
    from mfp_nova.cli import main
    captured = []
    monkeypatch.setattr(research, 'train', lambda config, name, **kwargs: captured.append((config, kwargs)))
    main(['train', '--model', 'theraset', '--task', 'therapep49', '--metric-suite', 'traditional5',
          '--batch-size', '64', '--accumulation-steps', '1', '--name', 'job',
          '--resume', 'runs/job/last.pt', '--resume-in-place', '--stop-file', 'runs/PAUSE'])
    config, options = captured[0]
    assert config['train']['metric_suite'] == 'traditional5'
    assert options == {'resume': 'runs/job/last.pt', 'resume_in_place': True, 'stop_file': 'runs/PAUSE'}


def completed_run(directory, config):
    directory.mkdir(parents=True)
    (directory/'metadata.json').write_text(json.dumps({'model': 'theraset', 'config': config}))
    (directory/'results.json').write_text(json.dumps({'best_epoch': 1, 'validation': {'micro_f1': .6}}))


def test_reuse_only_matching_controls_and_resume_command(tmp_path):
    old = tmp_path/'old'
    control = old/'theraset_fine39_finished'
    completed_run(control, job_config('fine39', 'micro_f1', 'full'))
    assert find_run(tmp_path/'queue', 'fine39', 'full', 'micro_f1', old) == control
    assert find_run(tmp_path/'queue', 'fine39', 'full', 'traditional5', old) != control
    major = old/'theraset_major15_b64'
    config = job_config('major15', 'micro_f1', 'full')
    config['train']['batch_size'] = 64
    completed_run(major, config)
    assert find_run(tmp_path/'queue', 'major15', 'full', 'micro_f1', old) != major
    paused = tmp_path/'queue/major15/no_pair/micro_f1'
    paused.mkdir(parents=True)
    (paused/'last.pt').touch()
    argv = command('major15', 'no_pair', 'micro_f1', paused, 'cuda:0', tmp_path/'PAUSE')
    assert '--resume-in-place' in argv
    assert argv[argv.index('--batch-size')+1] == '128'
    assert argv[argv.index('--ablate')+1] == 'pair_readout'


@pytest.mark.parametrize('suite', ['traditional5', 'micro_f1'])
def test_paused_theraset_resumes_same_epoch_history_and_weights(tmp_path, monkeypatch, suite):
    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    config = {'model_name': 'theraset',
              'model': {'dim': 32, 'heads': 4, 'cnn_channels': 8, 'pair_dim': 8, 'decoder_depth': 1, 'dropout': .1},
              'data': {'root': str(root), 'track': 'major15', 'features': str(features)},
              'encoder': {'name': 'fixture', 'revision': 'main', 'max_length': 80},
              'loss': {'modality_dropout': .1, 'rdrop': .1},
              'train': {'device': 'cpu', 'seed': 42, 'epochs': 3, 'batch_size': 2,
                        'learning_rate': .001, 'weight_decay': 0, 'patience': 0, 'num_workers': 0,
                        'warmup_epochs': 0, 'aux_warmup_epochs': 0, 'accumulation_steps': 1,
                        'ema_decay': .9, 'ema_start_epoch': 1, 'rdrop_start_epoch': 1, 'rdrop_ramp_epochs': 1,
                        'metric_suite': suite, 'save_selection_candidates': True,
                        'export_validation_predictions': True, 'output': str(tmp_path)}}
    full_result = research.train(copy.deepcopy(config), 'full')
    stop = tmp_path/'PAUSE'
    real_format = research.format_epoch
    def pause_after_epoch(*args):
        if args[4]['epoch'] == 2:
            stop.touch()
        return real_format(*args)
    with monkeypatch.context() as patch:
        patch.setattr(research, 'format_epoch', pause_after_epoch)
        paused = research.train(copy.deepcopy(config), 'partial', stop_file=stop)
    assert paused['paused'] and paused['epoch'] == 2
    assert not (tmp_path/'partial/results.json').exists()
    # Simulate interruption between checkpoint replacement and history append.
    history_path = tmp_path/'partial/history.jsonl'
    lines = history_path.read_text().splitlines()
    history_path.write_text(lines[0]+'\n{"epoch":', encoding='utf-8')
    stop.unlink()
    resumed = research.train(copy.deepcopy(config), 'partial', resume=tmp_path/'partial/last.pt',
                             resume_in_place=True, stop_file=stop)
    full = torch.load(tmp_path/'full/last.pt', weights_only=True)
    saved = torch.load(tmp_path/'partial/last.pt', weights_only=True)
    for key in ('model_state', 'ema_state', 'best_model_state', 'online_best_state'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, saved[key][name], rtol=0, atol=0)
    assert full['global_step'] == saved['global_step']
    assert full['scheduler_state'] == saved['scheduler_state']
    assert full_result['validation'] == resumed['validation']
    assert [json.loads(l)['epoch'] for l in history_path.read_text().splitlines()] == [1, 2, 3]
    assert not (tmp_path/'partial/last.pt.tmp').exists()


def test_interrupted_save_preserves_last_complete_checkpoint(tmp_path, monkeypatch):
    target = tmp_path/'last.pt'
    torch.save({'epoch': 1}, target)
    def failed_save(value, path):
        Path(path).write_bytes(b'incomplete')
        raise OSError('simulated disk interruption')
    monkeypatch.setattr(research.torch, 'save', failed_save)
    with pytest.raises(OSError, match='simulated'):
        research.save_last_checkpoint({'epoch': 2}, target)
    assert torch.load(target, weights_only=True)['epoch'] == 1


@pytest.mark.parametrize('suite', ['traditional5', 'micro_f1'])
@pytest.mark.parametrize('completed_epochs', [0, 1])
def test_helixset_ctrl_c_replays_uncommitted_epoch_in_place(tmp_path, monkeypatch, suite, completed_epochs):
    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    config = {'model_name': 'helixset',
        'model': {'dim': 32, 'heads': 4, 'dropout': .1, 'sequence_depth': 1, 'property_depth': 1,
                  'geometry_depth': 1, 'pair_depth': 1, 'pair_dim': 8, 'latent_depth': 1, 'decoder_depth': 1},
        'data': {'root': str(root), 'track': 'major15', 'features': str(features)},
        'encoder': {'name': 'fixture', 'revision': 'main', 'max_length': 80},
        'loss': {'modality_dropout': .1, 'rdrop': .1},
        'train': {'device': 'cpu', 'seed': 42, 'epochs': 3, 'batch_size': 2, 'accumulation_steps': 2,
                  'learning_rate': .001, 'weight_decay': 0, 'patience': 0, 'num_workers': 0,
                  'warmup_epochs': 0, 'aux_warmup_epochs': 0, 'ema_decay': .9, 'ema_start_epoch': 1,
                  'rdrop_start_epoch': 1, 'rdrop_ramp_epochs': 1, 'metric_suite': suite,
                  'save_selection_candidates': True, 'output': str(tmp_path)}}
    expected = research.train(copy.deepcopy(config), 'full')
    make_model = research.make_model
    def interrupted_model(*args):
        model = make_model(*args)
        compute_loss = model.compute_loss
        calls = 0
        def interrupt(*values, **options):
            nonlocal calls
            calls += 1
            # Two microbatches per epoch; interrupt the second microbatch before an update.
            if calls == completed_epochs*2 + 2:
                raise KeyboardInterrupt
            return compute_loss(*values, **options)
        model.compute_loss = interrupt
        return model
    with monkeypatch.context() as patch:
        patch.setattr(research, 'make_model', interrupted_model)
        with pytest.raises(KeyboardInterrupt):
            research.train(copy.deepcopy(config), 'partial')
    last = tmp_path/'partial/last.pt'
    assert last.exists() == bool(completed_epochs)
    resumed = research.train(copy.deepcopy(config), 'partial',
                             resume=last if last.exists() else None, resume_in_place=True)
    full = torch.load(tmp_path/'full/last.pt', weights_only=True)
    saved = torch.load(last, weights_only=True)
    for key in ('model_state', 'ema_state', 'best_model_state', 'online_best_state'):
        for name, value in full[key].items():
            torch.testing.assert_close(value, saved[key][name], rtol=0, atol=0)
    assert full['global_step'] == saved['global_step']
    assert full['scheduler_state'] == saved['scheduler_state']
    assert full['stale'] == saved['stale']
    assert expected['validation'] == resumed['validation']
    history = (tmp_path/'partial/history.jsonl').read_text().splitlines()
    assert [json.loads(line)['epoch'] for line in history] == [1, 2, 3]


@pytest.mark.parametrize('has_checkpoint', [False, True])
def test_cli_ctrl_c_reports_epoch_boundary_resume(tmp_path, monkeypatch, capsys, has_checkpoint):
    from mfp_nova.cli import main
    run = tmp_path/'existing_run'
    run.mkdir()
    if has_checkpoint:
        (run/'last.pt').touch()
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(research, 'train', interrupt)
    with pytest.raises(SystemExit) as error:
        main(['train', '--model', 'helixset', '--name', str(run)])
    assert error.value.code == 130
    message = capsys.readouterr().out
    assert str(run) in message
    assert ('last complete epoch' if has_checkpoint else 'Restart epoch 1') in message


def test_pack_named_results_without_weights_including_reused_baseline(tmp_path):
    output, reuse = tmp_path/'queue', tmp_path/'old'
    control = reuse/'theraset_fine39_completed'
    completed_run(control, job_config('fine39', 'micro_f1', 'full'))
    (control/'best.pt').write_bytes(b'large weight')
    (control/'best_validation.npz').write_bytes(b'predictions')
    args = Namespace(output=str(output), tasks=['fine39'], suites=['micro_f1'], variants=['full'],
                     reuse_root=str(reuse), zip=None, platform='v100')
    pack(args)
    package = next(tmp_path.glob('theraset_full_seed42_v100_fine39-b64_microf1_1of1_*.zip'))
    with ZipFile(package) as z:
        assert 'summary.csv' in z.namelist()
        assert any(n.endswith('/best_validation.npz') for n in z.namelist())
        assert not any(n.endswith('.pt') for n in z.namelist())


def test_queue_resumes_unfinished_jobs_and_skips_finished(tmp_path, monkeypatch):
    output = tmp_path/'queue'
    done = output/'fine39/full/micro_f1'
    completed_run(done, job_config('fine39', 'micro_f1', 'full'))
    paused = output/'fine39/no_pair/micro_f1'
    paused.mkdir(parents=True)
    (paused/'last.pt').touch()
    observed = []
    class Process:
        stdout = []
        def __init__(self, argv, **kwargs):
            observed.append(argv)
            (paused/'results.json').write_text('{}')
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait(self): return 0
    monkeypatch.setattr('mfp_nova.theraset_ablation.subprocess.Popen', Process)
    run(Namespace(output=str(output), tasks=['fine39'], suites=['micro_f1'],
                  variants=['full', 'no_pair'], reuse_root=None, device='cuda:0', cpu_threads=4))
    assert len(observed) == 1 and '--resume-in-place' in observed[0]
