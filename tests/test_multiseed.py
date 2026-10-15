import copy
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mfp_nova import experiments, research
from mfp_nova.metrics import evaluate_metrics
from mfp_nova.multimodal import function_projection
from test_research import multimodal_fixture


def test_traditional_five_and_undefined_statistics():
    y = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    p = np.array([[.9, .1, .8], [.1, .9, .1], [.1, .1, .1]])
    projection = function_projection(['antibacterial', 'antifungal', 'antiviral'], 'major15')
    score = evaluate_metrics(y, p, train_support=[1200, 400, 10], function_projection=projection)
    assert score['accuracy'] == pytest.approx(4/9)
    assert score['precision'] == score['coverage'] == score['sample_f1'] == .5
    assert score['absolute_true'] == pytest.approx(2/3)
    assert score['absolute_false'] == pytest.approx(2/9)
    assert score['label_ap'][2] is None and score['label_auroc'][2] is None
    assert score['undefined_metrics']['label_ap']['2'] == 'no_positive_targets'
    assert score['tail_macro_ap'] is None
    assert score['mfp_detection']['true_positive'] == 1
    assert score['mfp_detection']['support'] == score['mfp_detection']['predicted'] == 1
    json.dumps(score, allow_nan=False)


def train_configuration(tmp_path, name='trirelay', epochs=1, patience=0):
    root, features = multimodal_fixture(tmp_path)
    return {'model_name': name, 'model': {'dim': 32, 'heads': 4, 'dropout': 0},
            'data': {'root': str(root), 'track': 'major15', 'features': str(features)},
            'encoder': {'name': 'fixture', 'revision': 'main', 'max_length': 80},
            'loss': {'mask_ratio': .5, 'modality_dropout': .2},
            'train': {'device': 'cpu', 'seed': 42, 'epochs': epochs, 'batch_size': 2,
                      'learning_rate': .001, 'weight_decay': 0, 'patience': patience,
                      'warmup_epochs': 0, 'num_workers': 0, 'output': str(tmp_path)}}


@pytest.mark.parametrize('name', experiments.MODELS)
def test_epoch_metrics_reuse_forward_and_save_groups(tmp_path, name, monkeypatch, capsys):
    torch.set_num_threads(1)
    config = train_configuration(tmp_path, name)
    original = research.make_model
    calls, observations = [], []

    def model_factory(model_name, arguments):
        model = original(model_name, arguments)
        def observe(module, inputs, output):
            calls.append(module.training)
            if module.training:
                observations.append((inputs[0]['y'].detach().numpy(), output['logits'].detach().sigmoid().numpy()))
        model.register_forward_hook(observe)
        return model

    monkeypatch.setattr(research, 'make_model', model_factory)
    result = research.train(config, 'complete')
    assert calls == [True, True, False, False]  # train batches, one Val, one final-best Val
    history = json.loads((tmp_path/'complete/history.jsonl').read_text().splitlines()[0])
    assert history['train_rows'] == 4 and history['validation_rows'] == 2
    assert history['monitor_threshold'] == .5
    assert history['train_metrics_scope'] == 'online_augmented_changing_weights'
    assert history['validation_loss'] == pytest.approx(result['validation_loss'])
    for split in ('train', 'validation'):
        report = history[split]
        for field in ('accuracy', 'precision', 'coverage', 'absolute_true', 'absolute_false',
                      'macro_ap', 'micro_ap', 'macro_f1', 'micro_f1', 'sample_f1', 'macro_mcc',
                      'label_ap', 'label_auroc', 'label_mcc', 'mfp_detection'):
            assert field in report
        assert len(report['label_f1']) == 15
    y = np.concatenate([r[0] for r in observations])
    p = np.concatenate([r[1] for r in observations])
    assert history['train']['accuracy'] == evaluate_metrics(y, p)['accuracy']
    assert history['validation_groups']['structure_present']['rows'] == 1
    assert history['validation_groups']['D_annotation']['rows'] == 1
    assert history['validation_groups']['length_33_plus']['metrics'] is None
    assert result['validation_groups']['structure_missing']['rows'] == 1
    text = capsys.readouterr().out
    assert all(key in text for key in ('Train', 'Val', 'Acc=', 'Prec=', 'Cov=', 'AT=', 'AF=',
                                       'MacroAP=', 'MicroAP=', 'MacroF1=', 'MicroF1=', 'HeadAP=',
                                       'MediumAP=', 'TailAP=', 'MFP-F1=', 'LR=', 'Time=', 'Best=', 'Patience='))


def test_checkpoint_ties_and_terminal_resume_do_not_add_an_epoch(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = train_configuration(tmp_path, epochs=8, patience=2)
    original = research.metrics
    count = 0
    def tied_metrics(y, p, stats, threshold=.5):
        nonlocal count
        count += 1
        score = original(y, p, stats, threshold)
        epoch = (count+1)//2
        score.update(macro_ap=.4, micro_ap=.5 if epoch==1 else .6)
        return score
    monkeypatch.setattr(research, 'metrics', tied_metrics)
    result = research.train(config, 'first')
    assert result['best_epoch'] == 2 and result['epochs_completed'] == 4
    checkpoint = tmp_path/'first/last.pt'
    previous = torch.load(checkpoint, weights_only=True)
    history_before = (tmp_path/'first/history.jsonl').read_bytes()
    resumed = research.train(copy.deepcopy(config), 'finalized', resume=checkpoint)
    assert resumed['epochs_completed'] == 4 and resumed['global_step'] == previous['global_step']
    assert not (tmp_path/'finalized/history.jsonl').exists()
    assert (tmp_path/'finalized/last.pt').exists()
    assert (tmp_path/'first/history.jsonl').read_bytes() == history_before


def finished_run(root, model='trirelay', task='major15', seed=42, macro_ap=.4, metadata=False):
    name = f'{model}_{task}_esm650m_seed{seed}_rtx5070_fp32_b64'
    path = root/name
    path.mkdir(parents=True)
    score = evaluate_metrics(np.eye(2), np.array([[.8, .1], [.2, .7]]))
    score['macro_ap'] = macro_ap
    # Exercise aliases from the real old seed=42 schema.
    for field in ('accuracy', 'precision', 'coverage', 'absolute_true', 'absolute_false'):
        score.pop(field)
    result = {'model': model, 'task': task, 'seed': seed, 'precision': 'fp32', 'accumulation_steps': 1,
              'diagnostic_subset': False, 'threshold_strategy': 'global', 'threshold_fit_split': 'val',
              'threshold': .65, 'best_epoch': 12, 'epochs_completed': 27, 'global_step': 16659,
              'validation': score, 'labels': ['alpha', 'beta'], 'trainable_parameters': 100,
              'elapsed_seconds': 10, 'peak_cuda_memory_bytes': None}
    (path/'results.json').write_text(json.dumps(result))
    if metadata:
        config = yaml.safe_load((experiments.PROJECT/f'configs/{model}.yaml').read_text(encoding='utf-8'))
        config['train'].update(batch_size=64, precision='fp32', seed=seed)
        meta = {'model': model, 'task': task, 'seed': seed, 'config': config, 'labels': ['alpha', 'beta'],
                'encoder': {'name': 'facebook/esm2_t33_650M_UR50D'}, 'cohort': 'original_resolved_backbone'}
        (path/'metadata.json').write_text(json.dumps(meta))
    return path


def test_fixed_plan_reuses_six_legacy_results_and_identifies_live_resume(tmp_path):
    for model in experiments.MODELS:
        for task in experiments.TASKS:
            finished_run(tmp_path, model, task)
    plan = experiments.make_plan(tmp_path, running={})
    assert plan['total'] == 30 and plan['counts']['completed'] == 6 and plan['counts']['not_started'] == 24
    job = next(j for j in plan['jobs'] if j['seed']==101)
    interrupted = tmp_path/job['run_name']; interrupted.mkdir()
    (interrupted/'last.pt').write_bytes(b'not loaded by planner')
    resumed = experiments.make_plan(tmp_path, running={})['jobs'][1]
    assert resumed['status'] == 'unfinished_checkpoint'
    assert resumed['resume'] == str(interrupted/'last.pt')
    assert resumed['run_name'] != interrupted.name
    live = experiments.make_plan(tmp_path, running={interrupted.name: 1234})['jobs'][1]
    assert live['status'] == 'running' and live['running'][interrupted.name] == 1234
    assert not (tmp_path/resumed['run_name']).exists()


def test_no_checkpoint_restart_and_incompatibility_are_explicit(tmp_path):
    path = finished_run(tmp_path, seed=101, metadata=True)
    (path/'results.json').unlink()
    plan = experiments.make_plan(tmp_path, models=('trirelay',), tasks=('major15',), seeds=(101,), running={})
    job = plan['jobs'][0]
    assert job['status'] == 'unfinished_no_checkpoint' and job['run_name'] != path.name
    meta = json.loads((path/'metadata.json').read_text()); meta['config']['train']['precision'] = 'fp16'
    (path/'metadata.json').write_text(json.dumps(meta))
    (path/'last.pt').write_bytes(b'incompatible')
    job = experiments.make_plan(tmp_path, models=('trirelay',), tasks=('major15',), seeds=(101,), running={})['jobs'][0]
    assert job['status'] == 'incompatible' and 'precision' in ' '.join(job['issues'])


def test_stage_summary_std_legacy_aliases_and_missing_group(tmp_path):
    finished_run(tmp_path, seed=42, macro_ap=.4)
    finished_run(tmp_path, seed=101, macro_ap=.6, metadata=True)
    plan = experiments.make_plan(tmp_path, models=('trirelay',), tasks=('major15',), running={})
    summary = experiments.summarize(plan)['comparisons'][0]
    assert summary['seed_count'] == 2 and not summary['complete_five_seed']
    assert summary['metrics']['macro_ap']['mean'] == .5
    assert summary['metrics']['macro_ap']['std'] == pytest.approx(np.std([.4, .6], ddof=1))
    assert summary['metrics']['accuracy']['available_seeds'] == 2
    single = experiments.make_plan(tmp_path, models=('trirelay',), tasks=('major15',), seeds=(42,), running={})
    assert experiments.summarize(single)['comparisons'][0]['metrics']['macro_ap']['std'] is None
    label = experiments.summarize(plan, label='alpha')['comparisons'][0]
    assert label['metrics']['ap']['available_seeds'] == 2
    groups = experiments.summarize(plan, group='structure_missing')['comparisons'][0]
    assert groups['seed_count'] == 0 and len(groups['missing']) == 2
    for seed in (202, 303, 404):
        finished_run(tmp_path, seed=seed)
    all_seeds = experiments.make_plan(tmp_path, models=('trirelay',), tasks=('major15',), running={})
    assert experiments.summarize(all_seeds)['comparisons'][0]['complete_five_seed']
    with pytest.raises(ValueError, match='fixed protocol'):
        experiments.seed_list('42,999')


def test_completed_repeats_count_one_seed_and_diagnostics_are_excluded(tmp_path):
    first = finished_run(tmp_path, macro_ap=.9)
    second = tmp_path/(first.name+'_resume_later')
    shutil.copytree(first, second)
    result = json.loads((second/'results.json').read_text())
    result.update(global_step=result['global_step']+1)
    result['validation']['macro_ap'] = .3
    (second/'results.json').write_text(json.dumps(result))
    diagnostic = finished_run(tmp_path, seed=101)
    result['seed'], result['diagnostic_subset'] = 101, True
    (diagnostic/'results.json').write_text(json.dumps(result))
    plan = experiments.make_plan(tmp_path, models=('trirelay',), tasks=('major15',), running={})
    assert plan['counts']['completed'] == 1
    assert plan['jobs'][0]['completed_attempts'] == 2
    assert plan['jobs'][0]['macro_ap'] == .3  # furthest continuation, not largest AP
    assert experiments.summarize(plan)['comparisons'][0]['seeds'] == [42]


@pytest.mark.parametrize('shell', ['powershell', 'pwsh'])
@pytest.mark.parametrize('scenario', ['resume', 'failure', 'live'])
def test_powershell_resume_failure_and_live_process(tmp_path, shell, scenario):
    executable = shutil.which(shell)
    if executable is None:
        pytest.skip(f'{shell} unavailable')
    log = tmp_path/'calls.jsonl'
    checkpoint = str(tmp_path/'original/last.pt')
    jobs = [{'model': 'trirelay', 'task': 'major15', 'seed': seed,
             'status': 'unfinished_checkpoint' if seed==101 else 'not_started',
             'run_name': f'trirelay_major15_seed{seed}_new_attempt',
             'resume': checkpoint if seed==101 else None, 'issues': [], 'running': {}}
            for seed in (101, 202)]
    if scenario=='live':
        # Even a completed seed being rerun must prevent a second GPU job.
        jobs.append({'model': 'geofactor', 'task': 'fine39', 'seed': 42,
                     'status': 'completed', 'run_name': 'live_completed_repeat',
                     'resume': None, 'issues': [], 'running': {'live_completed_repeat': 1234}})
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'total': len(jobs), 'counts': {'completed': 0}, 'jobs': jobs}))
    fake = tmp_path/'fake_python.ps1'
    fake.write_text("""param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CallArgs)
if ($CallArgs -contains 'mfp_nova.experiments') {
    if ($CallArgs -contains '--json') { Get-Content -LiteralPath 'PLAN' -Raw }
    else { Write-Output 'Status displayed.' }
    exit 0
}
ConvertTo-Json -InputObject $CallArgs -Compress | Add-Content -LiteralPath 'LOG'
exit CODE
""".replace('PLAN', str(plan)).replace('LOG', str(log)).replace('CODE', '1' if scenario=='failure' else '0'),
                    encoding='utf-8')
    completed = subprocess.run([executable, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                                str(experiments.PROJECT/'scripts/run_multiseed.ps1'),
                                '-Python', str(fake), '-MaxRuns', '3'], capture_output=True, text=True,
                                encoding='utf-8' if shell=='pwsh' else None)
    calls = [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
    assert completed.returncode == 0 if scenario=='resume' else completed.returncode != 0
    if scenario=='live':
        assert calls == []
    else:
        assert len(calls) == (2 if scenario=='resume' else 1)
        assert calls[0][calls[0].index('--resume')+1] == checkpoint
        assert calls[0][calls[0].index('--name')+1].endswith('_new_attempt')


@pytest.mark.parametrize('shell', ['powershell', 'pwsh'])
@pytest.mark.parametrize('arguments,expected', [(['-DryRun'], 0), ([], 1), (['-MaxRuns', '3'], 3),
                                               (['-All'], 24), (['-All', '-MaxRuns', '1'], -1)])
def test_powershell_limits_with_fake_python(tmp_path, shell, arguments, expected):
    executable = shutil.which(shell)
    if executable is None:
        pytest.skip(f'{shell} unavailable')
    log = tmp_path/'calls.jsonl'
    jobs = []
    for model in experiments.MODELS:
        for task in experiments.TASKS:
            for seed in experiments.SEEDS:
                jobs.append({'model': model, 'task': task, 'seed': seed, 'status': 'completed' if seed==42 else 'not_started',
                             'run_name': f'{model}_{task}_seed{seed}', 'resume': None, 'issues': []})
    plan = tmp_path/'plan.json'
    plan.write_text(json.dumps({'total': 30, 'counts': {'completed': 6}, 'jobs': jobs}))
    fake = tmp_path/'fake_python.ps1'
    fake.write_text("""param([Parameter(ValueFromRemainingArguments=$true)][string[]]$CallArgs)
if ($CallArgs -contains 'mfp_nova.experiments') {
    if ($CallArgs -contains '--json') { Get-Content -LiteralPath 'PLAN' -Raw }
    else { Write-Output 'Status displayed.' }
} else { ConvertTo-Json -InputObject $CallArgs -Compress | Add-Content -LiteralPath 'LOG' }
exit 0
""".replace('PLAN', str(plan)).replace('LOG', str(log)), encoding='utf-8')
    script = experiments.PROJECT/'scripts/run_multiseed.ps1'
    completed = subprocess.run([executable, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script),
                                '-Python', str(fake), *arguments], capture_output=True, text=True,
                                encoding='utf-8' if shell=='pwsh' else None)
    if expected == -1:
        assert completed.returncode != 0 and not log.exists()
    else:
        assert completed.returncode == 0, completed.stdout+completed.stderr
        calls = [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
        assert len(calls) == expected
        assert all(call[call.index('--seed')+1] != '42' for call in calls)
        assert all('--resume' not in call for call in calls)
