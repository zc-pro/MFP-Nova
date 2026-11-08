"""Seed42 TheraSet ablations from section 8 of the architecture document."""
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

from .therapep import configure_task

PROJECT = Path(__file__).resolve().parents[2]
TASKS = ('major15', 'fine39', 'therapep49')
SUITES = ('traditional5', 'micro_f1')
BATCHES = {'major15': 128, 'fine39': 64, 'therapep49': 64}
VARIANTS = {
    'full': (),
    'cnn': ('evidence_decoder', 'chemical_evidence'),
    'cnn_chemical': ('evidence_decoder',),
    'no_plm': ('contextual_evidence',),
    'no_property': ('property_evidence',),
    'no_structure': ('structure_evidence',),
    'no_pair': ('pair_readout',),
    'single_prototype': ('prototype_readout',),
    'no_label_context': ('label_context',),
    'no_rdrop': (),
}
# Preserve the original 54-run experiment as the default matrix.
DEFAULT_VARIANTS = tuple(VARIANTS)
VARIANTS.update({
    'no_label_context_single_prototype': ('label_context', 'prototype_readout'),
    'no_label_context_no_pair': ('label_context', 'pair_readout'),
})
# These options are already disabled for every author-published row.
THERAPEP_REDUNDANT = {'cnn_chemical', 'no_property', 'no_structure'}


def job_config(task, suite, variant, device='cuda:0'):
    config = yaml.safe_load((PROJECT/'configs/theraset.yaml').read_text(encoding='utf-8'))
    configure_task(config, task)
    config['train'].update(seed=42, batch_size=BATCHES[task], accumulation_steps=1,
                           precision='fp32', metric_suite=suite, device=device)
    for switch in VARIANTS[variant]:
        config['model'][switch] = False
    if variant == 'no_rdrop':
        config['loss']['rdrop'] = 0.
    return config


def jobs(tasks=TASKS, suites=SUITES, variants=DEFAULT_VARIANTS):
    for task in tasks:
        for variant in variants:
            if task == 'therapep49' and variant in THERAPEP_REDUNDANT:
                continue
            for suite in suites:
                yield task, variant, suite


def same_experiment(saved, expected):
    """Compare scientific settings; GPU location and output paths can differ."""
    saved, expected = copy.deepcopy(saved), copy.deepcopy(expected)
    for config in (saved, expected):
        for key in ('device', 'output', 'num_workers'):
            config['train'].pop(key, None)
    return saved == expected


def find_run(output, task, variant, suite, reuse_root, device='cuda:0'):
    expected = job_config(task, suite, variant, device)
    directory = output/task/variant/suite
    metadata = directory/'metadata.json'
    if metadata.exists():
        saved = json.loads(metadata.read_text(encoding='utf-8'))
        if not same_experiment(saved['config'], expected):
            raise ValueError(f'Experiment settings differ in {directory}; choose a new --output')
    if (directory/'results.json').exists() or (directory/'last.pt').exists():
        return directory
    # Reuse an explicitly identified completed experiment, never choose by score.
    if reuse_root is not None:
        candidates = [Path(reuse_root)/task/variant/suite]
        if variant == 'full':
            candidates.extend(sorted(Path(reuse_root).glob('theraset_*')))
        for candidate in candidates:
            if not (candidate/'results.json').is_file() or not (candidate/'metadata.json').is_file():
                continue
            saved = json.loads((candidate/'metadata.json').read_text(encoding='utf-8'))
            if saved.get('model') == 'theraset' and not saved.get('diagnostic_subset') and same_experiment(saved['config'], expected):
                return candidate.resolve()
    return directory


def command(task, variant, suite, directory, device, stop_file):
    argv = [sys.executable, '-u', '-m', 'mfp_nova.cli', 'train', '--model', 'theraset',
            '--config', str(PROJECT/'configs/theraset.yaml'), '--task', task,
            '--seed', '42', '--metric-suite', suite, '--device', device, '--precision', 'fp32',
            '--batch-size', str(BATCHES[task]), '--accumulation-steps', '1',
            '--num-workers', '0', '--name', str(directory.resolve()), '--stop-file', str(stop_file)]
    if VARIANTS[variant]:
        argv += ['--ablate', *VARIANTS[variant]]
    if variant == 'no_rdrop':
        argv += ['--rdrop-weight', '0']
    if (directory/'last.pt').is_file():
        argv += ['--resume', str(directory/'last.pt'), '--resume-in-place']
    elif directory.exists():
        argv += ['--resume-in-place']
    return argv


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stop_file = output/'PAUSE'
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
               PYTHONPATH=str(PROJECT/'src') + os.pathsep + os.environ.get('PYTHONPATH', ''),
               OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
               OPENBLAS_NUM_THREADS=str(args.cpu_threads))
    for task, variant, suite in jobs(args.tasks, args.suites, args.variants):
        if stop_file.exists():
            print(f'Queue paused. Remove {stop_file} and repeat this run command.', flush=True)
            return
        directory = find_run(output, task, variant, suite, args.reuse_root, args.device)
        if (directory/'results.json').exists():
            print(f'SKIP completed: {task}/{variant}/{suite}: {directory}', flush=True)
            if getattr(args, 'diagnostics', False) and not (directory/'evidence_diagnostics.json').is_file():
                diagnose_run(directory, args)
            continue
        argv = command(task, variant, suite, directory, args.device, stop_file)
        print(f'{"RESUME" if (directory/"last.pt").exists() else "START"}: {task}/{variant}/{suite}', flush=True)
        print(subprocess.list2cmdline(argv), flush=True)
        directory.parent.mkdir(parents=True, exist_ok=True)
        # Every experiment has its own appended console log, including continuations.
        with directory.with_suffix('.log').open('a', encoding='utf-8') as log:
            with subprocess.Popen(argv, cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as process:
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except KeyboardInterrupt:
                    stop_file.touch()
                    print('Pause requested. Waiting for the training process to stop.', flush=True)
                    process.wait()
                    return
        if code:
            raise SystemExit(f'Training exited with code {code}: {directory}. Repeat this command to resume last.pt.')
        if not (directory/'results.json').exists():
            if stop_file.exists():
                return
            raise RuntimeError(f'Training ended without results.json: {directory}')
        if getattr(args, 'diagnostics', False):
            diagnose_run(directory, args)
    print('Selected seed42 ablations completed.', flush=True)


def diagnose_run(directory, args):
    # Separate process releases the training CUDA context before the Val pass.
    subprocess.run([sys.executable, '-u', '-m', 'mfp_nova.theraset_diagnostics',
                    '--run', str(directory), '--device', args.device,
                    '--cpu-threads', str(args.cpu_threads)], check=True, cwd=PROJECT)


def diagnose(args):
    for task, variant, suite in jobs(args.tasks, args.suites, args.variants):
        directory = find_run(Path(args.output).resolve(), task, variant, suite, args.reuse_root, args.device)
        diagnose_run(directory, args)


def prepare(args):
    """Reuse existing Train/Val caches; extraction batch is independent of training."""
    if 'therapep49' in args.tasks:
        subprocess.run([sys.executable, '-m', 'mfp_nova.cli', 'download-therapep'], check=True, cwd=PROJECT)
    for task in ('major15', 'therapep49'):
        if task == 'major15' and not set(args.tasks) & {'major15', 'fine39'}:
            continue
        if task == 'therapep49' and task not in args.tasks:
            continue
        config = job_config(task, 'micro_f1', 'full', args.device)
        for view, key in (('primary', 'features'), ('peptide', 'peptide_features')):
            for split in ('train', 'val'):
                if (PROJECT/config['data'][key]/split/'metadata.json').is_file():
                    print(f'SKIP cached: {task}/{view}/{split}', flush=True)
                    continue
                subprocess.run([sys.executable, '-u', '-m', 'mfp_nova.cli', 'features',
                                '--config', str(PROJECT/'configs/theraset.yaml'), '--track', task,
                                '--encoder-view', view, '--splits', split, '--device', args.device,
                                '--batch-size', '8'], check=True, cwd=PROJECT)


def pack(args):
    output = Path(args.output).resolve()
    rows, directories = [], {}
    fields = ('accuracy', 'precision', 'coverage', 'absolute_true', 'absolute_false',
              'micro_f1', 'macro_f1', 'micro_ap', 'macro_ap', 'head_macro_ap', 'medium_macro_ap', 'tail_macro_ap')
    for task, variant, suite in jobs(args.tasks, args.suites, args.variants):
        directory = find_run(output, task, variant, suite, args.reuse_root)
        result_path = directory/'results.json'
        record = {'task': task, 'variant': variant, 'suite': suite, 'seed': 42,
                  'batch_size': BATCHES[task], 'directory': str(directory), 'completed': result_path.exists(),
                  'diagnostics_completed': (directory/'evidence_diagnostics.json').is_file()}
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding='utf-8'))
            record.update({k: result.get(k) for k in ('best_epoch', 'epochs_completed', 'threshold', 'weight_source', 'elapsed_seconds', 'peak_cuda_memory_bytes')})
            record.update({k: result['validation'].get(k) for k in fields})
        rows.append(record)
        if directory.is_dir():
            directories[directory] = (task, variant, suite)
    completed = sum(r['completed'] for r in rows)
    tasks_tag = '_'.join(f'{task}-b{BATCHES[task]}' for task in args.tasks)
    suites_tag = '-'.join(s.replace('_', '') for s in args.suites)
    variants_tag = 'all-ablations' if set(args.variants) == set(DEFAULT_VARIANTS) else '-'.join(args.variants)
    filename = (f'theraset_{variants_tag}_seed42_{args.platform}_{tasks_tag}_{suites_tag}_'
                f'{completed}of{len(rows)}_{datetime.now():%Y%m%d_%H%M%S}.zip')
    destination = Path(args.zip).resolve() if args.zip else output.parent/filename
    columns = list(dict.fromkeys(k for row in rows for k in row))
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output/'summary.csv'
    with summary_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, 'x', ZIP_DEFLATED) as archive:
        archive.write(summary_path, 'summary.csv')
        for directory in sorted(directories):
            prefix = ('runs/'+directory.relative_to(output).as_posix() if directory.is_relative_to(output)
                      else 'reused/'+'/'.join(directories[directory]))
            for file in sorted(directory.rglob('*')):
                if file.is_file() and file.suffix.lower() in {'.json', '.jsonl', '.npz', '.csv', '.log'}:
                    archive.write(file, prefix+'/'+file.relative_to(directory).as_posix())
            log = directory.with_suffix('.log')
            if log.is_file():
                archive.write(log, prefix+'/console.log')
    print(f'{sum(r["completed"] for r in rows)}/{len(rows)} complete; package: {destination}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'prepare', 'run', 'pause', 'pack', 'diagnose'))
    parser.add_argument('--output', default='runs/theraset_ablation_seed42')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--suites', nargs='+', choices=SUITES, default=list(SUITES))
    parser.add_argument('--variants', nargs='+', choices=tuple(VARIANTS), default=list(DEFAULT_VARIANTS))
    parser.add_argument('--reuse-root', default='runs', help='Reuse matching completed full controls from this directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--diagnostics', action='store_true', help='run: export direct/final Val readouts after each completed job')
    parser.add_argument('--platform', choices=('rtx5070', 'v100'), default='rtx5070', help='Platform label in the result package filename')
    parser.add_argument('--zip', help='pack: destination ZIP, excluding weights/caches')
    args = parser.parse_args(argv)
    os.chdir(PROJECT)
    if args.action == 'plan':
        counts = {'completed': 0, 'resume': 0, 'new': 0}
        for task, variant, suite in jobs(args.tasks, args.suites, args.variants):
            directory = find_run(Path(args.output).resolve(), task, variant, suite, args.reuse_root, args.device)
            state = 'completed' if (directory/'results.json').exists() else 'resume' if (directory/'last.pt').exists() else 'new'
            counts[state] += 1
            print(f'{task:10} {variant:18} {suite:13} b{BATCHES[task]:3} {state:10} {directory}')
        print(counts)
    elif args.action == 'pause':
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output/'PAUSE').touch()
        print(f'Pause requested: {output / "PAUSE"}; active jobs stop after their current epoch.')
    else:
        {'prepare': prepare, 'run': run, 'pack': pack, 'diagnose': diagnose}[args.action](args)


if __name__ == '__main__':
    main()
