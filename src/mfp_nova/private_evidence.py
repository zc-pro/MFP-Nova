"""Export paired full-Val private-evidence interventions from existing checkpoints."""
import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from .research import evaluate


METRICS = ('macro_ap', 'micro_ap', 'macro_f1', 'micro_f1', 'accuracy', 'precision', 'coverage',
           'absolute_true', 'absolute_false', 'tail_macro_ap', 'mfp_detection.f1', 'mfp_function_prediction.f1')


def metric(mapping, key):
    for part in key.split('.'):
        if mapping is None:
            return None
        mapping = mapping[part]
    return mapping


def difference(left, right):
    return None if left is None or right is None else left-right


def export(runs, output, device='cpu', batch_size=64, data_root='data/therapeutic_2025', feature_root=None):
    """Two deterministic Val passes per run; no fitting, training or Test access."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    comparisons = []
    for directory in runs:
        directory = Path(directory)
        stored = json.loads((directory/'results.json').read_text(encoding='utf-8'))
        config = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))['config']
        destination = output/directory.name
        destination.mkdir()
        results = {}
        for variant, disabled in (('full', False), ('without_private', True)):
            print(f"{stored['task']} | {variant} | {directory/'best.pt'}", flush=True)
            results[variant] = evaluate(directory/'best.pt', data_root, feature_root or config['data']['features'],
                                       'val', device, destination/f'{variant}.json',
                                       predictions=destination/f'{variant}.npz', batch_size=batch_size,
                                       without_private_evidence=disabled)
        full, ablated = results['full'], results['without_private']
        # Report reproduction differences without refitting thresholds or silently
        # replacing the recorded result. Small batch-dependent rounding is visible.
        reproduction = {key:difference(metric(full['metrics'], key), metric(stored['validation'], key)) for key in METRICS}
        deltas = {key:difference(metric(ablated['metrics'], key), metric(full['metrics'], key)) for key in METRICS}
        with np.load(destination/'full.npz', allow_pickle=False) as left, np.load(destination/'without_private.npz', allow_pickle=False) as right:
            change = np.abs(left['probabilities']-right['probabilities'])
            threshold = left['threshold']
            changed_cells = int(((left['probabilities'] >= threshold)!=(right['probabilities'] >= threshold)).sum())
            per_label = [{'label':label,
                          'ap_delta':None if a is None else a-b,
                          'f1_delta':c-d}
                         for label,a,b,c,d in zip(full['labels'], ablated['metrics']['label_ap'],
                             full['metrics']['label_ap'], ablated['metrics']['label_f1'], full['metrics']['label_f1'])]
            prediction_change = {'mean_absolute_probability_change':float(change.mean()),
                                 'max_absolute_probability_change':float(change.max()),
                                 'changed_label_decisions':changed_cells,
                                 'changed_sample_decisions':int(((left['probabilities'] >= threshold)!=(right['probabilities'] >= threshold)).any(1).sum())}
        comparisons.append({'run':str(directory), 'task':full['task'], 'rows':full['rows'], 'threshold':full['threshold'],
                            'branch_supervision':config['train'].get('branch_supervision','mixed_private'),
                            'saved_threshold_matches_results':np.array_equal(full['threshold'], stored['threshold']),
                            'full_reproduction_deltas':reproduction, 'without_minus_full':deltas,
                            'prediction_change':prediction_change, 'per_label_without_minus_full':per_label})
        maximum = max(abs(value) for value in reproduction.values() if value is not None)
        print(f"{full['task']} | full reproduction max metric difference={maximum:.3g} | threshold={full['threshold']}", flush=True)
        print(f"{full['task']} | without-private minus full: "
              f"MacroAP={deltas['macro_ap']:+.6f} Jaccard={deltas['accuracy']:+.6f} "
              f"MFP-F1={deltas['mfp_detection.f1']:+.6f}", flush=True)
    summary = {'split':'val', 'threshold_fit':False, 'weight_updates':False,
               'scope':'Inference intervention on the same trained weights; not a retrained no-private baseline',
               'comparisons':comparisons}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    archive = output.with_name(output.name+'.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as zipped:
        for file in sorted(output.rglob('*')):
            if file.is_file():
                zipped.write(file,file.relative_to(output))
        for directory in runs:
            directory=Path(directory)
            for name in ('metadata.json','results.json','history.jsonl'):
                zipped.write(directory/name,Path(directory.name)/name)
    print(f'Completed Val only. Transfer: {archive}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True, help='Completed shared-private TriRelay run directories')
    parser.add_argument('--output', required=True, help='New directory; ZIP is written beside it')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data-root', default='data/therapeutic_2025')
    parser.add_argument('--feature-root')
    args = parser.parse_args()
    export(args.runs, args.output, args.device, args.batch_size, args.data_root, args.feature_root)


if __name__ == '__main__':
    main()
