"""Observe TheraSet's trained direct head and evidence correction on Val."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from .research import load_model, metrics, to_device, validate_cache
from .trainer import write_json


def transitions(y, direct, final, threshold):
    """Decision changes against recorded labels; zeros are not verified negatives."""
    observed = y.astype(bool)
    before, after = direct >= threshold, final >= threshold
    repaired = (before != observed) & (after == observed)
    harmed = (before == observed) & (after != observed)
    return {
        'label_decisions': int(y.size),
        'changed': int((before != after).sum()),
        'repaired': int(repaired.sum()), 'harmed': int(harmed.sum()),
        'repaired_per_label': repaired.sum(0).tolist(),
        'harmed_per_label': harmed.sum(0).tolist(),
        'observed_fn_repaired': int((repaired & observed).sum()),
        'observed_fp_repaired': int((repaired & ~observed).sum()),
        'observed_fn_created': int((harmed & observed).sum()),
        'observed_fp_created': int((harmed & ~observed).sum()),
    }


@torch.inference_mode()
def diagnose(directory, device='cpu', batch_size=None, cpu_threads=4, output_dir=None):
    torch.set_num_threads(cpu_threads)
    directory = Path(directory).resolve()
    model, checkpoint = load_model(directory/'best.pt', device)
    if checkpoint['model_name'] != 'theraset':
        raise ValueError('TheraSet diagnostics require a TheraSet best.pt')
    config = checkpoint['config']
    source = MultimodalData(config['data']['root'], checkpoint['track'], checkpoint['statistics'])
    dataset = MultimodalPeptides(source, config['data']['features'], 'val',
                                limit=config['train'].get('limit_samples'),
                                peptide_feature_root=config['data'].get('peptide_features'))
    validate_cache(dataset, config, checkpoint['encoder'])
    batch_size = batch_size or config['train']['batch_size']
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_multimodal)
    collected = {}
    for step, batch in enumerate(loader, 1):
        out = model(to_device(batch, device), return_evidence=True)
        values = {'row_id': batch['row_id'], 'targets': batch['y'],
                  'direct_logits': out['direct'].float(), 'logits': out['logits'].float(),
                  'direct_probabilities': out['direct'].float().sigmoid(),
                  'probabilities': out['logits'].float().sigmoid()}
        if model.evidence_decoder:
            weights = out['prototype_weights']
            values.update(correction_gate=out['correction_gate'], correction=out['correction'],
                          prototype_weights=weights,
                          source_weights=(out['source_weights']*weights[..., None]).sum(2),
                          scale_weights=(out['scale_weights']*weights[..., None]).sum(2))
        for key, value in values.items():
            collected.setdefault(key, []).append(value.cpu().numpy().copy())
        if step == 1 or step % 25 == 0 or step == len(loader):
            print(f'Val diagnostics batches={step}/{len(loader)}', flush=True)
    collected = {key: np.concatenate(value) for key, value in collected.items()}
    y, direct, final = (collected[k] for k in ('targets', 'direct_probabilities', 'probabilities'))
    threshold = checkpoint['threshold']
    report = {
        'checkpoint': str(directory/'best.pt'), 'task': checkpoint['track'], 'split': 'val',
        'seed': config['train']['seed'], 'best_epoch': checkpoint['best_epoch'],
        'weight_source': checkpoint.get('weight_source', 'online'), 'rows': len(y),
        'diagnostic_subset': bool(config['train'].get('limit_samples') or config['train'].get('max_steps')),
        'threshold_fit': False,
        'scope': 'Same trained checkpoint. Direct head is not a separately trained CNN baseline. '
                 'All heads use the same threshold; zeros denote unannotated entries, not verified negatives. '
                 'Gate/attention averages describe readouts, not causal modality contributions.',
        'label_names': source.data.labels, 'train_support': source.statistics['labels']['support'],
        'source_names': ['motif', 'plm_context', 'properties', 'structure', 'pair'],
        'scale_names': ['kernel3', 'kernel5', 'kernel7', 'kernel9'],
        'comparisons': {}, 'groups_at_fixed05': {},
        'device': device, 'torch': str(torch.__version__), 'precision': 'fp32', 'batch_size': batch_size,
    }
    for name, cutoff in (('saved_threshold', threshold), ('fixed05', .5)):
        report['comparisons'][name] = {
            'threshold': cutoff, 'direct': metrics(y, direct, source.statistics, cutoff),
            'final': metrics(y, final, source.statistics, cutoff),
            'transitions': transitions(y, direct, final, cutoff),
        }
    stored = json.loads((directory/'results.json').read_text(encoding='utf-8'))
    measured = report['comparisons']['saved_threshold']['final']
    report['final_metric_deltas_vs_results'] = {
        name: measured[name]-stored['validation'][name]
        for name in ('accuracy', 'micro_f1', 'macro_f1', 'macro_ap', 'micro_ap')}
    for name, selected in source.groups(dataset.indices, y).items():
        if selected.any():
            report['groups_at_fixed05'][name] = {'rows': int(selected.sum()),
                **transitions(y[selected], direct[selected], final[selected], .5)}
    if model.evidence_decoder:
        report['mean_gate_per_label'] = collected['correction_gate'].mean(0).tolist()
        report['mean_abs_correction_per_label'] = np.abs(collected['correction']).mean(0).tolist()
        report['mean_source_weights_per_label'] = collected['source_weights'].mean(0).tolist()
        report['mean_scale_weights_per_label'] = collected['scale_weights'].mean(0).tolist()
        report['active_prototypes_per_label'] = model.active.sum(-1).cpu().tolist()
    output = Path(output_dir) if output_dir is not None else directory
    if output_dir is not None:
        output.mkdir(parents=True)
    np.savez_compressed(output/'evidence_validation.npz', **collected,
                        label_names=np.array(source.data.labels), ID=source.data.ids[dataset.indices],
                        cluster_id=source.data.clusters[dataset.indices], threshold=threshold,
                        epoch=checkpoint['best_epoch'], split='val')
    write_json(output/'evidence_diagnostics.json', report)
    print(f"Saved {output/'evidence_diagnostics.json'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, help='Write diagnostics in a separate new directory')
    args = parser.parse_args()
    diagnose(args.run, args.device, args.batch_size, args.cpu_threads, args.output_dir)


if __name__ == '__main__':
    main()
