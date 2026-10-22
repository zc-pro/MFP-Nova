"""Observe actual first-read TriRelay heads on complete Val, without retraining."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from .private_evidence import METRICS, difference, metric, package_export
from .research import load_model, metrics, to_device, validate_cache
from .trainer import write_json


MODALITIES = ('sequence', 'geometry', 'properties')
HEADS = tuple(f'{path}_{modality}' for path in ('shared', 'private', 'mixed') for modality in MODALITIES) + ('direct', 'final')


@torch.inference_mode()
def collect_readouts(model, loader, device):
    """One ordinary full-model pass, with additional reads of existing tensors."""
    model.eval()
    targets, rows, predictions, available, gates, fusion = [], [], [], [], [], []
    for step, batch in enumerate(loader, 1):
        out = model(to_device(batch, device), first_readouts=True)
        # Each per-modality classifier is the existing TailClassifier. Do not
        # invent a shared-only fusion head that was never trained as such.
        # Keep final sigmoid's original [batch, labels] layout: applying it
        # after packing all heads can change vectorization rounding/tied ranks.
        probabilities = torch.cat([out[key].float().sigmoid() for key in
                                   ('shared_branch','unmixed_private_branch','first_mixed_branch')] +
                                  [out[key].float().sigmoid()[:, None] for key in ('direct','logits')], 1)
        predictions.append(probabilities.cpu().numpy())
        targets.append(batch['y'].numpy())
        rows.append(batch['row_id'].numpy())
        present = out['available'].cpu().numpy()
        available.append(np.concatenate([present, present, present, np.ones((len(present), 2), bool)], 1))
        gates.append(out['first_private_gate'].float().cpu().numpy())
        fusion.append(out['first_fusion'].float().cpu().numpy())
        if step == 1 or step % 25 == 0 or step == len(loader):
            print(f'  Val batches={step}/{len(loader)}', flush=True)
    return {'row_id': np.concatenate(rows), 'targets': np.concatenate(targets),
            'probabilities': np.concatenate(predictions), 'head_available': np.concatenate(available),
            'first_private_gate': np.concatenate(gates), 'first_fusion': np.concatenate(fusion)}


def summarize_heads(collected, statistics, threshold):
    """AP is the primary branch diagnostic; never fit branch thresholds."""
    y, p, available = (collected[k] for k in ('targets', 'probabilities', 'head_available'))
    common = available.all(1)
    head_reports = {}
    for j, name in enumerate(HEADS):
        valid = available[:, j]
        report = {'rows': int(valid.sum()),
                  'metrics': metrics(y[valid], p[valid, j], statistics, threshold) if valid.any() else None,
                  'common_available_rows': int(common.sum()),
                  'common_available_metrics': metrics(y[common], p[common, j], statistics, threshold) if common.any() else None}
        head_reports[name] = report
    paired = {}
    for modality in MODALITIES:
        shared, private, mixed = [head_reports[f'{path}_{modality}']['metrics'] for path in ('shared','private','mixed')]
        paired[modality] = {f'{name}_minus_shared': {key:difference(metric(value,key),metric(shared,key)) for key in METRICS}
                           for name,value in (('private',private),('mixed',mixed))}
    # Gates/weights summarize only available modalities. A large gate alone is
    # not evidence of a large causal contribution or gradient domination.
    gate_summary = {}
    for j, modality in enumerate(MODALITIES):
        valid = available[:, j]
        gate_summary[modality] = {'rows':int(valid.sum()),
            'mean_gate_per_label':collected['first_private_gate'][valid,:,j].mean(0).tolist() if valid.any() else None,
            'mean_fusion_per_label':collected['first_fusion'][valid,:,j].mean(0).tolist() if valid.any() else None}
    return head_reports, paired, gate_summary


def export(runs, output, device='cpu', batch_size=64, data_root='data/therapeutic_2025', feature_root=None, cpu_threads=None):
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for directory in map(Path, runs):
        started = time.perf_counter()
        stored = json.loads((directory/'results.json').read_text(encoding='utf-8'))
        model, checkpoint = load_model(directory/'best.pt', device)
        if checkpoint['model_name'] != 'trirelay' or not hasattr(model, 'evidence_partition'):
            raise ValueError('First-read diagnostics require a shared-private TriRelay checkpoint')
        source = MultimodalData(data_root, checkpoint['track'], checkpoint['statistics'])
        dataset = MultimodalPeptides(source, feature_root or checkpoint['config']['data']['features'], 'val')
        validate_cache(dataset, checkpoint['config'], checkpoint['encoder'])
        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_multimodal)
        print(f"{checkpoint['track']} | first-read heads | {directory/'best.pt'}", flush=True)
        collected = collect_readouts(model, loader, device)
        threshold = checkpoint['threshold']
        heads, paired, gates = summarize_heads(collected, source.statistics, threshold)
        reproduction = {key:difference(metric(heads['final']['metrics'], key), metric(stored['validation'], key)) for key in METRICS}
        report = {'run':str(directory), 'checkpoint':str(directory/'best.pt'), 'model':'trirelay',
                  'task':checkpoint['track'], 'seed':checkpoint['config']['train']['seed'], 'split':'val',
                  'rows':len(dataset), 'labels':source.data.labels, 'threshold':threshold,
                  'saved_threshold_matches_results':np.array_equal(threshold, stored['threshold']),
                  'weight_source':checkpoint.get('weight_source','online'), 'best_epoch':checkpoint['best_epoch'],
                  'branch_supervision':checkpoint['config']['train'].get('branch_supervision','mixed_private'),
                  'evidence_fusion':model.evidence_partition.fusion,
                  'git_commit_trained':json.loads((directory/'metadata.json').read_text(encoding='utf-8'))['git_commit'],
                  'threshold_fit':False, 'weight_updates':False, 'inference_intervention':False,
                  'scope':'First read before mixed-label feedback; final follows the unchanged complete model',
                  'head_definitions':{
                      'shared_*':'Per-modality unmixed shared evidence after relay, before feedback; RMS-normalized in normalized_competitive/normalized_residual modes; existing classifier',
                      'private_*':'Per-modality unmixed private evidence, ungated, before feedback; RMS-normalized in normalized_competitive/normalized_residual modes; existing classifier',
                      'mixed_*':('(1-g)*normalized_shared + g*normalized_private' if model.evidence_partition.fusion=='normalized_competitive' else
                                 'normalized_shared + g*normalized_private' if model.evidence_partition.fusion=='normalized_residual' else
                                 'shared + gated private')+' before feedback; diagnostic, not a separately supervised first-read mixed branch',
                      'direct':'Original first-read learned multimodal fusion classifier, supervised by direct loss',
                      'final':'Original final prediction after feedback and label relations'},
                  'threshold_scope':'All heads use the unchanged final-checkpoint threshold; head-specific calibration is not fitted. Prioritize AP for branch comparisons.',
                  'availability_scope':'Per-head metrics exclude missing modality rows. common_available_metrics uses identical rows for all heads; neither is a new split.',
                  'probability_axes':['row_id','head_names','label_names'],
                  'heads':heads, 'same_modality_deltas':paired, 'first_gate_and_fusion':gates,
                  'final_reproduction_deltas':reproduction,
                  'device':device, 'torch':str(torch.__version__),
                  'gpu':torch.cuda.get_device_name(device) if torch.device(device).type=='cuda' else None,
                  'batch_size':batch_size, 'precision':'fp32', 'cpu_threads':torch.get_num_threads(),
                  'elapsed_seconds':time.perf_counter()-started}
        destination = output/directory.name
        destination.mkdir()
        write_json(destination/'branches.json', report)
        np.savez_compressed(destination/'readouts.npz', **collected, head_names=np.array(HEADS),
                            label_names=np.array(source.data.labels), ID=source.data.ids[dataset.indices],
                            threshold=threshold, split='val', task=checkpoint['track'])
        for name, head in heads.items():
            m = head['metrics']
            print(f"  {name:20} n={head['rows']:5} MacroAP={m['macro_ap']:.6f} MicroAP={m['micro_ap']:.6f} "
                  f"Jaccard={m['accuracy']:.6f} MFP-F1={m['mfp_detection']['f1']:.6f}" if m else
                  f"  {name:20} n=0 metrics=unavailable", flush=True)
        maximum = max(abs(v) for v in reproduction.values() if v is not None)
        print(f"  Final reproduction max difference={maximum:.3g}; threshold={threshold}; no fitting or updates", flush=True)
        summaries.append(report)
        del model
    summary = {'split':'val', 'weight_updates':False, 'threshold_fit':False,
               'scope':'Observed first-read heads, not private-off interventions or retrained branch-only models',
               'comparisons':summaries}
    write_json(output/'summary.json', summary)
    archive = package_export(runs, output)
    print(f'Completed. Transfer: {archive}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--output', required=True, help='New output directory; ZIP is written beside it')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--data-root', default='data/therapeutic_2025')
    parser.add_argument('--feature-root')
    args = parser.parse_args()
    export(args.runs, args.output, args.device, args.batch_size, args.data_root, args.feature_root, args.cpu_threads)


if __name__ == '__main__':
    main()
