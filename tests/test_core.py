import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from mfp_nova.data import CachedPeptides, PeptideData, collate_features
from mfp_nova.cli import main
from mfp_nova.encoder import FrozenEncoder
from mfp_nova.inference import evaluate, load_model, predict_sequences
from mfp_nova.losses import asymmetric_loss
from mfp_nova.metrics import evaluate_metrics, fit_global_threshold
from mfp_nova.model import NovaSequenceModel
from mfp_nova.trainer import seed_everything, train


def test_padding_and_backward():
    torch.set_num_threads(1)
    seed_everything(3)
    model = NovaSequenceModel(8, 3, hidden_dim=8, channels=4, dropout=0).eval()
    samples = [(torch.randn(2, 8), torch.tensor([1., 0., 1.])),
               (torch.randn(7, 8), torch.tensor([0., 1., 0.]))]
    x, mask, y = collate_features(samples)
    alone = model(samples[0][0][None], torch.ones(1, 2, dtype=torch.bool))
    batched = model(x, mask)
    torch.testing.assert_close(alone[0], batched[0])
    corrupted = x.clone()
    corrupted[~mask] = 123
    torch.testing.assert_close(model(corrupted, mask), batched)
    loss = asymmetric_loss(batched, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_asl_values_and_focusing_gradient():
    logits = torch.tensor([[0., 0.]], requires_grad=True)
    target = torch.tensor([[1., 0.]])
    loss = asymmetric_loss(logits, target, gamma_neg=2, clip=0)
    assert loss.item() == pytest.approx(np.log(2) * (1 + .25) / 2)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[-.25, .0625]]))
    extreme = torch.tensor([[-100., 100.]], requires_grad=True)
    asymmetric_loss(extreme, target).backward()
    assert torch.isfinite(extreme.grad).all()


def test_metric_semantics_and_threshold():
    y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
    p = np.array([[.5, .1, .01], [.5, .9, .01], [.5, .1, .01]])
    metrics = evaluate_metrics(y, p)
    assert metrics['label_ap'][0] == pytest.approx(average_precision_score(y[:, 0], p[:, 0]))
    assert metrics['label_ap'][2] is None
    assert metrics['label_positive_support'] == [2, 1, 0]
    assert metrics['macro_ap'] == pytest.approx((2/3 + 1) / 2)
    assert metrics['macro_f1'] == pytest.approx((.8 + 1 + 0) / 3)
    threshold = fit_global_threshold(y, p)
    assert .05 <= threshold <= .95
    assert evaluate_metrics(y, p, threshold)['sample_jaccard'] >= metrics['sample_jaccard']


def test_frozen_encoder_masks_and_limits(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import transformers

    class Batch(dict):
        def to(self, device):
            return Batch({k: v.to(device) for k, v in self.items()})

    class Tokenizer:
        def __call__(self, sequences, **kwargs):
            assert all(s == s.upper() for s in sequences)
            lengths = [len(s) + 2 for s in sequences]
            width = max(lengths)
            attention = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
            special = torch.zeros_like(attention)
            special[:, 0] = True
            for i, length in enumerate(lengths):
                special[i, length - 1:] = True
            return Batch(input_ids=attention.long(), attention_mask=attention.long(), special_tokens_mask=special.long())

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.config = SimpleNamespace(_commit_hash='test-revision', hidden_size=4)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=input_ids[..., None].float() * self.weight)

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    monkeypatch.setattr(transformers.AutoModel, 'from_pretrained', lambda *a, **k: Encoder())
    encoder = FrozenEncoder('test', max_length=8)
    states, mask = encoder.encode(['Aa', 'AcDE'])
    assert mask.sum(1).tolist() == [2, 4]
    assert not mask[:, 0].any()
    assert not states.requires_grad
    assert not encoder.model.training
    assert all(not p.requires_grad for p in encoder.model.parameters())
    with pytest.raises(ValueError, match='no silent truncation'):
        encoder.encode(['ACDEFGH'])
    with pytest.raises(ValueError, match='20-AA'):
        encoder.encode(['Ac-NH2'])
    make_fixture(tmp_path / 'data')
    config = {'encoder': {'name': 'test', 'revision': 'main', 'max_length': 8, 'batch_size': 2},
              'data': {'root': str(tmp_path / 'data'), 'track': 'fine39', 'augment': False,
                       'features': str(tmp_path / 'features')}}
    config_path = tmp_path / 'config.yaml'
    config['train'] = {'device': 'cpu'}
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    main(['features', '--config', str(config_path), '--batch-size', '1', '--splits', 'train', 'val'])
    cached = CachedPeptides(PeptideData(tmp_path / 'data'), tmp_path / 'features', 'train')
    assert len(cached) == 2
    assert cached.values.dtype == np.float16
    assert cached[0][0].shape == (2, 4)
    assert cached.meta['resolved_revision'] == 'test-revision'
    assert np.load(tmp_path / 'features/train/sequences.npy').tolist() == ['Aa', 'AC']
    assert PeptideData(tmp_path / 'data').d_mask[0, 1]
    # Live prediction must match the residue-only FP16 cache, despite BOS/EOS padding.
    head = NovaSequenceModel(4, 2, hidden_dim=8, channels=4, dropout=0).eval()
    checkpoint_path = tmp_path / 'head.pt'
    torch.save({'model_args': {'input_dim': 4, 'num_labels': 2, 'hidden_dim': 8, 'channels': 4, 'dropout': 0},
                'model_state': head.state_dict(), 'encoder': cached.meta, 'config': config,
                'labels': ['x', 'y'], 'threshold': .5}, checkpoint_path)
    sequences = ['Aa', 'AcDE']
    live = predict_sequences(checkpoint_path, sequences)
    assert [item['sequence'] for item in live] == sequences
    states, mask = encoder.encode(sequences)
    samples = [(states[i, mask[i]].half().float(), torch.zeros(2)) for i in range(2)]
    x, residue_mask, _ = collate_features(samples)
    with torch.inference_mode():
        expected = head(x, residue_mask).sigmoid().numpy()
    np.testing.assert_allclose([list(item['probabilities'].values()) for item in live], expected, rtol=1e-6)

    # Reproduce a failure after the first batch, using the legacy residues-only cache.
    original_encode = FrozenEncoder.encode
    interrupted_root = tmp_path / 'interrupted'
    feature_args = ['features', '--config', str(config_path), '--feature-root', str(interrupted_root),
                    '--batch-size', '1', '--splits', 'train', 'val']
    def fail_second_batch(self, sequences):
        if sequences == ['AC']:
            raise RuntimeError('simulated interrupted extraction')
        return original_encode(self, sequences)
    with monkeypatch.context() as patch:
        patch.setattr(FrozenEncoder, 'encode', fail_second_batch)
        with pytest.raises(RuntimeError, match='simulated interrupted'):
            main(feature_args)
    prefix = np.load(interrupted_root / 'train/residues.npy')[:2].copy()
    assert not (interrupted_root / 'train/metadata.json').exists()
    resumed_batches = []
    def record_resume(self, sequences):
        resumed_batches.extend(sequences)
        return original_encode(self, sequences)
    with monkeypatch.context() as patch:
        patch.setattr(FrozenEncoder, 'encode', record_resume)
        main(feature_args + ['--resume-from', '1'])
    assert 'Aa' not in resumed_batches
    np.testing.assert_array_equal(np.load(interrupted_root / 'train/residues.npy')[:2], prefix)
    recovered = CachedPeptides(PeptideData(tmp_path / 'data'), interrupted_root, 'train')
    assert len(recovered) == 2
    assert (interrupted_root / 'val/metadata.json').exists()
    with pytest.raises(FileExistsError, match='already complete'):
        main(feature_args + ['--resume-from', '1'])


def make_fixture(root):
    (root / 'arrays').mkdir(parents=True)
    (root / 'tracks/fine39').mkdir(parents=True)
    (root / 'manifest.json').write_text(json.dumps({'version_date': 'fixture'}), encoding='utf-8')
    rows = np.array([10, 20, 30, 40, 50, 60])
    ids = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
    seqs = np.array(['Aa', 'AC', 'Ag', 'AD', 'ZZ', 'AE'])
    d_mask = np.array([[c.islower() for c in s] for s in seqs], dtype=bool)
    np.savez(root / 'arrays/sequences.npz', row_id=rows, ID=ids, canonical_backbone=seqs,
             canonical_available=np.array([1, 1, 1, 1, 0, 1], bool), D_annotation_mask=d_mask, raw_sequence=seqs)
    np.savez(root / 'tracks/fine39/dataset.npz', row_id=rows, Y=[[1, 0], [0, 1], [1, 1], [0, 1], [1, 0], [1, 0]],
             label_names=['x', 'y'], split=['train', 'train', 'val', 'test', 'train', 'val'], cluster_id=rows,
             is_augmented=np.zeros(6, bool), row_in_scope=np.ones(6, bool),
             train_original_mask=np.array([1, 1, 0, 0, 1, 0], bool), val_original_mask=np.array([0, 0, 1, 0, 0, 1], bool),
             test_original_mask=np.array([0, 0, 0, 1, 0, 0], bool), train_aug_mask=np.array([1, 1, 0, 0, 1, 0], bool))
    return rows, ids


def test_cache_alignment_and_short_training(tmp_path):
    rows, ids = make_fixture(tmp_path / 'data')
    data = PeptideData(tmp_path / 'data')
    assert data.indices('train').tolist() == [0, 1]
    with pytest.raises(ValueError, match='train-only'):
        data.indices('val', augment=True)
    datasets = []
    for split, positions in [('train', [1, 0]), ('val', [5, 2])]:
        cache = tmp_path / 'features' / split
        cache.mkdir(parents=True)
        np.save(cache / 'row_ids.npy', rows[positions])
        np.save(cache / 'ids.npy', ids[positions])
        np.save(cache / 'sequences.npy', data.sequences[positions])
        np.save(cache / 'residues.npy', np.arange(len(positions) * 2 * 8, dtype=np.float32).reshape(-1, 8))
        np.save(cache / 'offsets.npy', np.arange(len(positions) + 1) * 2)
        (cache / 'metadata.json').write_text(json.dumps({
            'input_dim': 8, 'name': 'test', 'requested_revision': 'main', 'resolved_revision': 'fixture',
            'max_length': 80, 'dataset': 'fixture', 'extraction_dtype': 'float32', 'storage_dtype': 'float16'
        }), encoding='utf-8')
        datasets.append(CachedPeptides(data, tmp_path / 'features', split))
    assert datasets[0][0][1].tolist() == [1., 0.]
    assert datasets[0][0][0][0, 0].item() == 16
    loaders = [DataLoader(d, batch_size=2, collate_fn=collate_features) for d in datasets]
    config = {'train': {'device': 'cpu', 'epochs': 2, 'learning_rate': .001, 'weight_decay': 0, 'patience': 0,
                        'batch_size': 2, 'num_workers': 0, 'seed': 42, 'output': str(tmp_path)},
              'loss': {'name': 'asl', 'gamma_pos': 0, 'gamma_neg': 2, 'clip': .03},
              'data': {'root': str(tmp_path / 'data'), 'track': 'fine39', 'features': str(tmp_path / 'features'), 'augment': False},
              'encoder': {'name': 'test', 'revision': 'main', 'max_length': 80},
              'model': {'hidden_dim': 8, 'channels': 4, 'dropout': 0}}
    run = tmp_path / 'run'
    config_path = tmp_path / 'config.yaml'
    config['train']['epochs'] = 60
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    main(['train', '--config', str(config_path), '--name', 'run', '--epochs', '2',
          '--batch-size', '1', '--loss', 'bce', '--global-only', '--seed', '7'])
    result = json.loads((run / 'results.json').read_text())
    assert result['best_epoch'] in [1, 2]
    assert result['epochs_completed'] == 2
    assert result['elapsed_seconds'] > 0
    checkpoint = torch.load(run / 'best.pt', weights_only=True)
    assert checkpoint['config']['train']['seed'] == 7
    assert checkpoint['config']['train']['batch_size'] == 1
    assert checkpoint['config']['loss']['name'] == 'bce'
    assert not checkpoint['model_args']['use_local']
    restored = NovaSequenceModel(**checkpoint['model_args']).eval()
    restored.load_state_dict(checkpoint['model_state'])
    model, _ = load_model(run / 'best.pt', 'cpu')
    x, mask, _ = next(iter(loaders[1]))
    torch.testing.assert_close(model(x, mask), restored(x, mask))
    assert len((run / 'history.jsonl').read_text().splitlines()) == 2
    metadata = json.loads((run / 'metadata.json').read_text())
    assert metadata['trainable_parameters'] == sum(p.numel() for p in restored.parameters())
    assert 'transformers' in metadata['packages']
    main(['evaluate', '--checkpoint', str(run / 'best.pt'), '--data-root', str(tmp_path / 'data'),
          '--feature-root', str(tmp_path / 'features'), '--split', 'val', '--batch-size', '1',
          '--output', str(run / 'evaluation/val.json'), '--predictions', str(run / 'evaluation/val.npz')])
    report = json.loads((run / 'evaluation/val.json').read_text())
    assert report['split'] == 'val'
    with np.load(run / 'evaluation/val.npz', allow_pickle=False) as predictions:
        assert predictions['row_id'].tolist() == [30, 60]
        assert predictions['ID'].tolist() == ['c', 'f']
        assert predictions['label_names'].tolist() == ['x', 'y']
        np.testing.assert_array_equal(predictions['targets'], [[1, 1], [1, 0]])
        np.testing.assert_allclose(predictions['probabilities'], model(x, mask).sigmoid().detach().numpy())
        assert predictions['threshold'].item() == report['threshold']
        assert evaluate_metrics(predictions['targets'], predictions['probabilities'], report['threshold']) == report['metrics']
    with pytest.raises(FileExistsError):
        evaluate(run / 'best.pt', tmp_path / 'data', tmp_path / 'features', 'val', 'cpu', run / 'evaluation/val.json')
    with pytest.raises(FileExistsError):
        train(config, 'run')
    # Training succeeds with no Test cache, making accidental Test access visible.
    assert not (tmp_path / 'features/test').exists()


def test_real_data_contract():
    root = Path('data/therapeutic_2025')
    if not root.exists():
        pytest.skip('Local dataset package is not present')
    for track, nlabels in [('major15', 15), ('fine39', 39)]:
        data = PeptideData(root, track)
        assert len(data.rows) == 59231
        assert len(data.labels) == nlabels
        assert all(s and not set(s.upper()) - set('ACDEFGHIKLMNPQRSTVWY') for s in data.sequences[data.available])
        indices = data.indices('train')
        assert data.available[indices].all()
        assert not data.augmented[indices].any()
        assert data.y[indices].sum(1).min() > 0
        groups = [set(data.clusters[data.indices(split)]) for split in ('train', 'val', 'test')]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        backbones = [{s.upper() for s in data.sequences[data.indices(split)]} for split in ('train', 'val', 'test')]
        assert not (backbones[0] & backbones[1] or backbones[0] & backbones[2] or backbones[1] & backbones[2])
