import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from mfp_nova.data import CachedPeptides, PeptideData, collate_features
from mfp_nova.encoder import FrozenEncoder, prepare_features
from mfp_nova.inference import evaluate, load_model
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
    states, mask = encoder.encode(['AA', 'ACDE'])
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
    prepare_features(config, ['train', 'val'], 'cpu')
    cached = CachedPeptides(PeptideData(tmp_path / 'data'), tmp_path / 'features', 'train')
    assert len(cached) == 2
    assert cached.values.dtype == np.float16
    assert cached[0][0].shape == (2, 4)
    assert cached.meta['resolved_revision'] == 'test-revision'


def make_fixture(root):
    (root / 'arrays').mkdir(parents=True)
    (root / 'tracks/fine39').mkdir(parents=True)
    (root / 'manifest.json').write_text(json.dumps({'version_date': 'fixture'}), encoding='utf-8')
    rows = np.array([10, 20, 30, 40, 50])
    ids = np.array(['a', 'b', 'c', 'd', 'e'])
    seqs = np.array(['AA', 'AC', 'AG', 'AD', 'ZZ'])
    np.savez(root / 'arrays/sequences.npz', row_id=rows, ID=ids, canonical_backbone=seqs,
             canonical_available=np.array([1, 1, 1, 1, 0], bool), D_annotation_mask=np.zeros((5, 2), bool), raw_sequence=seqs)
    np.savez(root / 'tracks/fine39/dataset.npz', row_id=rows, Y=[[1, 0], [0, 1], [1, 1], [0, 1], [1, 0]],
             label_names=['x', 'y'], split=['train', 'train', 'val', 'test', 'train'], cluster_id=rows,
             is_augmented=np.zeros(5, bool), row_in_scope=np.ones(5, bool),
             train_original_mask=np.array([1, 1, 0, 0, 1], bool), val_original_mask=np.array([0, 0, 1, 0, 0], bool),
             test_original_mask=np.array([0, 0, 0, 1, 0], bool), train_aug_mask=np.array([1, 1, 0, 0, 1], bool))
    return rows, ids


def test_cache_alignment_and_short_training(tmp_path):
    rows, ids = make_fixture(tmp_path / 'data')
    data = PeptideData(tmp_path / 'data')
    assert data.indices('train').tolist() == [0, 1]
    with pytest.raises(ValueError, match='train-only'):
        data.indices('val', augment=True)
    datasets = []
    for split, positions in [('train', [1, 0]), ('val', [2])]:
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
    result = train(config, 'run')
    assert result['best_epoch'] in [1, 2]
    checkpoint = torch.load(run / 'best.pt', weights_only=True)
    restored = NovaSequenceModel(**checkpoint['model_args']).eval()
    restored.load_state_dict(checkpoint['model_state'])
    model, _ = load_model(run / 'best.pt', 'cpu')
    x, mask, _ = next(iter(loaders[1]))
    torch.testing.assert_close(model(x, mask), restored(x, mask))
    assert len((run / 'history.jsonl').read_text().splitlines()) == 2
    evaluate(run / 'best.pt', tmp_path / 'data', tmp_path / 'features', 'val', 'cpu', run / 'val.json')
    assert json.loads((run / 'val.json').read_text())['split'] == 'val'
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
        indices = data.indices('train')
        assert data.available[indices].all()
        assert not data.augmented[indices].any()
        assert data.y[indices].sum(1).min() > 0
        groups = [set(data.clusters[data.indices(split)]) for split in ('train', 'val', 'test')]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
