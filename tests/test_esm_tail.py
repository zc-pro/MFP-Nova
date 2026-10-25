"""Prefix/tail numerical equivalence and LoRA gradients; no network downloads."""
import copy

import pytest
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, EsmConfig, EsmModel

from mfp_nova.encoder import FrozenEncoder, prepare_features
from mfp_nova.esm_tail import ESMTail
from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
from mfp_nova.research import checkpoint_state, restore_model_state, validate_cache
from test_research import multimodal_fixture


def local_esm(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = EsmConfig(vocab_size=33, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                       num_attention_heads=4, pad_token_id=1, mask_token_id=32,
                       position_embedding_type='rotary', token_dropout=False,
                       hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    config._attn_implementation = 'eager'
    config._commit_hash = 'fixture'
    full = EsmModel(config, add_pooling_layer=False).eval()

    class Batch(dict):
        def to(self, device):
            return Batch({k: v.to(device) for k, v in self.items()})

    class Tokenizer:
        def __call__(self, sequences, **kwargs):
            lengths = torch.tensor([len(s) + 2 for s in sequences])
            ids = torch.ones(len(sequences), int(lengths.max()), dtype=torch.long)
            for i, seq in enumerate(sequences):
                ids[i, 0] = 0
                ids[i, 1:len(seq) + 1] = torch.tensor(['ACDEFGHIKLMNPQRSTVWY'.index(c) + 4 for c in seq])
                ids[i, len(seq) + 1] = 2
            mask = torch.arange(ids.shape[1])[None] < lengths[:, None]
            special = (ids == 0) | (ids == 1) | (ids == 2)
            return Batch(input_ids=ids, attention_mask=mask.long(), special_tokens_mask=special.long())

    monkeypatch.setattr(AutoModel, 'from_pretrained', lambda *a, **k: copy.deepcopy(full))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: Tokenizer())
    return full, Tokenizer()


def test_prefix_plus_zero_lora_matches_full_esm_and_ignores_padding(monkeypatch):
    full, tokenizer = local_esm(monkeypatch)
    encoder = FrozenEncoder('fixture', representation_layer=2)
    tail = ESMTail('fixture', 'fixture', representation_layer=2, rank=4, alpha=8, dropout=0).eval()
    sequences = ['ACD', 'ACDEF']
    states, mask = encoder.encode(sequences)
    lengths = mask.sum(1)
    batch = {'esm': states[:, 1:-1].clone() * mask[:, 1:-1, None], 'mask': mask[:, 1:-1].clone(),
             'esm_cls': states[:, 0].clone(),
             'esm_eos': states[torch.arange(2), lengths + 1].clone()}
    tokens = tokenizer(sequences)
    tokens.pop('special_tokens_mask')
    with torch.no_grad():
        expected = full(**tokens).last_hidden_state[:, 1:-1] * batch['mask'][..., None]
        torch.testing.assert_close(tail(batch), expected, atol=1e-5, rtol=1e-4)
        corrupted = dict(batch, esm=batch['esm'].masked_fill(~batch['mask'][..., None], 100))
        torch.testing.assert_close(tail(corrupted), expected, atol=1e-5, rtol=1e-4)
    # Cached inference tensors must be normal training tensors before backward.
    tail.train()
    tail(batch)[..., 0].sum().backward()
    assert tail.layers[0].attention.self.query.lora_B.weight.grad.abs().sum() > 0
    assert all(p.grad is None for key, p in tail.named_parameters() if '.lora_' not in key)
    assert all(torch.isfinite(p.grad).all() for p in tail.parameters() if p.grad is not None)


def test_compact_checkpoint_restores_adapters_and_rejects_missing_trainable_keys(monkeypatch):
    local_esm(monkeypatch)
    def model():
        result = nn.Module()
        result.esm_tail = ESMTail('fixture', 'fixture', representation_layer=2, rank=4, alpha=8)
        return result
    first, second = model(), model()
    with torch.no_grad():
        first.esm_tail.layers[0].attention.self.query.lora_B.weight.fill_(.02)
    state = checkpoint_state(first, cpu=True)
    assert 'esm_tail.layers.0.attention.self.query.base.weight' not in state
    restore_model_state(second, state)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key])
    state.pop('esm_tail.layers.0.attention.self.query.lora_B.weight')
    with pytest.raises(RuntimeError, match='Checkpoint/model mismatch'):
        restore_model_state(second, state)


def test_prefix_cache_special_tokens_follow_dataset_rows(monkeypatch, tmp_path):
    local_esm(monkeypatch)
    root, _ = multimodal_fixture(tmp_path)
    features = tmp_path / 'prefix_features'
    config = {'encoder': {'name': 'fixture', 'revision': 'fixture', 'max_length': 80,
                           'batch_size': 2, 'representation_layer': 2},
              'data': {'root': str(root), 'track': 'major15', 'augment': False, 'features': str(features)}}
    prepare_features(config, ['train'], 'cpu')
    source = MultimodalData(root, 'major15')
    dataset = MultimodalPeptides(source, features, 'train')
    validate_cache(dataset, config)
    batch = collate_multimodal([dataset[2], dataset[0]])
    encoder = FrozenEncoder('fixture', revision='fixture', representation_layer=2)
    sequences = source.data.sequences[dataset.indices[[2, 0]]].tolist()
    states, mask = encoder.encode(sequences)
    torch.testing.assert_close(batch['esm_cls'], states[:, 0].half().float())
    eos = states[torch.arange(2), mask.sum(1) + 1].half().float()
    torch.testing.assert_close(batch['esm_eos'], eos)
    wrong = copy.deepcopy(config)
    wrong['encoder'].pop('representation_layer')
    with pytest.raises(ValueError, match='prefix cache'):
        validate_cache(dataset, wrong)
