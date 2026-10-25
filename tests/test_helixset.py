"""Synthetic correctness checks, never peptide performance measurements."""
import torch

from mfp_nova.models.helixset import HelixSet, annotation_set_loss
from mfp_nova.research import epoch_loss_settings
from mfp_nova.research_layers import unlabeled_weights
from test_research import sample_statistics, synthetic_batch


def fixture():
    batch = synthetic_batch()
    model = HelixSet(12, 15, 10, sample_statistics(15), dim=32, heads=4, dropout=0,
                    pair_dim=8, evidence_readout=True)
    return model, batch


def change_geometry(batch):
    changed = {key: value.clone() for key, value in batch.items()}
    changed['geometry'] = torch.randn_like(batch['geometry'])
    changed['coords'] = 3 * torch.randn_like(batch['coords'])
    changed['struct_global'] = torch.randn_like(batch['struct_global'])
    changed['dssp'] = (batch['dssp'] + 1) % 9
    changed['edge_type'] = torch.zeros_like(batch['edge_type'])
    return changed


def test_structure_ablation_removes_all_explicit_geometry_routes():
    model, batch = fixture()
    model.eval()
    with torch.no_grad():
        model.structure_evidence = False
        torch.testing.assert_close(model(batch)['logits'], model(change_geometry(batch))['logits'])


def test_short_peptide_prediction_is_independent_of_batch_padding():
    model, batch = fixture()
    model.eval()
    # Exercise the context-dependent gate, not just its constant 0.5 initialization.
    with torch.no_grad():
        model.depth_gate[-1].weight.normal_(std=.1)
    short = {key: value[:1].clone() for key, value in batch.items()}
    for key in ('esm', 'aa', 'd', 'residue_chem', 'mask', 'coords', 'geometry',
                'struct_mask', 'dssp', 'confidence_known'):
        short[key] = short[key][:, :3]
    short['edge_type'] = short['edge_type'][:, :3, :3]
    with torch.no_grad():
        torch.testing.assert_close(model(short)['logits'], model(batch)['logits'][:1], rtol=1e-4, atol=1e-5)


def test_single_residue_without_optional_modalities_is_finite():
    model, _ = fixture()
    batch = synthetic_batch(length=1)
    batch['struct_mask'].zero_()
    batch['struct_global_mask'].zero_()
    batch['phys_mask'].zero_()
    model.eval()
    with torch.no_grad():
        output = model(batch, return_evidence=True)
        assert torch.isfinite(output['logits']).all()
        assert output['pair_attention'].count_nonzero() == 0


def test_loss_reaches_pair_latent_and_geometry_branches():
    model, batch = fixture()
    loss, terms = model.compute_loss(batch, {'modality_dropout': 0, 'direct': .1,
                                           'evidence': .1, 'annotation_set': .05})
    loss.backward()
    assert torch.isfinite(loss)
    assert terms['branch'].item() > 0
    for parameter in (model.sequence.plm.weight,
                      model.latent_blocks[0].combine[0].weight,
                      model.decoder[0].pair_read.queries.weight,
                      model.pair_geometry.weight,
                      model.early_evidence.sequence_read.attention.in_proj_weight,
                      model.early_evidence.property_read.attention.in_proj_weight,
                      model.depth_gate[-1].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert model.label_queries.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_shallow_evidence_survives_geometry_changes_and_never_reads_targets():
    model, batch = fixture()
    model.eval()
    without_targets = {k: v for k, v in batch.items() if k != 'y'}
    with torch.no_grad():
        first = model(batch, return_evidence=True)
        changed = model(change_geometry(batch))
        torch.testing.assert_close(first['shallow_logits'], changed['shallow_logits'])
        torch.testing.assert_close(first['logits'], model(without_targets)['logits'])
        torch.testing.assert_close(first['deep_weight'], torch.full((3, 15), .5))
        torch.testing.assert_close(first['logits'], .5 * (first['shallow_logits'] + first['deep_logits']))


def test_missing_properties_cannot_enter_the_early_readout():
    model, batch = fixture()
    batch['phys_mask'].zero_()
    changed = dict(batch, phys=torch.full_like(batch['phys'], 1000))
    model.eval()
    with torch.no_grad():
        output = model(batch, return_evidence=True)
        assert output['early_modality_weights'][..., 1].count_nonzero() == 0
        torch.testing.assert_close(output['logits'], model(changed)['logits'])


def test_annotation_set_loss_gradients_and_empty_annotations():
    logits = torch.zeros(1, 2, requires_grad=True)
    targets = torch.tensor([[1., 0.]])
    stats = {'labels': ['antibacterial', 'antifungal']}
    loss = annotation_set_loss(logits, targets, stats)
    torch.testing.assert_close(loss, torch.tensor(1 - .5 / 1.15))
    loss.backward()
    assert logits.grad[0, 0] < 0 and logits.grad[0, 1] > 0
    empty = torch.tensor([[100., -100.]], requires_grad=True)
    zero = annotation_set_loss(empty, torch.zeros_like(empty), stats)
    zero.backward()
    assert zero.item() == 0 and empty.grad.count_nonzero() == 0
    assert torch.isfinite(annotation_set_loss(torch.tensor([[100., -100.]]), targets, stats))


def test_fine_unknown_keeps_weak_sibling_penalty_in_both_losses():
    targets = torch.tensor([[1., 0., 0.]])
    stats = {'labels': ['antibacterial__unknown', 'anti_gram+', 'antifungal__unknown']}
    weights = unlabeled_weights(targets, stats)
    torch.testing.assert_close(weights, torch.tensor([[.3, .075, .3]]))
    loss = annotation_set_loss(torch.zeros_like(targets), targets, stats)
    torch.testing.assert_close(loss, torch.tensor(1 - .5 / (1 + .5 * (.075 + .3))))


def test_new_auxiliary_terms_follow_existing_warmup():
    config = {'train': {'aux_warmup_epochs': 5},
              'loss': {'evidence': .1, 'annotation_set': .05, 'unlabeled_weight': .3}}
    first = epoch_loss_settings(config, 1)
    assert abs(first['evidence'] - .02) < 1e-8
    assert abs(first['annotation_set'] - .01) < 1e-8
    assert epoch_loss_settings(config, 5) == config['loss']


def test_original_checkpoint_has_no_new_required_parameters():
    args = dict(input_dim=12, num_labels=15, vocab_size=10, statistics=sample_statistics(15),
                dim=32, heads=4, pair_dim=8)
    original = HelixSet(**args)
    assert not any(k.startswith(('early_evidence.', 'depth_gate.', 'depth_bias')) for k in original.state_dict())
    HelixSet(**args, evidence_readout=False).load_state_dict(original.state_dict(), strict=True)
