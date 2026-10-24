"""Synthetic correctness checks, never peptide performance measurements."""
import torch

from mfp_nova.models.helixset import HelixSet
from test_research import sample_statistics, synthetic_batch


def fixture():
    batch = synthetic_batch()
    model = HelixSet(12, 15, 10, sample_statistics(15), dim=32, heads=4, dropout=0, pair_dim=8)
    return model, batch


def change_geometry(batch):
    changed = {key: value.clone() for key, value in batch.items()}
    changed['geometry'] = torch.randn_like(batch['geometry'])
    changed['coords'] = 3 * torch.randn_like(batch['coords'])
    changed['struct_global'] = torch.randn_like(batch['struct_global'])
    changed['dssp'] = (batch['dssp'] + 1) % 9
    changed['edge_type'] = torch.zeros_like(batch['edge_type'])
    return changed


def test_geometry_cannot_change_main_evidence_and_missing_correction_is_zero():
    model, batch = fixture()
    model.eval()
    with torch.no_grad():
        output = model(batch, return_evidence=True)
        changed = model(change_geometry(batch), return_evidence=True)
        torch.testing.assert_close(output['direct'], changed['direct'])
        assert not torch.allclose(output['geometry_delta'][0], changed['geometry_delta'][0])
        assert output['geometry_gate'][1].count_nonzero() == 0
        assert output['geometry_delta'][1].count_nonzero() == 0
        assert output['geometry_delta'].abs().max() <= 2
        assert output['relation_delta'].abs().max() <= .5
        model.structure_evidence = False
        torch.testing.assert_close(model(batch)['logits'], model(change_geometry(batch))['logits'])


def test_short_peptide_prediction_is_independent_of_batch_padding():
    model, batch = fixture()
    model.eval()
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
        assert output['geometry_pair_attention'].count_nonzero() == 0
        assert output['geometry_delta'].count_nonzero() == 0


def test_loss_reaches_both_query_roles_and_geometry_branch():
    model, batch = fixture()
    loss, terms = model.compute_loss(batch, {'modality_dropout': 0, 'direct': .1})
    loss.backward()
    assert torch.isfinite(loss)
    assert terms['branch'].item() == 0
    for parameter in (model.sequence.plm.weight, model.skip_gate.weight,
                      model.latent_write.combine[0].weight, model.latent_blocks[0].attention.attention.in_proj_weight,
                      model.pair_read.queries.weight, model.geometry_output.weight,
                      model.pair_geometry.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert (model.label_queries.grad.abs().sum((0, 2)) > 0).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
