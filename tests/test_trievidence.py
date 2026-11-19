"""Check HelixSet exit parity and the packed train/evaluate path."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mfp_nova.models.helixset import HelixSet
from mfp_nova.models.theraset import TheraSet
from mfp_nova.models.trievidence import LocalLabelReadout, TriEvidence, empty_top10
from mfp_nova.multimodal import MultimodalData, MultimodalPeptides
from test_research import multimodal_fixture, peptide_cache_fixture, sample_statistics, synthetic_batch


def test_latent_exit_matches_full_helixset_and_only_cnn_receives_gradients():
    batch = synthetic_batch()
    stats = sample_statistics(15)
    common = dict(input_dim=12, num_labels=15, vocab_size=10, statistics=stats,
                  dim=16, heads=4, dropout=.2, pair_dim=8)
    helix_args = dict(common, sequence_depth=1, geometry_depth=1, pair_depth=1,
                      latent_depth=2, decoder_depth=2)
    full = HelixSet(**helix_args).eval()
    joint = TriEvidence(dict(common, cnn_channels=8), helix_args, cnn_channels=8).train()
    joint.helixset.load_state_dict({key: full.state_dict()[key] for key in joint.helixset.state_dict()})
    with torch.no_grad():
        expected = full(batch)["direct"]
    out = joint({key: value for key, value in batch.items() if key != "y"})
    torch.testing.assert_close(out["latent_logits"], expected, rtol=1e-6, atol=1e-6)
    assert not joint.helixset.training and not joint.theraset.training
    assert not hasattr(joint.helixset, "decoder")
    out["logits"].sum().backward()
    assert all(p.grad is None for p in joint.theraset.parameters())
    assert all(p.grad is None for p in joint.helixset.parameters())
    assert joint.aa34.classifier[-1].weight.grad.abs().sum() > 0


def test_empty_top10_preserves_nonempty_and_uses_only_anchor_candidates():
    anchor = torch.arange(12).float().repeat(3, 1)
    logits = -torch.ones_like(anchor)
    logits[0, 0] = -.1  # Best overall is outside TheraSet's top ten.
    logits[0, 4] = -.2
    logits[1, 1] = .2   # Already nonempty, including labels outside the top ten.
    predicted = empty_top10(logits, anchor)
    assert predicted.nonzero().tolist() == [[0, 4], [1, 1], [2, 7]]


def test_local_background_readout_masks_padding_and_preserves_gradients():
    torch.manual_seed(17)
    reader = LocalLabelReadout(8, 3, subtract_background=True)
    maps = torch.randn(2, 6, 8, requires_grad=True)
    mask = torch.tensor([[True, True, False, False, False, False], [True] * 6])
    logits = reader(maps, mask)
    torch.testing.assert_close(logits[:1], reader(maps[:1, :2], mask[:1, :2]))
    repeated = maps[:, :1].expand_as(maps)
    torch.testing.assert_close(reader(repeated, mask), torch.zeros_like(logits))
    logits.square().sum().backward()
    assert torch.isfinite(maps.grad).all()
    assert maps.grad[mask].abs().sum() > 0
    assert maps.grad[~mask].count_nonzero() == 0


@pytest.mark.parametrize("reader_head", ["global", "contrast"])
def test_packed_checkpoint_replays_validation_after_training(tmp_path, monkeypatch, reader_head):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_trievidence import main

    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    (root / "tracks/major15").rename(root / "tracks/fine39")
    peptide = peptide_cache_fixture(features)
    source = MultimodalData(root, "fine39")
    dataset = MultimodalPeptides(source, features, "val", peptide_feature_root=peptide)
    config = {"data": {}, "encoder": {"name": "fixture", "revision": "fixture", "max_length": 80},
              "peptide_encoder": {"name": "peptide_fixture", "revision": "fixture", "max_length": 80}}
    common = dict(input_dim=12, peptide_input_dim=6, num_labels=15,
                  vocab_size=len(source.statistics["chemical_vocab"])+2, statistics=source.statistics["labels"],
                  dim=16, heads=4, pair_dim=8, decoder_depth=1)
    args = (dict(common, cnn_channels=8), dict(common, latent_depth=2, pair_depth=1, geometry_depth=1))
    for name, cls, model_args, threshold in zip(("theraset", "helixset"), (TheraSet, HelixSet), args, (.5, .64)):
        torch.save({"model_name": name, "track": "fine39", "labels": source.data.labels,
                    "statistics": source.statistics, "config": dict(config, model_name=name),
                    "encoder": dataset.meta, "threshold": threshold, "model_args": model_args,
                    "model_state": cls(**model_args).state_dict()}, tmp_path / f"{name}.pt")
    data_args = ["--data-root", str(root), "--features", str(features), "--peptide-features", str(peptide),
                 "--device", "cpu", "--batch-size", "2"]
    head_args = ["--reader-head", reader_head]
    first, second = tmp_path / "train", tmp_path / "evaluate"
    monkeypatch.setattr(sys, "argv", ["trievidence", "train", *data_args, *head_args, "--epochs", "2", "--cnn-batch-size", "2",
                                     "--channels", "8", "--theraset", str(tmp_path / "theraset.pt"),
                                     "--helixset", str(tmp_path / "helixset.pt"), "--output", str(first)])
    main()
    with np.load(first / "val_predictions.npz") as export:
        identity = {k: export[k] for k in ("row_id", "ID", "cluster_id", "label_names", "targets")}
        for name, logits, cutoff, column in [("theraset", "theraset_logits", .5, "probabilities"),
                                             ("helix", "latent_logits", .64, "direct")]:
            np.savez(tmp_path / f"{name}.npz", **identity, threshold=cutoff, split="val",
                     **{column: torch.from_numpy(export[logits]).sigmoid().numpy()})
    cached = tmp_path / "cached"
    monkeypatch.setattr(sys, "argv", ["trievidence", "train-cached", "--data-root", str(root), "--device", "cpu",
                                     *head_args, "--epochs", "2", "--cnn-batch-size", "2", "--channels", "8",
                                     "--theraset-val", str(tmp_path / "theraset.npz"),
                                     "--helix-exits", str(tmp_path / "helix.npz"), "--output", str(cached)])
    main()
    packed = tmp_path / "packed"
    monkeypatch.setattr(sys, "argv", ["trievidence", "train", *data_args, *head_args, "--cnn-batch-size", "2", "--channels", "8",
                                     "--theraset", str(tmp_path / "theraset.pt"),
                                     "--helixset", str(tmp_path / "helixset.pt"),
                                     "--reader", str(cached / "reader.pt"), "--output", str(packed)])
    main()
    (tmp_path / "theraset.pt").unlink()
    (tmp_path / "helixset.pt").unlink()
    monkeypatch.setattr(sys, "argv", ["trievidence", "evaluate", str(packed / "best.pt"), *data_args,
                                     "--output", str(second)])
    main()
    with np.load(first / "val_predictions.npz") as before, np.load(second / "val_predictions.npz") as after:
        for key in ("row_id", "aa34_logits", "probabilities", "predicted", "predicted_raw"):
            np.testing.assert_array_equal(before[key], after[key])
    if reader_head == "global":
        import runpy

        shallow = tmp_path / "shallow"
        script = Path(__file__).resolve().parents[1] / "scripts/refine_trievidence_depth.py"
        monkeypatch.setattr(sys, "argv", [str(script), str(packed / "best.pt"), *data_args,
                                         "--depth", "1", "--epochs", "1", "--output", str(shallow)])
        runpy.run_path(str(script), run_name="__main__")
        saved = torch.load(shallow / "best.pt", map_location="cpu", weights_only=True)
        assert saved["model_args"]["helixset_args"]["latent_depth"] == 1
        monkeypatch.setattr(sys, "argv", ["trievidence", "evaluate", str(shallow / "best.pt"), *data_args,
                                         "--output", str(tmp_path / "shallow_evaluate")])
        main()
