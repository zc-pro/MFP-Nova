"""Run with SA_MTP_TEST_REPO pointing to the author's repository with our patch applied."""
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from scripts.paper_repro import sa_mtp_therapep as sa


@pytest.fixture
def native_repo():
    repo = os.environ.get("SA_MTP_TEST_REPO")
    if not repo:
        pytest.skip("Set SA_MTP_TEST_REPO to the patched SA-MTP checkout")
    sys.path.insert(0, repo)
    try:
        yield Path(repo)
    finally:
        sys.path.remove(repo)


def inputs_fixture(root, labels):
    sequences = ["AC", "ACDEF", "G" * 74]
    root.mkdir()
    (root / "queries.fasta").write_text("".join(f">pep{i}\n{s}\n" for i, s in enumerate(sequences)))
    info = {"task": "therapep48" if labels == 48 else "therapep49_positive", "data_root": str(root),
            "labels": [f"label{i}" for i in range(labels)], "unique_sequences": 3,
            "splits": {s: {"rows": 4} for s in ("train", "val", "test")}}
    (root / "task.json").write_text(json.dumps(info))
    for split in ("train", "val", "test"):
        folder = root / split
        folder.mkdir()
        # The duplicate remains a separate row with a separate row_id/ID.
        indices = [0, 1, 2, 0]
        (folder / "seqs.fasta").write_text("".join(f">pep{i}\n{sequences[i]}\n" for i in indices))
        with (folder / "rows.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", "ID", "feature_id"])
            writer.writerows((i, f"{split}_{i}", f"pep{j}") for i, j in enumerate(indices))
        with (folder / "labels.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(info["labels"])
            for i in indices:
                y = [0] * labels
                y[i] = 1
                if labels == 49:
                    y[-1] = 1
                writer.writerow(y)
    return sequences


def cached_fixture(root, output, labels):
    sequences = inputs_fixture(root, labels)
    sa.prepare(root, output)
    (output / "features").mkdir()
    (output / "ss2").mkdir()
    _, offsets, _ = sa.feature_layout(sa.load_info(output))
    rng = np.random.default_rng(42)
    np.save(output / "features/esm2.npy", rng.normal(size=(offsets[-1], 1280)).astype(np.float32))
    (output / "features/esm2_completed.txt").write_text("pep0\npep1\npep2\n")
    for i, sequence in enumerate(sequences):
        # Native psipass2 emits independent sigmoid scores; they need not sum to 1.
        (output / f"ss2/pep{i}.ss2").write_text("# fixture only\n" + "".join(
            f"{j} {aa} C 0.7 0.2 0.2\n" for j, aa in enumerate(sequence, 1)))
    sa.graphs(output, 0)


def test_ss2_residue_alignment(tmp_path):
    path = tmp_path / "sample.ss2"
    path.write_text("1 A C 0.7 0.2 0.2\n2 C H 0.1 0.8 0.2\n")
    assert sa.read_ss2(path, "AC").shape == (2, 3)
    with pytest.raises(ValueError, match="residues differ"):
        sa.read_ss2(path, "CA")


def test_single_sequence_short_peptide_is_padded_then_cropped(tmp_path, monkeypatch):
    output = tmp_path / "run"
    inputs_fixture(tmp_path / "input", 48)
    sa.prepare(tmp_path / "input", output)
    padded_queries = []

    def fake_tools(command, stdout=None, **kwargs):
        executable = Path(command[0]).name
        if executable == "seq2mtx":
            padded_queries.append(sa.fasta(Path(command[1]))[0][1])
            stdout.write("fixture MTX\n")
        elif executable == "psipred":
            stdout.write("fixture pass1\n")
        elif executable == "psipass2":
            query = Path(command[5]).parent / "query.fasta"
            sequence = sa.fasta(query)[0][1]
            Path(command[5]).write_text("".join(
                f"{i} {aa} C 0.7 0.2 0.2\n" for i, aa in enumerate(sequence, 1)))
        else:
            raise AssertionError(command)

    monkeypatch.setattr(sa.subprocess, "run", fake_tools)
    sa.ss2(output, tmp_path / "prefix", 0)
    assert padded_queries[0] == "XACXX"
    assert sa.read_ss2(output / "ss2/pep0.ss2", "AC").shape == (2, 3)
    # A repeated request reuses the validated native output rather than rerunning the tools.
    sa.ss2(output, tmp_path / "prefix", 0)
    assert len(padded_queries) == 3


@pytest.mark.parametrize("labels", [48, 49])
def test_native_model_update_and_padding_mask(native_repo, tmp_path, labels):
    from models.tppred import TPMLC_single
    torch.set_num_threads(2)
    torch.manual_seed(42)
    output = tmp_path / "run"
    cached_fixture(tmp_path / "input", output, labels)
    dataset = sa.Peptides(output, "val")
    assert len(dataset) == 4
    np.testing.assert_array_equal(dataset[0][0], dataset[3][0])
    assert dataset[0][2].sum() == 48
    assert dataset[2][2].sum() == 0
    np.testing.assert_array_equal(sa.positions(74), np.r_[np.arange(25), np.arange(49, 74)])
    model = TPMLC_single(1280, labels, 50, 256, torch.device("cpu"), nhead=4,
                         n_enc_layers=2, n_dec_layers=2, dropout=0.1,
                         use_esm2=True, use_gat=True, use_learnable_lambda=True)
    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
    x, y, mask, tokens, graph = batch
    model.eval()
    with torch.no_grad():
        expected = model(x, mask, tokens, adj_matrix=graph)[0]
        altered = x.clone()
        altered[mask] = 10000
        actual = model(altered, mask, tokens, adj_matrix=graph)[0]
    torch.testing.assert_close(expected, actual)
    assert actual.shape == (2, labels)
    model.train()
    before = model.rp.label_embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    p = model(x, mask, tokens, adj_matrix=graph)[0]
    loss = torch.nn.functional.binary_cross_entropy(p, y) + model.get_film_regularization_loss(1e-4)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert not torch.equal(before, model.rp.label_embedding.weight)


def test_short_training_restores_checkpoint_and_exports_rows(native_repo, tmp_path, monkeypatch):
    torch.set_num_threads(2)
    output = tmp_path / "run"
    cached_fixture(tmp_path / "input", output, 48)
    # Fixture metadata only; the actual native model, train_epoch, BCE and AdamW are exercised.
    monkeypatch.setattr(sa.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(sa.subprocess, "check_output", lambda *a, **k: "fixture\n")
    sa.train(output, native_repo, 42, 2, 1, 2, 0)
    result = output / "training/seed42_epochs2_limit0"
    summary = json.loads((result / "comparison.json").read_text())
    checkpoint = torch.load(result / "best_model.pth", weights_only=True)
    assert summary["best_epoch"] == checkpoint["epoch"]
    assert summary["test_rows"] == 4
    with np.load(result / "test_predictions.npz") as archive:
        assert archive["row_ids"].tolist() == [0, 1, 2, 3]
        assert archive["ids"].tolist() == ["test_0", "test_1", "test_2", "test_3"]
        assert archive["probabilities"].shape == (4, 48)
        assert archive["thresholds"][47] == 0.5
