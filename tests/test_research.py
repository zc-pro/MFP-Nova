import numpy as np
import pytest
import torch

from mfp_nova.multimodal import label_statistics, function_projection
from mfp_nova.research import make_model


def synthetic_batch(labels=15, length=7, input_dim=12):
    torch.set_num_threads(1)
    torch.manual_seed(8)
    b = 3
    mask = torch.ones(b, length, dtype=torch.bool)
    mask[0, 3:] = False
    smask = mask.clone()
    smask[1] = False
    edge = torch.zeros(b, length, length, dtype=torch.long)
    for i in range(length-1):
        edge[:, i, i+1] = edge[:, i+1, i] = 1
    edge[~smask[:, :, None].expand_as(edge)] = 0
    edge[~smask[:, None].expand_as(edge)] = 0
    geometry = torch.randn(b, length, 16)
    geometry[..., 6:9] = 1
    geometry[..., 9] = .8
    geometry[..., 12:15] = 1
    y = torch.zeros(b, labels)
    y[:, :3] = torch.eye(3)
    return {"esm": torch.randn(b, length, input_dim), "aa": torch.randint(1, 21, (b, length)),
            "d": torch.randint(0, 2, (b, length)), "residue_chem": torch.randint(0, 10, (b, length)),
            "chemical": torch.tensor([[2, 3], [4, 0], [0, 0]]), "mask": mask,
            "coords": torch.randn(b, length, 3), "geometry": geometry, "struct_mask": smask,
            "dssp": torch.randint(0, 9, (b, length)), "edge_type": edge,
            "phys": torch.randn(b, 515), "phys_mask": torch.ones(b, 515, dtype=torch.bool),
            "quality": torch.ones(b, 3, 6), "confidence_known": smask.clone(), "y": y,
            "struct_global": torch.randn(b, 25), "struct_global_mask": torch.ones(b, 25, dtype=torch.bool),
            "row_id": torch.arange(b)}


def sample_statistics(n):
    y = np.zeros((450, n), dtype=np.float32)
    y[:, 0] = 1; y[:220, 1] = 1; y[:80, 2] = 1; y[220:350, 3:5] = 1
    return label_statistics(y, np.arange(450), [f"label{i}" for i in range(n)], min_support=1)


@pytest.mark.parametrize("name", ["trirelay", "geofactor", "protoroute", "helixset"])
@pytest.mark.parametrize("labels", [15, 39])
def test_complete_model_masks_gradients(name, labels):
    stats = sample_statistics(labels)
    model = make_model(name, {"input_dim": 12, "num_labels": labels, "vocab_size": 10,
                             "statistics": stats, "dim": 32, "heads": 4, "dropout": 0,
                             **({"evidence_readout": True} if name == "helixset" else {})})
    batch = synthetic_batch(labels)
    model.eval()
    with torch.no_grad():
        out = model(batch)
        assert out["logits"].shape == (3, labels)
        corrupt = {k: v.clone() for k, v in batch.items()}
        corrupt["esm"][~batch["mask"]] = 100
        corrupt["geometry"][~batch["struct_mask"]] = 100
        corrupt["coords"][~batch["struct_mask"]] = 100
        torch.testing.assert_close(model(corrupt)["logits"], out["logits"])
        # A structure-free batch must remain finite and preserve sequence/chemistry paths.
        missing = dict(batch, struct_mask=torch.zeros_like(batch["struct_mask"]), confidence_known=torch.zeros_like(batch["confidence_known"]))
        assert torch.isfinite(model(missing)["logits"]).all()
        flipped = dict(batch, d=1-batch["d"])
        assert not torch.allclose(model(flipped)["logits"], out["logits"])
        if name == "trirelay":
            assert out["fusion"][1, :, 1].count_nonzero()==0
        if name == "protoroute":
            assert out["routing"][1, :, 1].count_nonzero()==0
            assert set(model.prototypes.active.sum(-1).flatten().tolist()) >= {1, 2, 4}
    model.train()
    total, terms = model.compute_loss(batch, {"mask_ratio": .5, "modality_dropout": .2})
    total.backward()
    assert torch.isfinite(total)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.sequence.plm.weight.grad.abs().sum()>0 if name == "trirelay" else True
    assert "classification" in terms
    if name == "trirelay":
        assert model.feedback[0].attention.in_proj_weight.grad.abs().sum()>0
        assert model.relations.factor[0][0].weight.grad.abs().sum()>0
    if name == "geofactor":
        assert model.factors.shared_factor[0].weight.grad.abs().sum()>0
    if name == "protoroute":
        assert model.prototypes.bank.grad.abs().sum()>0
        assert model.route[0].weight.grad.abs().sum()>0


def test_independent_function_projection():
    from mfp_nova.multimodal import independent_functions
    labels = ["anti_gram+", "anti_gram_", "antimicrobial__unknown", "antifungal__unknown"]
    projection = function_projection(labels, "fine39")
    y = np.array([[1, 1, 1, 0], [1, 0, 1, 1]])
    assert independent_functions(y, projection).sum(1).tolist() == [1, 2]


def multimodal_fixture(tmp_path):
    import csv
    import json
    root = tmp_path/"data"
    (root/"arrays").mkdir(parents=True)
    (root/"tracks/major15").mkdir(parents=True)
    n, length, channels = 8, 7, 12
    sequences = np.array(["ACDEFGH", "AcDEFGH"]*4)
    rows = np.arange(n)
    labels = ["antibacterial", "antifungal", "antiparasitic", "antiviral", "antimicrobial", "immunoactive", "anticancer", "antihypertensive", "thrombolytic", "neuropeptide", "antioxidant", "metabolic_regulatory", "growth_regeneration", "cell_communication", "drug_delivery"]
    y = np.zeros((n, 15), np.uint8)
    y[::2, :2] = 1
    y[1::2, 3] = 1
    train = rows<4
    np.savez(root/"tracks/major15/dataset.npz", row_id=rows, ID=rows.astype(str), Y=y, label_names=labels,
             split=np.where(train, "train", np.where(rows<6, "val", "test")), cluster_id=rows,
             is_augmented=np.zeros(n, bool), row_in_scope=np.ones(n, bool),
             train_original_mask=train, val_original_mask=(rows>=4)&(rows<6), test_original_mask=rows>=6, train_aug_mask=train)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    aa = np.array([[alphabet.index(c.upper())+1 for c in s] for s in sequences])
    d = np.array([[c.islower() for c in s] for s in sequences])
    np.savez(root/"arrays/sequences.npz", row_id=rows, ID=rows.astype(str), raw_sequence=sequences,
             canonical_backbone=sequences, canonical_available=np.ones(n, bool), D_annotation_mask=d, canonical_tokens=aa)
    rng = np.random.default_rng(8)
    phys = rng.normal(size=(n, 515)).astype(np.float32)
    phys[1, 480] = np.nan
    np.savez(root/"arrays/physchem_raw.npz", row_id=rows, X=phys, value_available=np.isfinite(phys), columns=np.arange(515).astype(str))
    smask = np.repeat((rows%2==0)[:, None], length, 1)
    coords = rng.normal(size=(n, length, 3)).astype(np.float32)
    node = np.where(smask, .8, np.nan).astype(np.float32)
    offsets = [0]
    us, vs = [], []
    for i in rows:
        if smask[i].any():
            us.extend(range(length-1)); vs.extend(range(1, length))
        offsets.append(len(us))
    np.savez(root/"arrays/structure_primary.npz", row_id=rows, CA_centered_A=coords, residue_mask=smask,
             structure_available=smask.any(1), confidence_verified=node*100, ASA_A2=node*100, RSA_unclipped=node,
             phi_degrees=node*100, psi_degrees=node*100, omega_degrees=node*100, dssp=np.where(smask, 0, -1),
             edge_offsets=offsets, edge_i=us, edge_j=vs, edge_type=np.ones(len(us), np.uint8),
             scalar_X=np.where(smask.any(1)[:, None], np.ones((n, 25), np.float32), np.nan))
    fields = ["row_id", "N-terminal modification", "C-terminal modification", "Post translation modifications", "HELM notation", "source_evidence"]
    with (root/"records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for i, s in enumerate(sequences):
            writer.writerow(dict(row_id=i, **{"HELM notation": "PEPTIDE1{[ac]."+".".join(s)+".[am]}$$$$", "source_evidence": "unknown"}))
    with (root/"symbolic_sequences.jsonl").open("w", encoding="utf-8") as f:
        for i, s in enumerate(sequences):
            f.write(json.dumps({"row_id": i, "parsed": True, "HELM_tokens": ["[ac]", *s, "[am]"]})+"\n")
    (root/"manifest.json").write_text(json.dumps({"version_date": "fixture"}))
    meta = {"input_dim": channels, "name": "fixture", "requested_revision": "main", "resolved_revision": "fixture", "max_length": 80,
            "dataset": "fixture", "extraction_dtype": "float32", "storage_dtype": "float16"}
    features = tmp_path/"features"
    for split, selected in [("train", rows[:4]), ("val", rows[4:6])]:
        directory = features/split; directory.mkdir(parents=True)
        np.save(directory/"row_ids.npy", rows[selected]); np.save(directory/"ids.npy", rows[selected].astype(str))
        np.save(directory/"sequences.npy", sequences[selected]); np.save(directory/"offsets.npy", np.arange(len(selected)+1)*length)
        np.save(directory/"residues.npy", rng.normal(size=(len(selected)*length, channels)).astype(np.float16))
        (directory/"metadata.json").write_text(json.dumps(meta))
    return root, features


@pytest.mark.parametrize("name", ["trirelay", "geofactor", "protoroute"])
def test_research_training_resume_inference(tmp_path, name, monkeypatch):
    import json
    from mfp_nova import research
    from mfp_nova.multimodal import MultimodalData, MultimodalPeptides, collate_multimodal
    torch.set_num_threads(1)
    root, features = multimodal_fixture(tmp_path)
    config = {"model_name": name, "model": {"dim": 32, "heads": 4, "dropout": 0},
              "data": {"root": str(root), "track": "major15", "features": str(features)},
              "encoder": {"name": "fixture", "revision": "main", "max_length": 80},
              "loss": {"mask_ratio": .5, "modality_dropout": .2},
              "train": {"device": "cpu", "seed": 42, "epochs": 1, "batch_size": 2, "learning_rate": .001,
                        "weight_decay": 0, "patience": 0, "warmup_epochs": 0, "num_workers": 0, "output": str(tmp_path)}}
    result = research.train(config, "first")
    assert result["epochs_completed"] == 1
    checkpoint = tmp_path/"first/best.pt"
    report = research.evaluate(checkpoint, root, features, "val", "cpu", tmp_path/"eval.json", batch_size=2)
    assert "groups" in report and "mfp_detection" in report["metrics"]
    source = MultimodalData(root, "major15", torch.load(checkpoint, weights_only=True)["statistics"])
    data = MultimodalPeptides(source, features, "val")
    batch = collate_multimodal([data[0], data[1]])
    np.savez(tmp_path/"inputs.npz", **{k: v.numpy() for k, v in batch.items() if k != "y"})
    prediction = research.predict_inputs(checkpoint, tmp_path/"inputs.npz")
    assert len(prediction)==2
    # Portable raw inputs use exactly the same checkpoint statistics as dataset inference.
    indices = data.indices
    raw = {"esm": batch["esm"].numpy(), "aa": batch["aa"].numpy(), "d": batch["d"].numpy(), "mask": batch["mask"].numpy(),
           "phys_raw": source.phys["X"][indices], "coords": source.struct["CA_centered_A"][indices],
           "struct_mask": source.struct["residue_mask"][indices], "dssp": source.struct["dssp"][indices],
           "ASA_A2": source.struct["ASA_A2"][indices], "RSA_unclipped": source.struct["RSA_unclipped"][indices],
           "confidence_verified": source.struct["confidence_verified"][indices], "struct_global_raw": source.struct["scalar_X"][indices],
           "angles_degrees": np.stack([source.struct[k][indices] for k in ("phi_degrees", "psi_degrees", "omega_degrees")], -1),
           "chemical_tokens": np.array([source.chemical[i] for i in indices]), "residue_chem_tokens": np.array([source.residue_chemical[i] for i in indices])}
    from mfp_nova.multimodal import prepare_inputs
    prepared = prepare_inputs(raw, source.statistics)
    # Fixture edges intentionally only contain backbone edges; compare numeric preprocessing.
    for key in ("phys", "geometry", "quality", "residue_chem", "struct_global"):
        torch.testing.assert_close(prepared[key], batch[key])
    np.savez(tmp_path/"raw_inputs.npz", **raw)
    assert len(research.predict_inputs(checkpoint, tmp_path/"raw_inputs.npz"))==2
    class Encoder:
        def __init__(self, *args):
            pass
        def encode(self, sequences):
            length = max(map(len, sequences))
            mask = torch.arange(length)[None]<torch.tensor([len(s) for s in sequences])[:, None]
            return torch.ones(len(sequences), length, 12), mask
    monkeypatch.setattr("mfp_nova.encoder.FrozenEncoder", Encoder)
    live = research.predict_sequences(checkpoint, ["AC", "AcD"])
    assert len(live)==2 and live[1]["sequence"]=="AcD"
    assert live[0]["evidence_scope"]=="sequence_and_stereochemistry_only"
    config["train"]["epochs"] = 2
    continued = research.train(config, "continued", resume=tmp_path/"first/last.pt")
    assert continued["epochs_completed"] == 2
    records = [json.loads(line) for line in (tmp_path/"continued/history.jsonl").read_text().splitlines()]
    assert records[0]["epoch"]==2
    assert not (features/"test").exists()
    with pytest.raises(FileExistsError):
        research.train(config, "first")


def test_extended_metric_definitions_and_unknown_supervision():
    from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score, matthews_corrcoef
    from mfp_nova.metrics import evaluate_metrics
    from mfp_nova.research_layers import observed_label_loss
    y = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0]])
    p = np.array([[.9, .4, .1], [.8, .3, .6], [.2, .8, .1]])
    m = evaluate_metrics(y, p, train_support=[1100, 300, 50])
    assert m["exact_match"]==accuracy_score(y, p>=.5)
    assert m["hamming_loss"]==hamming_loss(y, p>=.5)
    assert m["sample_jaccard"]==jaccard_score(y, p>=.5, average="samples")
    assert m["label_mcc"][0]==matthews_corrcoef(y[:, 0], p[:, 0]>=.5)
    assert m["label_auroc"][2] is None and m["tail_macro_ap"] is None
    stats = {"positive_weights": [1, 1], "labels": ["anticancer__unknown", "antibreastcancer"]}
    logits = torch.zeros(1, 2, requires_grad=True)
    observed_label_loss(logits, torch.tensor([[1., 0.]]), stats).backward()
    ambiguous = logits.grad[0, 1].item()
    logits.grad = None
    observed_label_loss(logits, torch.tensor([[0., 0.]]), stats).backward()
    assert logits.grad[0, 1].item()==pytest.approx(4*ambiguous)


def test_summary_keeps_protocols_and_aggregates_seeds(tmp_path):
    import json
    from mfp_nova.metrics import evaluate_metrics
    from mfp_nova.research import summarize_results
    paths, values = [], []
    for seed, probabilities in [(42, [[.9, .1], [.1, .9]]), (123, [[.1, .9], [.9, .1]])]:
        run = tmp_path/str(seed); run.mkdir(); paths.append(run)
        score = evaluate_metrics(np.eye(2), np.array(probabilities), train_support=[1200, 80])
        values.append(score["macro_ap"])
        config = {"model": {"dim": 256}, "loss": {"unlabeled_weight": .3}, "train": {"seed": seed, "epochs": 100}}
        metadata = {"model": "trirelay", "task": "major15", "seed": seed, "encoder": {"name": "fixture", "resolved_revision": "fixture"},
                    "dataset": "fixture", "cohort": "original_resolved_backbone", "config": config, "labels": ["a", "b"]}
        result = {"validation": score, "threshold_strategy": "global", "trainable_parameters": 100,
                  "elapsed_seconds": 2., "peak_cuda_memory_bytes": None}
        (run/"metadata.json").write_text(json.dumps(metadata)); (run/"results.json").write_text(json.dumps(result))
    summary = summarize_results(paths)
    assert len(summary["comparisons"])==1
    measured = summary["comparisons"][0]["metrics"]["macro_ap"]
    assert measured["mean"]==pytest.approx(np.mean(values))
    assert measured["std"]==pytest.approx(np.std(values, ddof=1))
    assert summary["comparisons"][0]["resources"]["peak_cuda_memory_bytes"] is None


@pytest.mark.parametrize("name,switch", [("trirelay", "feedback"), ("geofactor", "high_order"), ("protoroute", "dynamic_routing")])
def test_cli_model_selection(name, switch, monkeypatch):
    from mfp_nova.cli import main
    captured = {}
    def train(config, run, resume=None):
        captured.update(config=config, run=run)
    monkeypatch.setattr("mfp_nova.research.train", train)
    main(["train", "--model", name, "--task", "fine39", "--seed", "7", "--esm", "35m", "--ablate", switch])
    assert captured["config"]["data"]["track"]=="fine39"
    assert captured["config"]["encoder"]["name"]=="facebook/esm2_t12_35M_UR50D"
    assert captured["config"]["model"][switch] is False
    assert captured["run"]==f"{name}_fine39_seed7"


@pytest.mark.parametrize("name", ["trirelay", "geofactor", "protoroute"])
def test_research_ablation_paths(name):
    flags = {"trirelay": {"feedback": False, "relations": False, "high_order": False, "quality_aware": False, "property_geometry": False},
             "geofactor": {"structured": False, "pairwise": False, "high_order": False, "spatial": False, "quality_aware": False},
             "protoroute": {"shared_prototypes": False, "relations": False, "high_order": False, "quality_aware": False, "dynamic_routing": False}}[name]
    batch = synthetic_batch()
    model = make_model(name, {"input_dim": 12, "num_labels": 15, "vocab_size": 10,
                             "statistics": sample_statistics(15), "dim": 32, "heads": 4, "dropout": 0, **flags})
    output = model(batch)
    torch.testing.assert_close(output["logits"], output["direct"])
    total, _ = model.compute_loss(batch, {"mask_ratio": .2})
    total.backward()
    assert torch.isfinite(total)
