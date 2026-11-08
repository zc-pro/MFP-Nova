"""Data correctness tests; no ESM downloads or GPU work."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mfp_nova.rich48 import build_rich48
from mfp_nova.therapep import LABELS, configure_task


def fixture(root):
    author, source = root/"author", root/"source"
    author.mkdir()
    (source/"arrays").mkdir(parents=True)
    sequences = ["ACDE", "ACDE", "FGHI", "FGHI", "WWKK"]
    n, width = len(sequences), 4
    fields = ["row_id", "ID", "Sequence", "canonical_backbone_available", "is_augmented",
              "Is_natural_peptide", "HELM notation", "N-terminal modification",
              "C-terminal modification", "Post translation modifications", "source_evidence", "cluster_id"]
    with (source/"records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, seq in enumerate(sequences):
            writer.writerow(dict.fromkeys(fields, "") | {
                "row_id": i, "ID": f"original{i}", "Sequence": seq,
                "canonical_backbone_available": "True", "is_augmented": "False",
                "Is_natural_peptide": "False" if i == 1 else "True",
                "source_evidence": "unknown", "cluster_id": i})
    with (source/"symbolic_sequences.jsonl").open("w", encoding="utf-8") as f:
        for i, seq in enumerate(sequences):
            f.write(json.dumps({"row_id": i, "HELM_tokens": list(seq), "parsed": True})+"\n")
    aa = np.array([["ACDEFGHIKLMNPQRSTVWY".index(c)+1 for c in s] for s in sequences], np.uint8)
    np.savez(source/"arrays/sequences.npz", row_id=np.arange(n), canonical_backbone=sequences,
             canonical_available=np.ones(n, bool), canonical_tokens=aa,
             D_annotation_mask=np.zeros((n, width), bool), alphabet=list("ACDEFGHIKLMNPQRSTVWY"))
    x = np.full((n, 515), 10., np.float32)
    x[1] = 20  # Same sequence, wrong source for the natural author record.
    x[4] = 1000  # Val must not enter the Train fit.
    np.savez(source/"arrays/physchem_raw.npz", row_id=np.arange(n), X=x,
             value_available=np.ones_like(x, bool), columns=np.arange(515).astype(str))
    scalar = np.full((n, 25), 3., np.float32)
    angles = np.full((n, width), 30., np.float32)
    np.savez(source/"arrays/structure_primary.npz", row_id=np.arange(n),
             CA_centered_A=np.arange(n*width*3, dtype=np.float32).reshape(n, width, 3),
             residue_mask=np.ones((n, width), bool), structure_available=np.ones(n, bool),
             coordinate_candidate_available=np.ones(n, bool), raw_structure_index=np.arange(n),
             dssp=np.zeros((n, width), np.int8), dssp_code_names=np.arange(9).astype(str),
             phi_degrees=angles, psi_degrees=angles, omega_degrees=angles,
             ASA_A2=angles, RSA_unclipped=angles/100, confidence_verified=angles*2,
             scalar_X=scalar, scalar_value_available=np.ones_like(scalar, bool),
             scalar_columns=np.arange(25).astype(str), edge_offsets=np.arange(n+1),
             edge_i=np.zeros(n, np.int16), edge_j=np.ones(n, np.int16),
             edge_type=np.ones(n, np.uint8), edge_distance_A=np.arange(n, dtype=np.float32)+1)
    for split, entries in {"train": [("ACDE", True), ("ACDE", False), ("FGHI", True), ("RRKK", True)],
                           "val": [("WWKK", True)], "test": [("RRKK", True)]}.items():
        with (author/f"{split}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Sequence", "Label encoding", "is_therapeutic", "source"])
            writer.writeheader()
            for sequence, positive in entries:
                target = [0]*49
                if positive:
                    target[18] = target[48] = 1
                writer.writerow({"Sequence": sequence, "Label encoding": json.dumps(target),
                                 "is_therapeutic": str(positive),
                                 "source": ("TheraPepDB_recovered" if sequence == "RRKK" else "TheraPepDB")
                                           if positive else "synthetic_uniform"})
    return author, source


class Rich48Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.author, self.source = fixture(self.root)
        self.output = self.root/"rich48"
        self.result = build_rich48(self.author, self.source, self.output)

    def test_members_targets_mapping_and_sparse_edges(self):
        with np.load(self.output/"tracks/rich48/dataset.npz") as z:
            np.testing.assert_array_equal(z["row_id"], [0, 2, 3, 4, 5])
            np.testing.assert_array_equal(z["split"], ["train"]*3+["val", "test"])
            self.assertEqual(z["Y"].shape, (5, 48))
            self.assertEqual(z["label_names"].tolist(), LABELS[:48])
        with np.load(self.output/"arrays/physchem_raw.npz") as z:
            self.assertEqual(z["X"][0, 0], 10.)
            self.assertTrue(np.isnan(z["X"][[1, 2, 4]]).all())
            self.assertFalse(z["value_available"][[1, 2, 4]].any())
        with np.load(self.output/"arrays/structure_primary.npz") as z:
            np.testing.assert_array_equal(z["edge_offsets"], [0, 1, 1, 1, 2, 2])
            np.testing.assert_array_equal(z["edge_distance_A"], [1., 5.])
            self.assertFalse(z["residue_mask"][[1, 2, 4]].any())
            self.assertTrue(np.isnan(z["scalar_X"][[1, 2, 4]]).all())
        self.assertEqual(self.result["splits"]["train"]["ambiguous"], 1)
        self.assertEqual(self.result["splits"]["train"]["unmatched"], 1)
        with self.assertRaises(FileExistsError):
            build_rich48(self.author, self.source, self.output)

    def test_configuration_enables_multimodal_task(self):
        config = {"model_name": "theraset", "data": {"root": "data/therapeutic_2025",
                  "features": "features/esm2_650m", "augment": True}, "model": {"structure_evidence": True}}
        configure_task(config, "rich48")
        self.assertEqual(config["data"]["root"], "data/therapeutic_2025/rich48")
        self.assertEqual(config["data"]["features"], "features/rich48/esm2_650m")
        self.assertTrue(config["model"]["structure_evidence"])
        self.assertEqual(config["train"]["metric_suite"], "micro_f1")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for tensor loading")
    def test_train_only_preprocessing_and_multimodal_samples(self):
        import torch
        from mfp_nova.multimodal import MultimodalData, collate_multimodal
        source = MultimodalData(self.output, "rich48")
        self.assertEqual(source.statistics["train_rows"], 3)
        self.assertIsNone(source.statistics["function_projection"])
        self.assertNotIn("therapep_background_index", source.statistics["labels"])
        self.assertEqual(source.statistics["phys"]["impute"][0], 10.)
        self.assertEqual(source.statistics["phys"]["mean"][0], 10.)
        samples = [source.sample(i, torch.ones(4, 8)) for i in range(5)]
        self.assertTrue(samples[0]["struct_mask"].all())
        self.assertFalse(samples[1]["phys_mask"].any())
        self.assertFalse(samples[2]["struct_global_mask"].any())
        batch = collate_multimodal(samples)
        self.assertEqual(tuple(batch["y"].shape), (5, 48))
        self.assertTrue(torch.isfinite(batch["geometry"]).all())
        self.assertNotIn("multifunction", source.groups(np.arange(5), source.data.y))
        with self.assertRaises(ValueError):
            source.data.indices("train", augment=True)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for cache loading")
    def test_full_author_cache_reuse_with_background_gaps(self):
        from mfp_nova.data import PeptideData, CachedPeptides
        from mfp_nova.therapep import DATASET, read_data
        data = PeptideData(self.output, "rich48")
        author = read_data(self.author)
        cache = self.root/"features/train"
        cache.mkdir(parents=True)
        order = np.flatnonzero(author["splits"] == "train")[::-1]
        rows = author["rows"][order]
        np.save(cache/"row_ids.npy", rows)
        np.save(cache/"ids.npy", author["ids"][order])
        np.save(cache/"sequences.npy", author["sequences"][order])
        np.save(cache/"offsets.npy", np.arange(len(order)+1)*4)
        np.save(cache/"residues.npy", np.repeat(rows.astype(np.float32), 4)[:, None])
        (cache/"metadata.json").write_text(json.dumps({"dataset": DATASET}))
        loaded = CachedPeptides(data, self.root/"features", "train")
        self.assertEqual(len(loaded), 3)
        for i, row in enumerate([0, 2, 3]):
            esm, target = loaded[i]
            self.assertTrue((esm == row).all())
            np.testing.assert_array_equal(target.numpy(), data.y[i])


if __name__ == "__main__":
    unittest.main()
