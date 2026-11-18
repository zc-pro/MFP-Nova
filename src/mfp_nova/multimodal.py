"""Read-only multimodal inputs and Train-only statistics for research models."""
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import PeptideData, CachedPeptides
from .therapep import TRACKS as THERAPEP_TRACKS


PROPERTY_GROUPS = [list(range(37 + 20*i, 57 + 20*i)) for i in range(20)] + [
    list(range(17, 37)), list(range(437, 477)), list(range(477, 502)),
    list(range(502, 515)), list(range(17))]


def fit_numeric(values):
    if len(values)==0:
        width = values.shape[-1]
        return {"impute": [0.]*width, "mean": [0.]*width, "scale": [1.]*width, "active": [False]*width}
    observed = np.isfinite(values)
    count = observed.sum(0)
    median = np.array([np.median(values[observed[:, j], j]) if count[j] else 0
                       for j in range(values.shape[-1])], dtype=np.float32)
    filled = np.where(observed, values, median)
    mean, scale = filled.mean(0), filled.std(0)
    scale[scale < 1e-6] = 1
    return {"impute": median.tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
            "active": (count > 0).tolist()}


def transform_numeric(values, stats):
    observed = np.isfinite(values)
    transformed = (np.where(observed, values, stats["impute"]) - stats["mean"]) / stats["scale"]
    transformed[..., ~np.asarray(stats["active"], bool)] = 0
    # Saved with Train statistics: old checkpoints retain their original inputs.
    if stats.get("zero_missing", False):
        transformed[~observed] = 0
    return transformed.astype(np.float32), observed


def label_statistics(y, clusters, labels, min_support=20, max_factors=32):
    support = y.sum(0).astype(int)
    independent = [len(set(clusters[y[:, j] > 0])) for j in range(y.shape[1])]
    co = y.astype(np.float32).T @ y.astype(np.float32)
    pair = co / np.sqrt(np.maximum(support[:, None]*support[None, :], 1))
    pair *= co / (co + 20)
    np.fill_diagonal(pair, 0)
    counts, components = Counter(), defaultdict(set)
    for target, cluster in zip(y, clusters):
        positive = np.flatnonzero(target)
        for order in (3, 4):
            for combination in itertools.combinations(positive, order):
                counts[combination] += 1
                components[combination].add(int(cluster))
    selected = [k for k, n in counts.most_common() if n >= min_support and len(components[k]) >= 10][:max_factors]
    incidence = np.zeros((len(selected), y.shape[1]), dtype=np.float32)
    for i, combination in enumerate(selected):
        incidence[i, list(combination)] = 1
    # Fixed across architectures; these are observed-annotation counts, not true prevalence.
    strata = ["head" if n >= 1000 else "medium" if n >= 200 else "tail" for n in support]
    weights = np.sqrt(max(float(support.max()), 1) / np.maximum(support, 1)).clip(1, 4)
    weights /= weights.mean()
    return {"support": support.tolist(), "positive_clusters": independent, "strata": strata,
            "positive_weights": weights.tolist(), "pair_prior": pair.tolist(),
            "hyperedges": incidence.tolist(), "hyperedge_clusters": [len(components[k]) for k in selected],
            "prototype_counts": [min(4, max(1, n // 100)) for n in independent],
            "observed_prior": ((support + 1)/(len(y)+2)).tolist(), "labels": labels}


GRAM_LABELS = ("anti_gram+", "anti_gram_")


def gram_positive_multipliers(data, mode):
    """Compute Gram positive weights from the full original Train cohort."""
    columns = [data.labels.index(label) for label in GRAM_LABELS]
    train = data.indices("train")
    targets = data.y[train][:, columns].astype(bool)
    multipliers = np.ones(data.y.shape, dtype=np.float32)
    summary = {}
    for i, (label, column) in enumerate(zip(GRAM_LABELS, columns)):
        positive_rows = train[targets[:, i]]
        dual = targets[targets[:, i], 1 - i]
        _, inverse, counts = np.unique(data.clusters[positive_rows], return_inverse=True, return_counts=True)
        n = len(positive_rows)
        if not n or (mode == "cluster_context" and (dual.all() or not dual.any())):
            raise ValueError(f"Insufficient Train annotation contexts for {label} and {mode}")
        # Equal cluster mass, normalized to the label's original positive mass.
        cluster_weight = (n / len(counts)) / counts[inverse]
        weights = .5 + .5 * cluster_weight
        if mode == "cluster_context":
            balanced = weights.copy()
            for group in (dual, ~dual):
                balanced[group] *= (n / 2) / weights[group].sum()
            weights = .5 * (weights + balanced)
        multipliers[positive_rows, column] = weights
        summary[label] = {"positives": n, "clusters": len(counts), "single_recorded": int((~dual).sum()),
                          "single_weight_fraction": float(weights[~dual].sum() / n),
                          "maximum_multiplier": float(weights.max())}
    return multipliers, summary


def function_projection(labels, track):
    """Independent-function groups. Antimicrobial is generic evidence, not another function."""
    if track in (*THERAPEP_TRACKS, "rich48"):
        # Published aliases and the global therapeutic flag do not define 15
        # independent functions; do not fabricate the major15 MFP projection.
        return None
    axes = ["antibacterial", "antifungal", "antiparasitic", "antiviral", "antimicrobial",
            "immunoactive", "anticancer", "antihypertensive", "thrombolytic", "neuropeptide",
            "antioxidant", "metabolic_regulatory", "growth_regeneration", "cell_communication", "drug_delivery"]
    children = {
        "anti_gram+": "antibacterial", "anti_gram_": "antibacterial",
        "antiplasmodial": "antiparasitic", "antitrypanosomic": "antiparasitic",
        "antihcv": "antiviral", "antihiv": "antiviral", "antihsv": "antiviral", "antisars": "antiviral",
        "antiinflammatory": "immunoactive", "analgesic": "neuropeptide", "opioid": "neuropeptide",
        "glucose_metabolism": "metabolic_regulatory", "lipid_metabolism": "metabolic_regulatory",
        "growth_regulatory": "growth_regeneration", "quorum_sensing": "cell_communication",
        "cell_penetrating": "drug_delivery", "blood_brain_barrier_penetrating": "drug_delivery", "tumor_homing": "drug_delivery",
        "antiangiogenesis": "anticancer"}
    projection = np.zeros((len(labels), len(axes)), np.float32)
    for i, name in enumerate(labels):
        base = name.removesuffix("__unknown")
        axis = base if track == "major15" or base in axes else children.get(base)
        if axis is None and "cancer" in base:
            axis = "anticancer"
        if axis is None:
            raise ValueError(f"Independent-function mapping missing for {name}")
        projection[i, axes.index(axis)] = 1
    return projection.tolist()


def independent_functions(y, projection):
    return suppress_generic_antimicrobial((np.asarray(y) @ np.asarray(projection)) > 0)


def suppress_generic_antimicrobial(result):
    # Antibacterial/fungal/parasitic/viral evidence subsumes generic antimicrobial evidence.
    result[:, 4] &= ~result[:, :4].any(1)
    return result


def localize_terminal_chemistry(tokens):
    """Move supported, consistent acetyl/amidation records to the two ends."""
    terminal = []
    remaining = list(tokens)
    for marker, prefix, aliases in (
        ("[ac]", "N-terminal modification:", {"acetylation", "acetyl", "n-acetylation", "n-acetyl"}),
        ("[am]", "C-terminal modification:", {"amidation", "amide", "c-amidation", "c-terminal amidation"}),
    ):
        explicit = [t for t in tokens if t.startswith(prefix)]
        recognized = [t for t in explicit if t[len(prefix):].strip().casefold() in aliases]
        terminal.append((marker in tokens or bool(recognized)) and len(recognized) == len(explicit))
        remaining = [t for t in remaining if t != marker and t not in recognized]
    return remaining, terminal


class MultimodalData:
    def __init__(self, root, track, statistics=None, min_support=20, max_factors=32, sequence_only=False,
                 sequence_chemistry=False, train_indices=None, localize_terminals=False):
        self.data = PeptideData(root, track)
        self.root = Path(root)
        self.sequence_only = sequence_only
        self.sequence_chemistry = sequence_chemistry
        self.localize_terminals = localize_terminals
        if track in THERAPEP_TRACKS:
            from .therapep import init_multimodal
            init_multimodal(self, statistics, min_support, max_factors)
            return
        if sequence_only:
            with np.load(self.root/"arrays/sequences.npz", allow_pickle=False) as z:
                self.seq = {"canonical_tokens": z["canonical_tokens"]}
            # Row-level availability supports fair Val grouping without loading
            # coordinates, node descriptors or per-sample geometry calculations.
            with np.load(self.root/"arrays/structure_primary.npz", allow_pickle=False) as z:
                self.struct = {"structure_available": z["structure_available"]}
            train = self.data.indices("train")
            if sequence_chemistry:
                self.read_chemistry()
            if statistics is None:
                labels = label_statistics(self.data.y[train], self.data.clusters[train], self.data.labels, min_support, max_factors)
                labels.update(geometry_positive_clusters=[0]*len(self.data.labels), geometry_prototype_counts=[0]*len(self.data.labels))
                statistics = {"phys": fit_numeric(np.empty((0, 515))), "nodes": fit_numeric(np.empty((0, 2))),
                              "structure_global": fit_numeric(np.empty((0, 25))), "chemical_vocab": {}, "labels": labels,
                              "function_projection": function_projection(self.data.labels, track),
                              "dataset": json.loads((self.root/"manifest.json").read_text(encoding="utf-8"))["version_date"],
                              "cohort": "sequence_only_original", "train_rows": len(train)}
                statistics["phys"]["zero_missing"] = True
                if sequence_chemistry:
                    statistics["chemical_vocab"] = self.chemical_vocabulary(train)
                if localize_terminals:
                    statistics["localized_terminal_chemistry"] = True
            if statistics["labels"]["labels"] != self.data.labels:
                raise ValueError("Checkpoint label order differs from data")
            self.statistics = statistics
            return
        def read(path):
            with np.load(self.root/path, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        self.seq = read("arrays/sequences.npz")
        self.phys = read("arrays/physchem_raw.npz")
        self.struct = read("arrays/structure_primary.npz")
        for source in (self.phys, self.struct):
            if not np.array_equal(source["row_id"], self.data.rows):
                raise ValueError("Multimodal row_id order differs from targets")
        self.read_chemistry()
        train = self.data.indices("train") if train_indices is None else train_indices
        if statistics is None:
            node_values = np.stack([np.log1p(self.struct["ASA_A2"]), self.struct["RSA_unclipped"]], -1)
            selected_nodes = node_values[train][self.struct["residue_mask"][train]]
            statistics = {"phys": fit_numeric(self.phys["X"][train]),
                          "structure_global": fit_numeric(self.struct["scalar_X"][train]),
                          "nodes": fit_numeric(selected_nodes),
                          "chemical_vocab": self.chemical_vocabulary(train),
                          "labels": label_statistics(self.data.y[train], self.data.clusters[train], self.data.labels, min_support, max_factors),
                          "function_projection": function_projection(self.data.labels, track),
                          "dataset": json.loads((self.root/"manifest.json").read_text(encoding="utf-8"))["version_date"],
                          "cohort": ("rich48_author_positive_multimodal" if track == "rich48" else
                                     "original_resolved_backbone"), "train_rows": len(train)}
            statistics["phys"]["zero_missing"] = True
            geometry_clusters = [len(set(self.data.clusters[train][(self.data.y[train, j]>0)&self.struct["structure_available"][train]]))
                                 for j in range(len(self.data.labels))]
            statistics["labels"]["geometry_positive_clusters"] = geometry_clusters
            statistics["labels"]["geometry_prototype_counts"] = [min(4, max(1, n//100)) if n else 0 for n in geometry_clusters]
        if statistics["labels"]["labels"] != self.data.labels:
            raise ValueError("Checkpoint label order differs from data")
        self.statistics = statistics

    def read_chemistry(self):
        symbols = {}
        with (self.root/"symbolic_sequences.jsonl").open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                symbols[int(item["row_id"])] = item["HELM_tokens"] if item["parsed"] else []
        with (self.root/"records.csv").open(encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
        if [int(r["row_id"]) for r in records] != self.data.rows.tolist():
            raise ValueError("Chemical records differ from array row_id order")
        self.chemical, self.residue_chemical = [], []
        if self.localize_terminals:
            self.terminal_chemical = []
        self.source = []
        for i, record in enumerate(records):
            tokens = symbols[int(self.data.rows[i])]
            core = [t for t in tokens if t not in ("[ac]", "[am]")]
            self.residue_chemical.append(core if len(core) == len(self.data.sequences[i]) else [])
            global_tokens = [t for t in tokens if t in ("[ac]", "[am]") or len(t) != 1]
            for key in ("N-terminal modification", "C-terminal modification", "Post translation modifications"):
                value = record[key].strip()
                if value.lower() not in ("", "nan", "nd", "none"):
                    global_tokens.append(f"{key}:{value}")
            parts = record["HELM notation"].split("$")
            if len(parts)>1 and parts[1]:
                global_tokens.append("connection:"+parts[1])
            if self.localize_terminals:
                global_tokens, terminal = localize_terminal_chemistry(global_tokens)
                self.terminal_chemical.append(terminal)
            self.chemical.append(global_tokens)
            self.source.append(record["source_evidence"])

    def chemical_vocabulary(self, train):
        vocabulary = sorted({token for i in train for token in self.chemical[i] + self.residue_chemical[i]})
        return {token: index + 2 for index, token in enumerate(vocabulary)}

    def chemical_inputs(self, i, length):
        vocab = self.statistics["chemical_vocab"]
        inputs = {"chemical": torch.tensor([vocab.get(t, 1) for t in self.chemical[i]] or [0]),
                  "residue_chem": torch.tensor([vocab.get(t, 1) for t in self.residue_chemical[i]] or [0]*length)}
        if self.localize_terminals:
            inputs["terminal_chemical"] = torch.tensor(self.terminal_chemical[i], dtype=torch.bool)
        return inputs

    def enable_function_auxiliary(self):
        """Pair actual Train labels only; out-of-scope rows have no auxiliary supervision."""
        track = "fine39" if self.data.track == "major15" else "major15"
        other = PeptideData(self.root, track)
        if not np.array_equal(other.rows, self.data.rows) or not np.array_equal(other.splits, self.data.splits):
            raise ValueError("Cross-task row_id/split alignment differs")
        valid = np.zeros(len(self.data.rows), bool)
        valid[np.intersect1d(self.data.indices("train"), other.indices("train"))] = True
        self.aux_mask = valid
        self.aux_y = np.zeros_like(other.y)
        self.aux_y[valid] = other.y[valid]
        if "function_auxiliary" not in self.statistics:
            stats = label_statistics(other.y[valid], other.clusters[valid], other.labels)
            stats.update(track=track, train_rows=int(valid.sum()))
            self.statistics["function_auxiliary"] = stats
        if self.statistics["function_auxiliary"]["labels"] != other.labels:
            raise ValueError("Auxiliary checkpoint label order differs from data")

    def sample(self, i, esm):
        if self.sequence_only:
            sample = sequence_inputs(self.data, i, esm)
            if self.sequence_chemistry:
                length = len(sample["aa"])
                sample.update(self.chemical_inputs(i, length))
                sample["d"] = torch.from_numpy((self.data.d_mask[i, :length] & (sample["aa"].numpy() != 6)).astype(np.int64))
            return sample
        if self.data.track in THERAPEP_TRACKS:
            from .therapep import sequence_sample
            return sequence_sample(self.data, i, esm)
        length = len(self.data.sequences[i])
        s = self.struct
        mask = s["residue_mask"][i, :length].copy()
        angles = np.stack([s[k][i, :length] for k in ("phi_degrees", "psi_degrees", "omega_degrees")], -1)
        angle_mask = np.isfinite(angles)
        radians = np.radians(np.nan_to_num(angles))
        angular = np.stack([np.sin(radians), np.cos(radians)], -1).reshape(length, 6)
        angular *= np.repeat(angle_mask, 2, -1)
        continuous = np.stack([np.log1p(s["ASA_A2"][i, :length]), s["RSA_unclipped"][i, :length]], -1)
        nodes, node_mask = transform_numeric(continuous, self.statistics["nodes"])
        confidence = s["confidence_verified"][i, :length]
        known = np.isfinite(confidence) & mask
        coordinates = np.where(mask[:, None], s["CA_centered_A"][i, :length], 0).astype(np.float32)
        triple = np.zeros(length, np.float32)
        if length >= 4 and mask.all():
            bonds = np.diff(coordinates, axis=0)
            unit = bonds / np.maximum(np.linalg.norm(bonds, axis=-1, keepdims=True), 1e-6)
            triple[1:-2] = (np.cross(unit[:-2], unit[1:-1])*unit[2:]).sum(-1)
        geometry = np.concatenate([angular, angle_mask, np.nan_to_num(confidence[:, None])/100,
                                   nodes, known[:, None], node_mask, triple[:, None]], -1).astype(np.float32)
        geometry[~mask] = 0
        edge_type = np.zeros((length, length), np.int64)
        start, end = s["edge_offsets"][i:i+2]
        u, v = s["edge_i"][start:end], s["edge_j"][start:end]
        edge_type[u, v] = edge_type[v, u] = s["edge_type"][start:end]
        phys, phys_mask = transform_numeric(self.phys["X"][i], self.statistics["phys"])
        struct_global, struct_global_mask = transform_numeric(s["scalar_X"][i], self.statistics["structure_global"])
        quality = np.array([[1, 1, 1, 1, 1, 1],
                            [mask.any(), known.any(), float(np.nanmean(confidence[known])/100) if known.any() else 0,
                             float(np.nanmin(confidence[known])/100) if known.any() else 0, mask.mean(), self.source[i] != "unknown"],
                            [phys_mask.any(), phys_mask.mean(), phys_mask[502:].mean(), phys_mask[437:477].mean(), 0, 1]], np.float32)
        return {"esm": torch.as_tensor(esm), "aa": torch.from_numpy(self.seq["canonical_tokens"][i, :length].astype(np.int64)),
                "d": torch.from_numpy((self.data.d_mask[i, :length] & (self.seq["canonical_tokens"][i, :length]!=6)).astype(np.int64)),
                **self.chemical_inputs(i, length),
                "mask": torch.ones(length, dtype=torch.bool), "coords": torch.from_numpy(coordinates),
                "geometry": torch.from_numpy(geometry), "struct_mask": torch.from_numpy(mask),
                "dssp": torch.from_numpy(np.where(s["dssp"][i, :length]>=0, s["dssp"][i, :length], 9).astype(np.int64)),
                "edge_type": torch.from_numpy(edge_type), "phys": torch.from_numpy(phys),
                "phys_mask": torch.from_numpy(phys_mask), "quality": torch.from_numpy(quality),
                "struct_global": torch.from_numpy(struct_global), "struct_global_mask": torch.from_numpy(struct_global_mask),
                "confidence_known": torch.from_numpy(known),
                "y": torch.from_numpy(self.data.y[i].astype(np.float32)), "row_id": torch.tensor(int(self.data.rows[i]))}

    def groups(self, indices, targets):
        if self.data.track in THERAPEP_TRACKS:
            from .therapep import groups
            return groups(self.data, indices)
        if self.sequence_only:
            lengths = np.asarray([len(self.data.sequences[i]) for i in indices])
            result = {"length_2_8": lengths <= 8, "length_9_16": (lengths > 8) & (lengths <= 16),
                      "length_17_32": (lengths > 16) & (lengths <= 32), "length_33_plus": lengths > 32,
                      "structure_present": self.struct["structure_available"][indices],
                      "structure_missing": ~self.struct["structure_available"][indices]}
            if self.statistics["function_projection"] is not None:
                functions = independent_functions(targets, self.statistics["function_projection"])
                result.update(single_function=functions.sum(1) == 1, multifunction=functions.sum(1) >= 2)
            return result
        lengths = np.array([len(self.data.sequences[i]) for i in indices])
        stereo = self.data.d_mask[indices].copy()
        stereo &= self.seq["canonical_tokens"][indices] != 6  # G is achiral.
        q = self.struct["confidence_verified"][indices]
        finite = np.isfinite(q)
        means = np.nansum(q, 1)/np.maximum(finite.sum(1), 1)
        projection = self.statistics["function_projection"]
        function_groups = {}
        if projection is not None:
            functions = independent_functions(targets, projection)
            function_groups = {"single_function": functions.sum(1)==1, "multifunction": functions.sum(1)>=2}
        return {"structure_present": self.struct["structure_available"][indices],
                "structure_missing": ~self.struct["structure_available"][indices],
                "D_annotation": stereo.any(1), "L_only": ~stereo.any(1),
                "length_2_8": lengths<=8, "length_9_16": (lengths>8)&(lengths<=16),
                "length_17_32": (lengths>16)&(lengths<=32), "length_33_plus": lengths>32,
                **function_groups,
                "confidence_unknown": ~finite.any(1), "confidence_known_lt70": finite.any(1)&(means<70),
                "confidence_known_ge70": finite.any(1)&(means>=70),
                **{f"source_{name}": np.array([self.source[i]==name for i in indices])
                   for name in sorted(set(self.source))}}


def backbone_inputs(sequence):
    return {"aa": torch.tensor(["ACDEFGHIKLMNPQRSTVWY".index(c.upper())+1 for c in sequence]),
            "mask": torch.ones(len(sequence), dtype=torch.bool)}


def sequence_inputs(data, i, esm):
    return {"esm": torch.as_tensor(esm), **backbone_inputs(data.sequences[i]),
            "y": torch.from_numpy(data.y[i].astype(np.float32)), "row_id": torch.tensor(int(data.rows[i]))}


class BackbonePeptides(Dataset):
    """Original residues and labels, independent of any language-model cache."""

    def __init__(self, source, split, max_length, limit=None):
        self.source = source
        indices = source.data.indices(split)
        self.indices = diagnostic_indices(source, indices, limit, split == "train") if limit else indices
        self.lengths = [len(source.data.sequences[i]) for i in self.indices]
        if self.lengths and max(self.lengths) > max_length:
            raise ValueError(f"{split} contains a backbone above the {max_length}-residue limit")
        self.meta = {"name": "aa34", "resolved_revision": "project_residue_encoding", "max_length": max_length,
                     "length_unit": "residues", "dataset": source.statistics["dataset"]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        data = self.source.data
        i, length = self.indices[position], self.lengths[position]
        tokens = self.source.seq["canonical_tokens"][i, :length]
        return {"aa": torch.from_numpy(tokens.astype(np.int64)), "mask": torch.ones(length, dtype=torch.bool),
                "y": torch.from_numpy(data.y[i].astype(np.float32)), "row_id": torch.tensor(int(data.rows[i]))}


class MultimodalPeptides(Dataset):
    def __init__(self, source, feature_root, split, limit=None, peptide_feature_root=None, sequence_only=False, indices=None):
        self.source = source
        self.positive_multipliers = None
        self.sequence_only = sequence_only
        self.cache = CachedPeptides(source.data, feature_root, split)
        pool = self.cache.indices if indices is None else indices
        self.indices = diagnostic_indices(source, pool, limit, split=="train") if limit else pool
        lookup = {int(i): j for j, i in enumerate(self.cache.indices)}
        self.cache_indices = [lookup[int(i)] for i in self.indices]
        self.meta = dict(self.cache.meta)
        self.peptide_cache = None
        if peptide_feature_root is not None:
            # Each cache independently maps row_id to the same source rows and verifies
            # IDs/backbones; its physical storage order may differ from the other cache.
            self.peptide_cache = CachedPeptides(source.data, peptide_feature_root, split)
            self.meta["peptide_encoder"] = self.peptide_cache.meta

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        esm, _ = self.cache[self.cache_indices[index]]
        i = int(self.indices[index])
        sample = (sequence_inputs(self.source.data, i, esm) if self.sequence_only and not self.source.sequence_chemistry
                  else self.source.sample(i, esm))
        if self.peptide_cache is not None:
            peptide, _ = self.peptide_cache[self.cache_indices[index]]
            if peptide.shape[0] != esm.shape[0]:
                raise ValueError("Primary and peptide cache residue lengths differ")
            sample["esm_peptide"] = peptide
        if self.cache.special_tokens is not None:
            position = self.cache.positions[self.cache_indices[index]]
            special = torch.from_numpy(np.array(self.cache.special_tokens[position], dtype=np.float32))
            sample["esm_cls"], sample["esm_eos"] = special.unbind(0)
        if hasattr(self.source, "aux_mask"):
            sample["aux_y"] = torch.from_numpy(self.source.aux_y[i].astype(np.float32))
            sample["aux_mask"] = torch.tensor(bool(self.source.aux_mask[i]))
        if self.positive_multipliers is not None:
            sample["positive_multiplier"] = self.positive_multipliers[index]
        return sample


def diagnostic_indices(source, pool, count, cover_labels=True, seed=42):
    """Small real-data probe; original splits and full Train statistics stay intact."""
    count = min(count, len(pool))
    selected = []
    lengths = np.array([len(source.data.sequences[i]) for i in pool])
    d = (source.data.d_mask[pool] & (source.seq["canonical_tokens"][pool]!=6)).any(1)
    geometry = source.struct["structure_available"][pool]
    categories = [geometry, d, ~geometry & ~d, lengths<=8, (lengths>8)&(lengths<=16),
                  (lengths>16)&(lengths<=32), lengths>32, lengths==lengths.max()]
    if cover_labels:
        y = source.data.y[pool]
        categories += [y[:, j]>0 for j in np.argsort(y.sum(0))]
    for category in categories:
        candidate = [int(i) for i in pool[category] if int(i) not in selected]
        if not selected and candidate:
            candidate.sort(key=lambda i: -len(source.data.sequences[i]))
        if candidate and len(selected)<count:
            selected.append(candidate[0])
    for i in np.random.default_rng(seed).permutation(pool):
        if len(selected)>=count:
            break
        if int(i) not in selected:
            selected.append(int(i))
    return np.asarray(selected, dtype=np.int64)


def collate_multimodal(samples):
    length = max(len(s["aa"]) for s in samples)
    result = {}
    for key in samples[0]:
        if key == "edge_type":
            result[key] = torch.stack([torch.nn.functional.pad(s[key], (0, length-len(s["aa"]), 0, length-len(s["aa"]))) for s in samples])
        elif key in ("esm", "esm_peptide", "aa", "d", "residue_chem", "mask", "coords", "geometry", "struct_mask", "dssp", "confidence_known", "chemical"):
            padding = 9 if key == "dssp" else 0
            result[key] = torch.nn.utils.rnn.pad_sequence([s[key] for s in samples], batch_first=True, padding_value=padding)
        else:
            result[key] = torch.stack([s[key] for s in samples])
    return result


def prepare_sequence_inputs(raw, statistics):
    b, length = raw["aa"].shape
    vocabulary = statistics["chemical_vocab"]

    def chemical_ids(values):
        return np.array([[0 if not str(t) else vocabulary.get(str(t), 1) for t in row] for row in values], np.int64)

    chemistry = raw.get("chemical_tokens", np.full((b, 1), ""))
    localized = statistics.get("localized_terminal_chemistry")
    if localized:
        localized_rows = [localize_terminal_chemistry([str(t) for t in row if str(t)]) for row in chemistry]
        width = max(1, max((len(tokens) for tokens, _ in localized_rows), default=0))
        chemistry = [tokens + [""] * (width - len(tokens)) for tokens, _ in localized_rows]
    arrays = {
        "esm": raw["esm"].astype(np.float32), "aa": raw["aa"].astype(np.int64),
        "d": (raw["d"] * (raw["aa"] != 6)).astype(np.int64), "mask": raw["mask"].astype(bool),
        "chemical": chemical_ids(chemistry),
        "residue_chem": chemical_ids(raw.get("residue_chem_tokens", np.full((b, length), ""))),
    }
    if localized:
        arrays["terminal_chemical"] = np.asarray([terminal for _, terminal in localized_rows], dtype=bool).reshape(b, 2)
    for key in ("esm_cls", "esm_eos", "esm_peptide"):
        if key in raw:
            arrays[key] = raw[key].astype(np.float32)
    return {key: torch.from_numpy(value) for key, value in arrays.items()}


def prepare_inputs(raw, statistics):
    """External raw tensors -> checkpoint-normalized inputs, without labels.

    Required: esm[B,L,D], aa[B,L] (1..20 in ACDEFGHIKLMNPQRSTVWY order),
    d[B,L], mask[B,L], phys_raw[B,515] (NaN for unavailable values).
    Optional geometry uses Angstrom/degrees and only enters with struct_mask=True.
    chemical_tokens/residue_chem_tokens are strings, encoded by the Train vocabulary.
    """
    b, length = raw["aa"].shape
    mask = raw["mask"].astype(bool)
    smask = raw.get("struct_mask", np.zeros((b, length), bool)).astype(bool)&mask
    coordinates = np.where(smask[..., None], raw.get("coords", np.zeros((b, length, 3))), 0).astype(np.float32)
    angles = raw.get("angles_degrees", np.full((b, length, 3), np.nan))
    angle_mask = np.isfinite(angles)&smask[..., None]
    radians = np.radians(np.nan_to_num(angles))
    angular = np.stack([np.sin(radians), np.cos(radians)], -1).reshape(b, length, 6)*np.repeat(angle_mask, 2, -1)
    asa = raw.get("ASA_A2", np.full((b, length), np.nan))
    rsa = raw.get("RSA_unclipped", np.full((b, length), np.nan))
    continuous = np.stack([np.log1p(asa), rsa], -1)
    nodes, observed = transform_numeric(continuous, statistics["nodes"])
    observed &= smask[..., None]
    confidence = raw.get("confidence_verified", np.full((b, length), np.nan))
    known = np.isfinite(confidence)&smask
    triple = np.zeros((b, length), np.float32)
    if length>=4:
        bonds = np.diff(coordinates, axis=1)
        unit = bonds/np.maximum(np.linalg.norm(bonds, axis=-1, keepdims=True), 1e-6)
        valid = smask[:, :-3]&smask[:, 1:-2]&smask[:, 2:-1]&smask[:, 3:]
        triple[:, 1:-2] = (np.cross(unit[:, :-2], unit[:, 1:-1])*unit[:, 2:]).sum(-1)*valid
    geometry = np.concatenate([angular, angle_mask, np.where(known, confidence, 0)[..., None]/100,
                               nodes, known[..., None], observed, triple[..., None]], -1).astype(np.float32)
    geometry[~smask] = 0
    distance = np.linalg.norm(coordinates[:, :, None]-coordinates[:, None], axis=-1)
    delta = abs(np.arange(length)[:, None]-np.arange(length)[None])
    pair_mask = smask[:, :, None]&smask[:, None]
    edge_type = np.zeros((b, length, length), np.int64)
    edge_type[(delta[None]>=3)&(distance<8)&pair_mask] = 2
    edge_type[(delta[None]==1)&pair_mask] = 1
    phys, phys_mask = transform_numeric(raw["phys_raw"], statistics["phys"])
    global_values, global_observed = transform_numeric(raw.get("struct_global_raw", np.full((b, 25), np.nan)), statistics["structure_global"])
    quality = np.ones((b, 3, 6), np.float32)
    quality[:, 1] = 0
    quality[:, 1, 0] = smask.any(1); quality[:, 1, 1] = known.any(1)
    quality[:, 1, 2] = np.where(known, confidence, 0).sum(1)/np.maximum(known.sum(1), 1)/100
    quality[:, 1, 3] = np.where(known, confidence, np.inf).min(1)/100
    quality[:, 1, 3][~known.any(1)] = 0
    quality[:, 1, 4] = smask.sum(1)/np.maximum(mask.sum(1), 1)
    quality[:, 1, 5] = raw.get("structure_source_known", np.zeros(b))
    quality[:, 2] = np.stack([phys_mask.any(1), phys_mask.mean(1), phys_mask[:, 502:].mean(1),
                              phys_mask[:, 437:477].mean(1), np.zeros(b), np.ones(b)], -1)
    arrays = {"struct_mask": smask, "coords": coordinates, "geometry": geometry,
              "dssp": np.where(smask & (raw.get("dssp", np.full((b, length), 9))>=0), raw.get("dssp", np.full((b, length), 9)), 9).astype(np.int64),
              "edge_type": edge_type, "confidence_known": known, "phys": phys, "phys_mask": phys_mask,
              "quality": quality,
              "struct_global": global_values, "struct_global_mask": global_observed}
    return {**prepare_sequence_inputs(raw, statistics), **{key: torch.from_numpy(value) for key, value in arrays.items()}}
