"""SA-MTP-single: single-sequence PSIPRED, frozen FP32 ESM2 and native SA-MTP training."""
import argparse
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

PROJECT = Path(__file__).resolve().parents[2]
KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
from mfp_nova.metrics import evaluate_metrics
from mfp_nova.metric_protocol import format_primary
from mfp_nova.paper_benchmark import CLUSTERED, traditional5_report, traditional5_selection


def fasta(path):
    # structure_inputs.py writes exactly two lines per record, retaining duplicate rows.
    lines = path.read_text(encoding="utf-8").splitlines()
    return [(name[1:], sequence) for name, sequence in zip(lines[::2], lines[1::2])]


def load_info(output):
    return json.loads((output / "task.json").read_text(encoding="utf-8"))


def prepare(inputs, output):
    source = json.loads((inputs / "task.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    info = {"model": "SA-MTP-single", "task": source["task"], "input": str(inputs),
            "data_root": source["data_root"], "labels": source["labels"],
            "structure_method": "PSIPRED single sequence: seq2mtx -> psipred -> psipass2; no search database",
            "short_sequence_policy": "For SS2 prediction only, symmetrically X-pad sequences shorter than 5, then crop back to real residues",
            "method_note": "Single-sequence SS2 variant, not the original profile-based SA-MTP",
            "splits": source["splits"], "unique_sequences": source["unique_sequences"],
            "max_len": 50, "esm_dim": 1280, "feature_dtype": "float32"}
    (output / "task.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2), flush=True)


def queries(info, limit=0):
    inputs = Path(info["input"])
    all_queries = fasta(inputs / "queries.fasta")
    if not limit:
        return all_queries
    needed = set()
    for split in ("train", "val", "test"):
        needed.update(name for name, _ in fasta(inputs / split / "seqs.fasta")[:limit])
    return [(name, sequence) for name, sequence in all_queries if name in needed]


def read_ss2(path, sequence):
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    if len(rows) != len(sequence) or "".join(row[1] for row in rows) != sequence:
        raise ValueError(f"SS2 residues differ from the input sequence: {path}")
    probabilities = np.array([[float(v) for v in row[3:6]] for row in rows])
    if (probabilities.shape != (len(sequence), 3) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or (probabilities.sum(1) <= 0).any()):
        raise ValueError(f"Invalid C/H/E probabilities: {path}")
    return probabilities


def ss2(output, prefix, limit):
    info = load_info(output)
    folder = output / "ss2"
    folder.mkdir(exist_ok=True)
    weights = prefix / "share/psipred/data"
    selected = queries(info, limit)
    generated = 0
    # One temporary workspace is reused; only final SS2 files are retained.
    with tempfile.TemporaryDirectory(prefix="psipred_", dir=output) as temporary:
        work = Path(temporary)
        for i, (name, sequence) in enumerate(selected, 1):
            target = folder / f"{name}.ss2"
            if not target.exists():
                padding = max(0, 5 - len(sequence))
                left = padding // 2
                prediction_sequence = "X" * left + sequence + "X" * (padding - left)
                query = work / "query.fasta"
                query.write_text(f">{name}\n{prediction_sequence}\n", encoding="utf-8")
                with (work / "query.mtx").open("w") as stream:
                    subprocess.run([str(prefix / "bin/seq2mtx"), str(query)], stdout=stream, check=True)
                with (work / "query.ss").open("w") as stream:
                    subprocess.run([str(prefix / "bin/psipred"), str(work / "query.mtx"),
                                    *(str(weights / f) for f in ("weights.dat", "weights.dat2", "weights.dat3"))],
                                   stdout=stream, check=True)
                with (work / "query.horiz").open("w") as stream:
                    subprocess.run([str(prefix / "bin/psipass2"), str(weights / "weights_p2.dat"),
                                    "1", "1.0", "1.0", str(work / "query.ss2"), str(work / "query.ss")],
                                   stdout=stream, check=True)
                read_ss2(work / "query.ss2", prediction_sequence)
                if padding:
                    lines = [line.split() for line in (work / "query.ss2").read_text().splitlines()
                             if line.strip() and not line.lstrip().startswith("#")]
                    cropped = lines[left:left + len(sequence)]
                    for position, row in enumerate(cropped, 1):
                        row[0] = str(position)
                    (work / "query.ss2").write_text("# PSIPRED single; X-pad to 5 then crop\n" +
                                                   "\n".join(" ".join(row) for row in cropped) + "\n")
                read_ss2(work / "query.ss2", sequence)
                (work / "query.ss2").replace(target)
                generated += 1
            else:
                read_ss2(target, sequence)
            if i % 1000 == 0 or i == len(selected):
                print(f"SS2 {i}/{len(selected)}; newly generated={generated}", flush=True)
    print("Single-sequence SS2 complete; no protein search database was used.", flush=True)


def feature_layout(info):
    records = queries(info)
    offsets = np.r_[0, np.cumsum([len(sequence) for _, sequence in records])]
    index = {name: i for i, (name, _) in enumerate(records)}
    return records, offsets, index


def features(output, checkpoint, batch_size, limit):
    from utils.esm2_encoder import ESM2Encoder
    info = load_info(output)
    folder = output / "features"
    folder.mkdir(exist_ok=True)
    metadata = {"checkpoint": str(checkpoint), "dtype": "float32", "esm_dim": 1280,
                "extraction": "author encoder; full sequence, FP32, excludes CLS/EOS"}
    meta_path = folder / "encoder.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("Existing features were generated with different encoder settings")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    records, offsets, index = feature_layout(info)
    cache_path = folder / "esm2.npy"
    if cache_path.exists():
        cache = np.load(cache_path, mmap_mode="r+")
    else:
        cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32,
                                         shape=(int(offsets[-1]), 1280))
    done_path = folder / "esm2_completed.txt"
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()
    selected = [(name, sequence) for name, sequence in queries(info, limit) if name not in done]
    if not selected:
        print("Requested ESM2 features are already cached.", flush=True)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = ESM2Encoder(str(checkpoint), device=device)
    if encoder.model.config.hidden_size != 1280:
        raise ValueError("SA-MTP requires the ESM2 1280-dimensional checkpoint")
    with done_path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            embeddings = encoder.batch_encode_sequences([sequence for _, sequence in batch], batch_size=batch_size)
            for (name, sequence), embedding in zip(batch, embeddings):
                if embedding.shape != (len(sequence), 1280) or not np.isfinite(embedding).all():
                    raise ValueError(f"Invalid residue embeddings: {name}")
                i = index[name]
                cache[offsets[i]:offsets[i+1]] = embedding
            cache.flush()
            stream.writelines(name + "\n" for name, _ in batch)
            stream.flush()
            if (start + len(batch)) % 1000 == 0 or start + len(batch) == len(selected):
                print(f"ESM2 newly encoded {start+len(batch)}/{len(selected)}", flush=True)
    print(f"ESM2 cache: {cache_path}; logical size={cache.nbytes/2**30:.3f} GiB", flush=True)


def positions(length):
    if length < 50:
        return np.arange(length)
    return np.r_[np.arange(25), np.arange(length - 25, length)]


def padded_graph(output, name, sequence, embedding):
    from utils.graph_builder import build_dynamic_adjacency_matrix
    ss2_path = output / "ss2" / f"{name}.ss2"
    read_ss2(ss2_path, sequence)
    graph = build_dynamic_adjacency_matrix(ss2_path, embedding, beta=4.0,
                                           lambda_weight=0.25, top_k=None, training=False)
    # Check the full graph before truncation: NaNs can change its top-k topology.
    if not np.isfinite(graph).all():
        raise ValueError(f"Non-finite graph for {name}; apply sa_mtp_js.patch and repair the cache")
    keep = positions(len(sequence))
    padded = np.zeros((50, 50), dtype=np.float32)
    padded[:len(keep), :len(keep)] = graph[np.ix_(keep, keep)]
    return padded


def graphs(output, limit):
    info = load_info(output)
    records, offsets, index = feature_layout(info)
    folder = output / "features"
    embeddings = np.load(folder / "esm2.npy", mmap_mode="r")
    encoded = set((folder / "esm2_completed.txt").read_text().splitlines())
    path = folder / "graphs.npy"
    if path.exists():
        cache = np.load(path, mmap_mode="r+")
    else:
        cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                         shape=(len(records), 50, 50))
    done_path = folder / "graphs_completed.txt"
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()
    selected = [(name, sequence) for name, sequence in queries(info, limit) if name not in done]
    with done_path.open("a", encoding="utf-8") as stream:
        for n, (name, sequence) in enumerate(selected, 1):
            if name not in encoded:
                raise ValueError(f"ESM2 features have not been generated for {name}")
            i = index[name]
            embedding = np.asarray(embeddings[offsets[i]:offsets[i+1]])
            cache[i] = padded_graph(output, name, sequence, embedding)
            cache.flush()
            stream.write(name + "\n")
            stream.flush()
            if n % 1000 == 0 or n == len(selected):
                print(f"Graphs newly built {n}/{len(selected)}", flush=True)


def repair_graphs(output):
    """One-time repair of completed legacy graphs; keep SS2, ESM2 and finite short graphs."""
    info = load_info(output)
    records, offsets, index = feature_layout(info)
    folder = output / "features"
    embeddings = np.load(folder / "esm2.npy", mmap_mode="r")
    cache = np.load(folder / "graphs.npy", mmap_mode="r+")
    ready = set((folder / "graphs_completed.txt").read_text().splitlines())
    repaired = []
    for name, sequence in records:
        if name not in ready:
            continue
        i = index[name]
        # For a truncated graph, invalid edges outside the retained residues might
        # have displaced valid top-k edges. Rebuild these few long sequences too.
        if not np.isfinite(cache[i]).all() or len(sequence) > 50:
            embedding = np.asarray(embeddings[offsets[i]:offsets[i+1]])
            cache[i] = padded_graph(output, name, sequence, embedding)
            repaired.append(name)
            print(f"Repaired graph {name}; length={len(sequence)}", flush=True)
    cache.flush()
    print(f"Graph repair complete: checked={len(ready)}, rebuilt={len(repaired)}; "
          "SS2 and ESM2 retained. Run graphs next to finish any missing entries.", flush=True)
    return repaired


class Peptides(Dataset):
    def __init__(self, output, split, limit=0, feature_source=None):
        from utils.graph_builder import apply_dropedge
        info = load_info(output)
        inputs = Path(info["input"])
        self.records = fasta(inputs / split / "seqs.fasta")
        with (inputs / split / "labels.csv").open(newline="", encoding="utf-8") as stream:
            reader = csv.reader(stream)
            if next(reader) != info["labels"]:
                raise ValueError(f"Label order differs in {split}")
            self.y = np.array(list(reader), dtype=np.float32)
        with (inputs / split / "rows.csv").open(newline="", encoding="utf-8") as stream:
            self.rows = list(csv.DictReader(stream))
        if limit:
            self.records, self.y, self.rows = self.records[:limit], self.y[:limit], self.rows[:limit]
        _, self.offsets, self.index = feature_layout(info)
        folder = (feature_source or output) / "features"
        self.embeddings = np.load(folder / "esm2.npy", mmap_mode="r")
        self.graphs = np.load(folder / "graphs.npy", mmap_mode="r")
        ready = set((folder / "graphs_completed.txt").read_text().splitlines())
        for name, _ in self.records:
            if name not in ready:
                raise ValueError(f"Features/SS2/graph have not been completed for {name}")
            if not np.isfinite(self.graphs[self.index[name]]).all():
                raise ValueError(f"Non-finite cached graph for {name}; run repair-graphs before training")
        self.train_graphs = None
        if split == "train":
            # The released loader samples DropEdge once when loading Train, not every epoch.
            self.train_graphs = np.stack([apply_dropedge(self.graphs[self.index[name]], 0.1, training=True)
                                          for name, _ in self.records]).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        name, sequence = self.records[i]
        j = self.index[name]
        keep = positions(len(sequence))
        features = np.zeros((50, 1280), dtype=np.float32)
        features[:len(keep)] = self.embeddings[self.offsets[j]:self.offsets[j+1]][keep]
        mask = np.arange(50) >= len(keep)
        graph = self.train_graphs[i] if self.train_graphs is not None else self.graphs[j]
        return (torch.from_numpy(features), torch.from_numpy(self.y[i]), torch.from_numpy(mask),
                torch.arange(self.y.shape[1]), torch.tensor(np.asarray(graph), dtype=torch.float32))


def loaders(output, batch_size, limit, device, feature_source=None):
    def collate(batch):
        return tuple(value.to(device) for value in default_collate(batch))
    datasets = {split: Peptides(output, split, limit, feature_source) for split in ("train", "val", "test")}
    result = {split: DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                                num_workers=0, pin_memory=False, collate_fn=collate)
              for split, dataset in datasets.items()}
    return datasets, result


@torch.no_grad()
def predict(model, loader):
    model.eval()
    return np.concatenate([model(x, mask, tokens, adj_matrix=graph)[0].cpu().numpy()
                           for x, _, mask, tokens, graph in loader])


def train(output, repo, seed, epochs, warmup, batch_size, limit, run_name=None, feature_source=None):
    from models.tppred import TPMLC_single
    from run_network import Model
    from utils.threshold_optimizer import find_optimal_threshold
    subprocess.run(["git", "apply", "--reverse", "--check", "--ignore-space-change",
                    str(KIT / "sa_mtp_therapep.patch")], cwd=repo, check=True)
    subprocess.run(["git", "apply", "--reverse", "--check", "--ignore-space-change",
                    str(KIT / "sa_mtp_js.patch")], cwd=repo, check=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    info = load_info(output)
    clustered = info["task"] in CLUSTERED
    if feature_source is not None and queries(info) != queries(load_info(feature_source)):
        raise ValueError("Shared SA-MTP features require identical query IDs, sequences and order")
    result = output / "training" / (run_name or f"seed{seed}_epochs{epochs}_limit{limit}")
    result.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, batches = loaders(output, batch_size, limit, device, feature_source)
    model = TPMLC_single(1280, len(info["labels"]), 50, 256, device, nhead=4,
                         n_enc_layers=2, n_dec_layers=2, dropout=0.1, use_esm2=True,
                         esm_dim=1280, use_gat=True, use_film=True, film_dropout=0.2,
                         film_stable_mode=True, film_alpha=0.2,
                         use_learnable_lambda=True, lambda_init=0.25).to(device)
    context = SimpleNamespace(use_gat=True)
    criterion = torch.nn.BCELoss()
    settings = {"model": "SA-MTP-single", "task": info["task"], "seed": seed,
                "epochs": epochs, "warmup_epochs": warmup, "batch_size": batch_size,
                "performance_scope": "smoke test only" if limit else "full cohort",
                "limit_per_split": limit, "learning_rate": 1e-4, "weight_decay": 0,
                "film_regularization": 1e-4, "patience": 15,
                "structure_method": info["structure_method"], "decoder_padding_mask": True,
                "short_sequence_policy": info["short_sequence_policy"],
                "graph_note": "Released graph construction with fixed fusion 0.25; registered lambda "
                              "is unused by the released GAT forward. Train DropEdge sampled once.",
                "graph_js": "float64 base-2 JS divergence via rel_entr; clamp negative roundoff before exp",
                "threshold_protocol": "Val-only per-class F1 on 0.05..0.95 grid; single-class Val columns "
                                      "fixed at 0.5; fixed-0.5 results also reported"}
    if clustered:
        settings["threshold_protocol"] = "Validation only, global Jaccard"
    if feature_source is not None:
        settings["feature_source"] = str(feature_source)
    print(json.dumps(settings, indent=2), flush=True)
    history = []
    if warmup:
        Model.freeze_backbone(context, model)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0)
        for epoch in range(warmup):
            loss, val_loss, _, _ = Model.train_epoch(context, model, optimizer, criterion,
                                                    batches["train"], batches["val"], lambda_film=1e-4)
            print(f"Warmup {epoch+1}/{warmup}: train={loss:.6f} val={val_loss:.6f}", flush=True)
        Model.unfreeze_all(context, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_key, best_threshold = None, None
    for epoch in range(1, epochs+1):
        loss, val_loss, val_y, val_scores = Model.train_epoch(context, model, optimizer, criterion,
                                                              batches["train"], batches["val"], lambda_film=1e-4)
        if not np.isfinite(loss + val_loss):
            raise ValueError(f"Non-finite loss at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": loss, "val_loss": val_loss})
        print(f"Epoch {epoch}/{epochs}: train={loss:.6f} val={val_loss:.6f}", flush=True)
        if clustered:
            key, threshold, metrics = traditional5_selection(np.asarray(val_y), np.asarray(val_scores))
            history[-1].update(threshold=float(threshold), traditional5=metrics)
            print(f"Val threshold={threshold:.2f} {format_primary('traditional5', metrics)}", flush=True)
            improved = best_key is None or key > best_key
            if improved:
                best_key, best_threshold = key, threshold
        else:
            improved = val_loss <= best_loss
        if improved:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss, "settings": settings,
                        "threshold": None if best_threshold is None else float(best_threshold)}, result / "best_model.pth")
        else:
            stale += 1
            if stale >= 15:
                print(f"Early stopping at {epoch}; best={best_epoch}", flush=True)
                break
    model.load_state_dict(torch.load(result / "best_model.pth", map_location=device, weights_only=True)["model_state_dict"])
    val_p = predict(model, batches["val"])
    thresholds = np.full(len(info["labels"]), best_threshold if clustered else 0.5)
    if not clustered:
        for c in range(len(thresholds)):
            y = datasets["val"].y[:, c]
            if np.unique(y).size == 2:
                thresholds[c] = find_optimal_threshold(val_p[:, c], y, metric="f1")[0]
    # Test is predicted only after checkpoint and all thresholds have been fixed.
    test_p = predict(model, batches["test"])
    for split, probabilities in (("val", val_p), ("test", test_p)):
        dataset = datasets[split]
        np.savez_compressed(result / f"{split}_predictions.npz", targets=dataset.y,
                            probabilities=probabilities, predicted=probabilities >= (best_threshold if clustered else thresholds),
                            thresholds=thresholds, label_names=np.array(info["labels"]),
                            row_ids=np.array([int(row["row_id"]) for row in dataset.rows]),
                            ids=np.array([row["ID"] for row in dataset.rows]))
    y = datasets["test"].y
    summary = {**settings, "best_epoch": best_epoch, "best_val_loss": best_loss,
               "selection": "minimum Validation BCE; restored saved state before threshold fitting",
               "test_rows": len(y),
               "fixed_0.5": evaluate_metrics(y, test_p, threshold=0.5),
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "cuda": torch.version.cuda, "numpy": np.__version__,
                               "transformers": version("transformers"),
                               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
               "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
               "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()}
    if clustered:
        summary.update(traditional5_report(y, test_p, best_threshold, best_epoch, info["task"], info["labels"],
                                           datasets["train"].y.sum(0)),
                       selection="validation_traditional5")
    else:
        positive = y[:, :48].any(1)
        summary.update(full_test=evaluate_metrics(y, test_p, threshold=thresholds),
                       positive_test_rows=int(positive.sum()), positive_function48=evaluate_metrics(
                           y[positive, :48], test_p[positive, :48], threshold=thresholds[:48]))
    (result / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (result / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (result / "thresholds.json").write_text(json.dumps(thresholds.tolist()), encoding="utf-8")
    print(json.dumps({"model": summary["model"], "best_epoch": best_epoch,
                      "test_rows": len(y), "micro_f1": summary["full_test"]["micro_f1"],
                      "macro_f1": summary["full_test"]["macro_f1"],
                      "fixed_0.5_micro_f1": summary["fixed_0.5"]["micro_f1"]}, indent=2), flush=True)
    print(f"Completed. Results: {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "ss2", "features", "graphs", "repair-graphs", "train"))
    parser.add_argument("--input", type=Path, help="Existing structure_inputs.py output directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.home() / "paper-repro/SA-MTP")
    parser.add_argument("--psipred-prefix", type=Path, default=Path.home() / ".conda/envs/repro-features")
    parser.add_argument("--esm2", type=Path, default=Path.home() / "paper-repro/pretrained/esm2_t33_650M_UR50D")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--feature-source", type=Path, help="Existing SA-MTP feature output with identical queries")
    parser.add_argument("--limit-per-split", type=int, default=0, help="0=full cohort; positive values are smoke tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-name", help="Training experiment name; existing output is never overwritten")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.feature_batch_size) < 1 or min(args.warmup_epochs, args.limit_per_split) < 0:
        parser.error("epochs/batch sizes must be positive; warmup/limit must be nonnegative")
    repo, output = args.repo.resolve(), args.output.resolve()
    sys.path.insert(0, str(repo))
    if args.action == "prepare":
        if args.input is None:
            parser.error("prepare requires --input")
        prepare(args.input.resolve(), output)
    elif args.action == "ss2":
        ss2(output, args.psipred_prefix.resolve(), args.limit_per_split)
    elif args.action == "features":
        features(output, args.esm2.resolve(), args.feature_batch_size, args.limit_per_split)
    elif args.action == "graphs":
        graphs(output, args.limit_per_split)
    elif args.action == "repair-graphs":
        repair_graphs(output)
    else:
        train(output, repo, args.seed, args.epochs, args.warmup_epochs, args.batch_size,
              args.limit_per_split, args.run_name, args.feature_source.resolve() if args.feature_source else None)


if __name__ == "__main__":
    main()
