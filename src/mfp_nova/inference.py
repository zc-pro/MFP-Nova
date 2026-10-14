import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CachedPeptides, PeptideData, collate_features
from .encoder import FrozenEncoder
from .metrics import evaluate_metrics
from .model import NovaSequenceModel
from .trainer import predict_loader


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = NovaSequenceModel(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), checkpoint


def evaluate(checkpoint_path, data_root, feature_root, split, device, output, predictions=None, batch_size=128):
    model, checkpoint = load_model(checkpoint_path, device)
    data = PeptideData(data_root, checkpoint["track"])
    if data.labels != checkpoint["labels"]:
        raise ValueError("Checkpoint label order differs from dataset")
    dataset = CachedPeptides(data, feature_root, split)
    for key in ("name", "resolved_revision", "max_length", "dataset", "storage_dtype", "extraction_dtype"):
        if dataset.meta[key] != checkpoint["encoder"][key]:
            raise ValueError(f"Evaluation feature definition differs: {key}")
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_features)
    y, probabilities = predict_loader(model, loader, device)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = {"split": split, "track": data.track, "rows": len(dataset),
              "checkpoint": str(checkpoint_path), "threshold": checkpoint["threshold"],
              "labels": data.labels, "metrics": evaluate_metrics(y, probabilities, checkpoint["threshold"])}
    # Exclusive creation protects existing experiment results.
    with destination.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    if predictions is not None:
        prediction_path = Path(predictions)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open("xb") as f:
            np.savez_compressed(f, row_id=data.rows[dataset.indices], ID=data.ids[dataset.indices],
                                canonical_backbone=data.sequences[dataset.indices], label_names=np.array(data.labels),
                                targets=y.astype(np.uint8), probabilities=probabilities,
                                threshold=checkpoint["threshold"], split=split, track=data.track,
                                checkpoint=str(checkpoint_path), dataset_version=dataset.meta["dataset"],
                                encoder_revision=dataset.meta["resolved_revision"])
    return result


def predict_sequences(checkpoint_path, sequences, device="cpu"):
    model, checkpoint = load_model(checkpoint_path, device)
    info = checkpoint["encoder"]
    encoder = FrozenEncoder(info["name"], info["resolved_revision"], info["max_length"], device)
    output = []
    batch_size = checkpoint["config"]["encoder"]["batch_size"]
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        states, mask = encoder.encode(batch)
        # Match the float16 cache round-trip used during head training.
        with torch.inference_mode():
            probabilities = model(states.half().float(), mask).sigmoid().cpu().tolist()
        for sequence, probs in zip(batch, probabilities):
            output.append({"sequence": sequence, "probabilities": dict(zip(checkpoint["labels"], probs)),
                           "predicted_labels": [label for label, p in zip(checkpoint["labels"], probs)
                                                if p >= checkpoint["threshold"]]})
    return output
