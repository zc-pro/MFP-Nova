"""Build mean residue embeddings in METFAN's original split order."""
import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, EsmModel, T5EncoderModel

from data_utils import getSequenceData, PadEncode


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feature", choices=["esm2_mean", "prot_t5_mean"])
    parser.add_argument("checkpoint")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    t5 = args.feature == "prot_t5_mean"
    cls = T5EncoderModel if t5 else EsmModel
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, do_lower_case=False)
    model = cls.from_pretrained(args.checkpoint, torch_dtype=torch.float16).cuda().eval()
    folder = Path(os.environ.get("METFAN_FEATURES", "features")) / args.feature
    folder.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        seqs, labels = getSequenceData(".", split)
        # Recover exactly the residue identities, order and filtering used by TextCNN.
        encoded, _, lengths = PadEncode(seqs, labels, 50)
        residues = " ACDEFGHIKLMNPQRSTVWY"
        sequences = ["".join(residues[i] for i in row[:length])
                     for row, length in zip(encoded, lengths)]
        pooled = []
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start:start + args.batch_size]
            inputs = tokenizer([" ".join(seq) for seq in batch] if t5 else batch,
                               padding=True, return_special_tokens_mask=True,
                               return_tensors="pt").to("cuda")
            mask = inputs.pop("special_tokens_mask").eq(0) & inputs.attention_mask.bool()
            if mask.sum(1).tolist() != list(map(len, batch)):
                raise ValueError("Tokenization did not yield one token per residue")
            hidden = model(**inputs).last_hidden_state.float()
            pooled.append(((hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)).cpu())
        features = torch.cat(pooled)
        if features.shape[1] != (1024 if t5 else 1280):
            raise ValueError(f"Unexpected embedding width: {features.shape[1]}")
        torch.save(features, folder / f"{split}_{args.feature}.pt")
        print(args.checkpoint, split, tuple(features.shape), flush=True)


if __name__ == "__main__":
    main()
