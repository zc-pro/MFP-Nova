"""Two independent readers of a single TheraSet multimodal encoding."""

import torch
from torch import nn

from ..research_layers import Attention, FeedForward
from .sequence_features import prior_bias
from .theraset import TheraSet
from .trievidence import AA34Reader, empty_top10, fuse_evidence


class SharedReadout(nn.Module):
    def __init__(self, input_dim, pair_dim, statistics, mode):
        super().__init__()
        if mode not in ("states", "pairs"):
            raise ValueError(f"Unknown shared readout {mode}")
        width, heads = 128, 4
        count = len(statistics["labels"])
        self.project = nn.Linear(input_dim, width)
        self.interaction = Attention(width, heads, .1)
        self.feed_forward = FeedForward(width, 2, .1)
        self.pair_bias = nn.Linear(pair_dim, heads, bias=False)
        self.geometry_bias = nn.Linear(pair_dim, heads, bias=False)
        self.pair_bias.requires_grad_(mode == "pairs")
        self.geometry_bias.requires_grad_(mode == "pairs")
        # Start from the same attention in both arms.
        nn.init.zeros_(self.pair_bias.weight)
        nn.init.zeros_(self.geometry_bias.weight)
        self.mode = mode
        self.queries = nn.Parameter(torch.randn(count, width) * .02)
        self.read = Attention(width, heads, .1)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Parameter(torch.randn(count, width) * width**-.5)
        self.prior = nn.Parameter(prior_bias(statistics))

    def forward(self, memory):
        x, pair, geometry, quality, mask = memory[8:13]
        x = self.project(x) * mask[..., None]
        bias = None
        if self.mode == "pairs":
            bias = (self.pair_bias(pair) + self.geometry_bias(geometry) * quality[..., None]).permute(0, 3, 1, 2)
        x = self.feed_forward(x + self.interaction(x, x, mask, mask, bias), mask)
        queries = self.queries[None].expand(x.shape[0], -1, -1)
        valid = torch.ones(queries.shape[:2], device=x.device, dtype=torch.bool)
        features = self.norm(self.read(queries, x, valid, mask))
        return (features * self.output).sum(-1) + self.prior


class SharedTriEvidence(nn.Module):
    def __init__(self, theraset_args, cnn_channels=128, cnn_dropout=.2, reader_head="global", shared_mode="pairs"):
        super().__init__()
        if theraset_args.get("v5_mode") is not None or not theraset_args.get("evidence_decoder", True):
            raise ValueError("Use the original TheraSet evidence-decoder baseline")
        self.theraset = TheraSet(**theraset_args).requires_grad_(False).eval()
        self.aa34 = AA34Reader(theraset_args["statistics"], cnn_channels, cnn_dropout, reader_head)
        self.aa34.requires_grad_(False).eval()
        self.readout = SharedReadout(theraset_args.get("dim", 256), theraset_args.get("pair_dim", 32),
                                     theraset_args["statistics"], shared_mode)
        self.register_buffer("offsets", torch.zeros(3))

    def train(self, mode=True):
        super().train(mode)
        self.theraset.eval()
        self.aa34.eval()
        return self

    def head_logits(self, batch):
        with torch.no_grad():
            maps, _ = self.theraset.cnn(batch)
            memory = self.theraset.encode_evidence(batch, maps)
        return self.readout(memory)

    def forward(self, batch):
        reference = self.theraset(batch, return_memory=True)
        a, c, h = reference["logits"], self.aa34(batch), self.readout(reference["memory"])
        logits = fuse_evidence(a, c, h, self.offsets)
        return {"logits": logits, "theraset_logits": a, "aa34_logits": c, "latent_logits": h,
                "predicted": empty_top10(logits, a)}
