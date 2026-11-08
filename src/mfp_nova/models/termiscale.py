"""TermiScale: content-routed depthwise scales and conditional endpoint measures."""
import torch
from torch import nn
from torch.nn import functional as F

from .sequence_features import FrozenFeatures, prior_bias, residue_encoding, supervised_loss
from ..research_layers import masked_mean


def endpoint_positions(mask):
    n = mask.sum(1, keepdim=True).float()
    i = torch.arange(mask.shape[1], device=mask.device).float()[None].expand_as(mask)
    distance = n-1-i
    return torch.stack([i/(n-1).clamp_min(1), distance/(n-1).clamp_min(1),
                        (-i/4).exp(), (-distance/4).exp()], -1).masked_fill(~mask[..., None], 0)


def endpoint_log_measures(mask):
    n = mask.sum(1, keepdim=True).float()
    i = torch.arange(mask.shape[1], device=mask.device).float()[None].expand_as(mask)
    scores = torch.stack([torch.zeros_like(i), -i/4, -(n-1-i)/4], -1)
    return scores.masked_fill(~mask[..., None], -torch.inf).log_softmax(1)


class DynamicScaleBlock(nn.Module):
    def __init__(self, dim, dropout, routing):
        super().__init__()
        self.routing = routing
        self.norm, self.ff_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, k, padding=k//2, groups=dim, bias=False)
                                    for k in (3, 9, 17)])
        self.router = nn.Linear(dim+4, 3)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.mix = nn.Linear(dim, dim)
        self.ff1, self.ff2 = nn.Linear(dim, 2*dim), nn.Linear(2*dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, position, mask):
        u = self.norm(h).masked_fill(~mask[..., None], 0)
        content = torch.zeros_like(u) if self.routing == "position" else u
        alpha = self.router(torch.cat([content, position], -1)).softmax(-1)
        if self.routing == "uniform":
            alpha = torch.ones_like(alpha)/3
        values = torch.stack([F.gelu(conv(u.transpose(1, 2)).transpose(1, 2)) for conv in self.convs], -2)
        h = (h+self.dropout(self.mix((values*alpha[..., None]).sum(-2)))).masked_fill(~mask[..., None], 0)
        h = h+self.dropout(self.ff2(F.gelu(self.ff1(self.ff_norm(h)))))
        return h.masked_fill(~mask[..., None], 0), alpha


class TermiScale(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, dim=256,
                 dropout=.2, depth=4, peptide_input_dim=0, routing="content", pooling="conditional"):
        super().__init__()
        if routing not in ("content", "position", "uniform") or pooling not in ("conditional", "fixed", "uniform"):
            raise ValueError("Unknown TermiScale routing/pooling")
        self.statistics, self.pooling = statistics, pooling
        self.features = FrozenFeatures(input_dim, dim, peptide_input_dim)
        self.register_buffer("encoding", residue_encoding())
        self.aa, self.position = nn.Linear(34, dim), nn.Linear(4, dim, bias=False)
        self.input_norm, self.output_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([DynamicScaleBlock(dim, dropout, routing) for _ in range(depth)])
        self.label = nn.Linear(dim, num_labels, bias=False)
        nn.init.normal_(self.label.weight, std=.02)
        self.condition = nn.Linear(dim, 8, bias=False)
        self.condition_weights = nn.Parameter(torch.zeros(num_labels, 3, 8))
        self.measure_bias = nn.Parameter(torch.tensor([.8, .1, .1]).log().repeat(num_labels, 1))
        self.bias = nn.Parameter(prior_bias(statistics))

    def forward(self, batch, return_evidence=False):
        mask = batch["mask"]
        position = endpoint_positions(mask)
        h = self.input_norm(self.features(batch)+self.aa(self.encoding[batch["aa"]])+self.position(position))
        h = h.masked_fill(~mask[..., None], 0)
        weights = []
        for block in self.blocks:
            h, alpha = block(h, position, mask)
            if return_evidence:
                weights.append(alpha)
        with torch.autocast(device_type=h.device.type, enabled=False):
            scores = self.label(self.output_norm(h.float()))
            condition = torch.tanh(self.condition(F.layer_norm(masked_mean(h.float(), mask), (h.shape[-1],))))
            mixture = self.measure_bias[None].expand(h.shape[0], -1, -1)
            if self.pooling == "conditional":
                mixture = mixture+torch.einsum("caq,bq->bca", self.condition_weights, condition)
            rho = endpoint_log_measures(mask)
            # All -inf at padding would produce NaN mixture gradients in LSE.
            rho = rho.masked_fill(~mask[..., None], 0)
            if self.pooling == "uniform":
                log_mu = rho[:, :, :1].expand_as(scores)
            else:
                log_mu = torch.logsumexp(rho[:, :, None, :]+mixture.log_softmax(-1)[:, None], -1)
            logits = self.bias+torch.logsumexp((scores+log_mu).masked_fill(~mask[..., None], -torch.inf), 1)
        result = {"logits": logits}
        if return_evidence:
            result.update(log_measure=log_mu, scale_weights=torch.stack(weights, 1), mixture=mixture.softmax(-1))
        return result

    def compute_loss(self, batch, settings, return_output=False):
        return supervised_loss(self, batch, settings, return_output)
