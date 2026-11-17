"""Fine39 prediction from ordered sequence windows and aligned context."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .sequence_features import ResidueCNN, prior_bias, supervised_loss


WINDOW_WIDTHS = (2, 3, 4)
EVIDENCE_NAMES = ("global", "width_2", "width_3", "width_4")
DIAGNOSTIC_NAMES = ("without_interaction", "shifted_context")


def prefix_sum(x):
    return F.pad(x.float().cumsum(dim=1), (0, 0, 1, 0))


def pool_windows(scores, visible):
    """Log-mean-exp of up to two valid windows per label."""
    count = visible.sum(1).clamp(max=2)
    top = scores.float().masked_fill(~visible[..., None], -torch.inf).topk(
        min(2, scores.shape[1]), dim=1
    ).values
    # In mixed-length batches an entire width may be absent on some rows.
    # Keep its backward computation finite before masking it at the outer pool.
    top = top.masked_fill((count == 0)[:, None, None], 0)
    return top.logsumexp(dim=1) - count.clamp_min(1).float().log()[:, None]


def shift_context(context, positions, count):
    shifted = (positions + count // 2) % count.clamp_min(1)
    return context.gather(1, shifted[..., None].expand_as(context))


class PeptideV6(nn.Module):
    def __init__(self, input_dim, peptide_input_dim, num_labels, statistics, dim=128,
                 cnn_channels=128, interaction_rank=32, dropout=.2, mode="interaction"):
        super().__init__()
        if mode not in ("interaction", "additive"):
            raise ValueError("V6 mode must be interaction or additive")
        self.statistics = statistics
        self.mode = mode
        self.cnn = ResidueCNN(dropout, cnn_channels)
        self.primary = nn.Sequential(nn.LayerNorm(input_dim, elementwise_affine=False),
                                     nn.Linear(input_dim, dim))
        self.peptide = nn.Sequential(nn.LayerNorm(peptide_input_dim, elementwise_affine=False),
                                     nn.Linear(peptide_input_dim, dim))
        self.local = nn.ModuleList([
            nn.Sequential(nn.Linear(34 * width, dim), nn.GELU(), nn.LayerNorm(dim))
            for width in WINDOW_WIDTHS
        ])
        self.cnn_context = nn.Linear(4 * cnn_channels, dim)
        self.context = nn.Sequential(nn.Linear(70 + 3 * dim, dim), nn.LayerNorm(dim))
        self.global_evidence = nn.Sequential(
            nn.Linear(4 * cnn_channels + 2 * dim, dim), nn.LayerNorm(dim)
        )
        self.local_linear = nn.Linear(dim, dim, bias=False)
        self.context_linear = nn.Linear(dim, dim, bias=False)
        self.local_factor = nn.Linear(dim, interaction_rank, bias=False)
        self.context_factor = nn.Linear(dim, interaction_rank, bias=False)
        self.interaction = nn.Linear(interaction_rank, dim, bias=False)
        nn.init.zeros_(self.interaction.weight)
        self.norm = nn.LayerNorm(dim)
        self.label_query = nn.Parameter(torch.randn(num_labels, dim))
        self.bias = nn.Parameter(prior_bias(statistics))
        self.scale = math.sqrt(dim)

    def score(self, features):
        return (features @ self.label_query.T).float() / self.scale

    def forward(self, batch, diagnostics=False):
        mask = batch["mask"]
        valid = mask[..., None]
        lengths = mask.sum(dim=1, keepdim=True)
        aa = batch["aa"].masked_fill(~mask, 0)
        raw = self.cnn.encoding[aa]
        cnn = self.cnn({**batch, "aa": aa})
        primary = self.primary(batch["esm"].masked_fill(~valid, 0))
        peptide = self.peptide(batch["esm_peptide"].masked_fill(~valid, 0))
        esm = torch.cat((primary, peptide), dim=-1).masked_fill(~valid, 0)
        cnn_max = cnn.masked_fill(~valid, -torch.inf).amax(dim=1)
        global_state = self.global_evidence(torch.cat((cnn_max, esm.float().sum(1) / lengths), -1))
        global_scores = self.score(global_state)

        raw_prefix = prefix_sum(raw)
        aligned = torch.cat((self.cnn_context(cnn), esm), dim=-1).masked_fill(~valid, 0)
        aligned_prefix = prefix_sum(aligned)
        full_raw = raw_prefix[:, -1:]
        scores = [global_scores]
        available = [torch.ones_like(lengths, dtype=torch.bool)]
        without, shifted = [global_scores], [global_scores]
        for width, encoder in zip(WINDOW_WIDTHS, self.local):
            window_count = (lengths - width + 1).clamp_min(0)
            available.append(window_count > 0)
            if mask.shape[1] < width:
                scores.append(torch.zeros_like(global_scores))
                if diagnostics:
                    without.append(scores[-1])
                    shifted.append(scores[-1])
                continue
            starts = torch.arange(mask.shape[1] - width + 1, device=mask.device)
            ends = starts + width
            visible = starts < window_count
            local = encoder(raw.unfold(1, width, 1).transpose(-1, -2).flatten(-2))
            right_count = (lengths - ends).clamp_min(0)
            left = raw_prefix[:, starts] / starts.clamp_min(1)[None, :, None]
            right = (full_raw - raw_prefix[:, ends]) / right_count.clamp_min(1)[..., None]
            positions = torch.stack((starts.expand_as(right_count), right_count), -1) / lengths[..., None]
            aligned_window = (aligned_prefix[:, ends] - aligned_prefix[:, starts]) / width
            context = self.context(torch.cat((left, right, positions, aligned_window), -1))

            linear = self.local_linear(local) + self.context_linear(context)
            p, q = self.local_factor(local), self.context_factor(context)
            factors = p * q if self.mode == "interaction" else (F.gelu(p) + F.gelu(q)) / math.sqrt(2)
            h = self.norm(linear + self.interaction(factors))
            scores.append(pool_windows(self.score(h), visible))
            if diagnostics:
                without.append(pool_windows(self.score(self.norm(linear)), visible))
                rotated = p * shift_context(q, starts, window_count)
                shifted_state = self.norm(linear + self.interaction(rotated))
                shifted.append(pool_windows(self.score(shifted_state), visible))

        present = torch.cat(available, dim=1)
        offset = self.bias.float() - present.sum(1, keepdim=True).float().log()
        evidence = torch.stack(scores, dim=1).masked_fill(~present[..., None], -torch.inf)
        output = {"logits": evidence.logsumexp(dim=1) + offset}
        if diagnostics:
            output["evidence_weights"] = evidence.softmax(dim=1)
            for name, variant in zip(DIAGNOSTIC_NAMES, (without, shifted)):
                pooled = torch.stack(variant, dim=1).masked_fill(~present[..., None], -torch.inf)
                output[name] = pooled.logsumexp(dim=1) + offset
        return output

    def compute_loss(self, batch, settings, return_output=False):
        return supervised_loss(self, batch, settings, return_output)
