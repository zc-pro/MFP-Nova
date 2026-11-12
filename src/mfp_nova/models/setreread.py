"""Condition residue evidence on the functions that a proposed edit keeps."""

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from .sequence_features import FrozenFeatures, prior_bias
from .termiscale import endpoint_positions
from .theraset import MotifCNN
from ..research_layers import masked_mean, observed_label_loss


def select_labels(scores, eligible, count):
    indices = scores.detach().masked_fill(~eligible, -torch.inf).argsort(
        dim=-1, descending=True, stable=True)[:, :count]
    return indices, eligible.gather(1, indices)


def sample_jaccard(predicted, truth):
    return (predicted & truth).sum(-1) / (predicted | truth).sum(-1).clamp_min(1)


def masked_row_loss(predicted, target, valid):
    errors = F.smooth_l1_loss(predicted.float(), target, reduction="none", beta=.1)
    return (errors * valid).sum(1) / valid.sum(1).clamp_min(1)


class Intervals(NamedTuple):
    vectors: torch.Tensor
    scores: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    valid: torch.Tensor


class SetReread(nn.Module):
    def __init__(self, input_dim, num_labels, vocab_size, statistics, function_projection,
                 dim=128, cnn_channels=64, dropout=.2, correction=True, conditional_evidence=True,
                 positive_recovery=True, base_add_candidates=2, gain_add_candidates=2, drop_candidates=2,
                 interval_evidence=False):
        super().__init__()
        self.statistics = statistics
        self.correction = correction
        self.conditional_evidence = conditional_evidence
        self.positive_recovery = positive_recovery
        self.base_add_candidates = base_add_candidates
        self.gain_add_candidates = gain_add_candidates
        self.drop_candidates = drop_candidates
        self.interval_evidence = interval_evidence
        self.cnn = MotifCNN(vocab_size, cnn_channels, dropout, chemical_evidence=True, short_only=interval_evidence)
        self.features = FrozenFeatures(input_dim, dim)
        self.local_projection = nn.Linear(self.cnn.output_channels, dim)
        self.position = nn.Linear(4, dim, bias=False)
        self.context_norm = nn.LayerNorm(dim)
        self.local_query = nn.Linear(dim, num_labels, bias=False)
        self.context_query = nn.Linear(dim, num_labels, bias=False)
        self.context_summary = nn.Linear(2 * dim, dim)
        self.local_labels = nn.Parameter(torch.empty(num_labels, dim))
        self.context_labels = nn.Parameter(torch.empty(num_labels, dim))
        nn.init.normal_(self.local_labels, std=.02)
        nn.init.normal_(self.context_labels, std=.02)
        self.bias = nn.Parameter(prior_bias(statistics))
        if interval_evidence:
            self.interval_projection = nn.Linear(128 + cnn_channels + 3, dim)
        read_dim = 2 * dim + 5
        if correction:
            if interval_evidence:
                self.interval_routing = nn.Linear(2, 1, bias=False)
                read_dim += dim
            self.add_head = nn.Sequential(nn.Linear(read_dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
            self.drop_head = nn.Sequential(nn.Linear(read_dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
            for head in (self.add_head, self.drop_head):
                nn.init.zeros_(head[-1].weight)
                nn.init.constant_(head[-1].bias, -.01)
        self.register_buffer("correction_active", torch.tensor(False))
        self.register_buffer("function_projection", torch.as_tensor(function_projection, dtype=torch.float32))

    def encode_intervals(self, residues, short, mask):
        length = mask.shape[1]
        lengths = mask.sum(1, keepdim=True)
        widths = torch.tensor([1, 3, 5, 9, 17], device=mask.device)
        left = torch.arange(length, device=mask.device).repeat_interleave(len(widths))
        right = left + widths.repeat(length)
        valid = right <= lengths
        # A shorter sequence can also be read as one complete interval.
        whole = (lengths < widths[-1]) & (lengths != widths).all(1, keepdim=True)
        valid = torch.cat((valid, whole), 1)
        left = F.pad(left.expand(len(mask), -1), (0, 1))
        right = torch.cat((right.clamp_max(length).expand(len(mask), -1), lengths), 1)
        size = right - left
        rows = torch.arange(len(mask), device=mask.device)[:, None]
        prefix = F.pad(short.float().cumsum(1), (0, 0, 1, 0))
        interior = prefix[rows, right - 1] - prefix[rows, left + 1]
        interior = (interior / (size - 2).clamp_min(1)[..., None]).masked_fill(size[..., None] < 3, 0)
        position = torch.stack((left / lengths, (lengths - right) / lengths, size / lengths), -1)
        vectors = self.interval_projection(torch.cat((
            residues[rows, left], residues[rows, right - 1], interior, position
        ), -1)).masked_fill(~valid[..., None], 0)
        scores = self.local_query(vectors).transpose(1, 2).float()
        return Intervals(vectors, scores, left, right, valid)

    @staticmethod
    def interval_mass(weights, intervals):
        prefix = F.pad(weights.cumsum(-1), (1, 0))
        left = intervals.left[:, None].expand(-1, weights.shape[1], -1)
        right = intervals.right[:, None].expand_as(left)
        return prefix.gather(-1, right) - prefix.gather(-1, left)

    def encode(self, batch):
        mask = batch["mask"]
        residues = self.cnn.residue_inputs(batch)
        short = torch.cat(self.cnn.residue_maps(batch, residues), -1)
        local = self.local_projection(short)
        local = (local + self.position(endpoint_positions(mask))).masked_fill(~mask[..., None], 0)
        context = self.context_norm(local + self.features(batch)).masked_fill(~mask[..., None], 0)
        local_weights = self.local_query(local).transpose(1, 2).float().masked_fill(
            ~mask[:, None], -torch.inf).softmax(-1)
        context_weights = self.context_query(context).transpose(1, 2).float().masked_fill(
            ~mask[:, None], -torch.inf).softmax(-1)
        local_read = torch.bmm(local_weights.to(local.dtype), local)
        intervals = None
        if self.interval_evidence:
            intervals = self.encode_intervals(residues, short, mask)
            interval_weights = intervals.scores.masked_fill(~intervals.valid[:, None], -torch.inf).softmax(-1)
            local_read = (local_read + torch.bmm(interval_weights.to(local.dtype), intervals.vectors)) * .5
        context_read = torch.bmm(context_weights.to(context.dtype), context)
        summary = self.context_summary(torch.cat([
            masked_mean(context, mask), context.masked_fill(~mask[..., None], -torch.inf).amax(1)
        ], dim=-1))
        local_logits = ((local_read * self.local_labels).sum(-1) + self.bias).float()
        context_logits = (((context_read + summary[:, None]) * self.context_labels).sum(-1) + self.bias).float()
        local_prob, context_prob = local_logits.sigmoid(), context_logits.sigmoid()
        mean_prob = (local_prob + context_prob) * .5
        base_logits = mean_prob.clamp(1e-6, 1 - 1e-6).logit()
        signals = torch.stack((mean_prob, local_prob - context_prob), -1)
        return local, local_weights, signals, base_logits, local_logits, context_logits, intervals

    def read(self, local, weights, signals, retained, candidates=None, intervals=None):
        coverage = weights.masked_fill(~retained[..., None], 0).amax(1, keepdim=True)
        if candidates is not None:
            rows = torch.arange(len(local), device=local.device)[:, None]
            weights, signals = weights[rows, candidates], signals[rows, candidates]
        shared = torch.minimum(weights, coverage) if self.conditional_evidence else torch.zeros_like(weights)
        remaining = weights - shared
        size = retained.float().sum(1, keepdim=True).log1p().expand_as(weights[..., 0])
        evidence = [
            torch.bmm(shared.to(local.dtype), local), torch.bmm(remaining.to(local.dtype), local),
            signals, shared.sum(-1, keepdim=True), remaining.sum(-1, keepdim=True), size[..., None]
        ]
        if intervals is not None:
            scores = intervals.scores if candidates is None else intervals.scores[rows, candidates]
            masses = torch.stack((self.interval_mass(shared, intervals), self.interval_mass(remaining, intervals)), -1)
            scores = scores + self.interval_routing(masses).squeeze(-1)
            attention = scores.masked_fill(~intervals.valid[:, None], -torch.inf).softmax(-1)
            evidence.append(torch.bmm(attention.to(local.dtype), intervals.vectors))
        return torch.cat(evidence, -1)

    def score_edits(self, local, weights, signals, base_logits, intervals=None):
        initial = base_logits.detach() >= 0
        add_gains = self.add_head(self.read(local, weights, signals, initial, intervals=intervals)).squeeze(-1)
        base_add, base_valid = select_labels(base_logits, ~initial, self.base_add_candidates)
        chosen = torch.zeros_like(initial).scatter(1, base_add, base_valid)
        gain_add, gain_valid = select_labels(add_gains, ~initial & ~chosen, self.gain_add_candidates)
        additions = torch.cat((base_add, gain_add), 1)
        add_valid = torch.cat((base_valid, gain_valid), 1)
        deletions, drop_valid = select_labels(-base_logits, initial, self.drop_candidates)

        num_labels = initial.shape[1]
        add_bits = F.one_hot(additions, num_labels).bool()
        drop_bits = F.one_hot(deletions, num_labels).bool()
        retained = initial[:, None] & ~drop_bits
        drop_gains, swap_gains = [], []
        for deleted, context in zip(deletions.split(1, dim=1), retained.unbind(1)):
            candidates = torch.cat((deleted, additions), 1)
            read = self.read(local, weights, signals, context, candidates, intervals)
            drop = self.drop_head(read[:, :1]).squeeze(-1)
            swap = drop + self.add_head(read[:, 1:]).squeeze(-1)
            drop_gains.append(drop)
            swap_gains.append(swap)

        sets = torch.cat((
            initial[:, None], initial[:, None] | add_bits, retained,
            (retained[:, :, None] | add_bits[:, None]).flatten(1, 2)
        ), 1)
        gains = torch.cat((
            torch.zeros_like(base_logits[:, :1]), add_gains.gather(1, additions),
            torch.cat(drop_gains, 1), torch.cat(swap_gains, 1)
        ), 1).float()
        valid = torch.cat((
            torch.ones_like(initial[:, :1]), add_valid, drop_valid,
            (drop_valid[:, :, None] & add_valid[:, None]).flatten(1, 2)
        ), 1)
        return initial, add_gains, sets, gains, valid

    @staticmethod
    def decode(base_logits, sets, gains, valid):
        # Candidate order gives ties to keep, then single edits, then swaps.
        scores = gains.masked_fill(~valid, -torch.inf)
        best = sets[torch.arange(len(sets), device=sets.device), scores.argmax(1)]
        include = scores[..., None].masked_fill(~sets, -torch.inf).amax(1)
        exclude = scores[..., None].masked_fill(sets, -torch.inf).amax(1)
        magnitude = (include - exclude).abs().clamp_min(1e-4)
        changed = best != (base_logits >= 0)
        return torch.where(changed, torch.where(best, magnitude, -magnitude), base_logits)

    def recovery_loss(self, local, weights, signals, truth, intervals=None):
        functions = truth.float() @ self.function_projection > 0
        # Match independent_functions: specific antimicrobial groups subsume the generic one.
        functions[:, 4] &= ~functions[:, :4].any(1)
        axes = self.function_projection.argmax(1)
        eligible = truth & functions[:, axes] & (functions.sum(1, keepdim=True) >= 2)
        hidden = torch.rand_like(signals[..., 0]).masked_fill(~eligible, -1).argmax(1, keepdim=True)
        retained = truth.scatter(1, hidden, False)
        estimate = self.add_head(self.read(local, weights, signals, retained, hidden, intervals)).squeeze(-1)
        gain = truth.sum(1, keepdim=True).clamp_min(1).float().reciprocal()
        return masked_row_loss(estimate, gain, eligible.any(1, keepdim=True)).sum() / eligible.any(1).sum().clamp_min(1)

    def forward(self, batch):
        local, weights, signals, base_logits, _, _, intervals = self.encode(batch)
        if not self.correction or not self.correction_active:
            return {"logits": base_logits}
        _, _, sets, gains, valid = self.score_edits(local, weights, signals, base_logits, intervals)
        return {"logits": self.decode(base_logits, sets, gains, valid)}

    def compute_loss(self, batch, settings, return_output=False):
        local, weights, signals, base_logits, local_logits, context_logits, intervals = self.encode(batch)
        target = batch["y"]
        weak_weight = settings.get("unlabeled_weight", .3)
        classification = observed_label_loss(base_logits, target, self.statistics, weak_weight)
        branches = sum(observed_label_loss(logits, target, self.statistics, weak_weight)
                       for logits in (local_logits, context_logits)) * .5
        terms = {"classification": classification, "branch": branches}
        total = classification + settings.get("branch", .2) * branches
        output = {"logits": base_logits}
        active = self.correction and settings.get("edit_gain", 0) > 0
        if self.training:
            self.correction_active.fill_(active)
        if active:
            initial, add_gains, sets, gains, valid = self.score_edits(local, weights, signals, base_logits, intervals)
            truth = target.bool()
            base_jaccard = sample_jaccard(initial, truth)
            intersection = (initial & truth).sum(1, keepdim=True)
            union = (initial | truth).sum(1, keepdim=True)
            add_target = (intersection + truth) / (union + ~truth).clamp_min(1) - base_jaccard[:, None]
            # All unselected labels get add supervision. Avoid counting shortlist adds twice.
            start = 1 + min(self.base_add_candidates, truth.shape[1]) + min(self.gain_add_candidates, truth.shape[1])
            edit_target = sample_jaccard(sets[:, start:], truth[:, None]) - base_jaccard[:, None]
            gain_loss = (masked_row_loss(add_gains, add_target, ~initial)
                         + masked_row_loss(gains[:, start:], edit_target, valid[:, start:])).mean()
            terms["edit_gain"] = gain_loss
            total = total + settings["edit_gain"] * gain_loss
            if self.positive_recovery and settings.get("recovery", 0):
                recovery = self.recovery_loss(local, weights, signals, truth, intervals)
                terms["recovery"] = recovery
                total = total + settings["recovery"] * recovery
            with torch.no_grad():
                output["logits"] = self.decode(base_logits, sets, gains, valid)
        return (total, terms, output) if return_output else (total, terms)
