"""Function-conditioned residue likelihoods with amino-acid composition held fixed."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidueOrderEvidence(nn.Module):
    alphabet_size = 20

    def __init__(self, support, output_dim, width=32, context_dim=48, rank=16):
        super().__init__()
        self.embedding = nn.Embedding(1 + self.alphabet_size, width, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(width, width, 3, dilation=d) for d in (1, 2, 4))
        self.context = nn.Linear(width, context_dim)
        self.background = nn.Linear(context_dim, self.alphabet_size)
        self.context_factor = nn.Linear(context_dim, rank)
        self.label_factor = nn.Parameter(torch.randn(len(support), rank) * .1)
        self.residue_factor = nn.Linear(rank, self.alphabet_size, bias=False)
        self.write = nn.Linear(4, output_dim, bias=False)
        nn.init.zeros_(self.write.weight)
        self.register_buffer("positive_weight", torch.as_tensor(support).float().clamp_min(1).rsqrt(),
                             persistent=False)

    def prefix_context(self, aa, mask):
        valid = mask[:, None]
        x = self.embedding(aa.masked_fill(~mask, 0)).transpose(1, 2)
        for conv in self.convs:
            left_pad = (conv.kernel_size[0] - 1) * conv.dilation[0]
            x = (x + F.gelu(conv(F.pad(x, (left_pad, 0))))).masked_fill(~valid, 0.)
        # The shifted representation excludes the residue being predicted.
        return F.gelu(self.context(F.pad(x[..., :-1], (1, 0)).transpose(1, 2)))

    def observed_log_probabilities(self, aa, mask):
        context = self.prefix_context(aa, mask)
        tokens = aa.masked_fill(~mask, 0)
        one_hot = F.one_hot(tokens, self.alphabet_size + 1)[..., 1:]
        remaining = one_hot.sum(dim=1, keepdim=True) - one_hot.cumsum(1) + one_hot
        log_counts = remaining.clamp_min(1).float().log()
        log_counts = log_counts.masked_fill((remaining == 0) & mask[..., None], -torch.inf)

        base = self.background(context).float() + log_counts
        factor = self.context_factor(context)[:, :, None, :] * self.label_factor.to(context.dtype)
        shift = self.residue_factor(factor)
        conditional = base.detach()[:, :, None, :] + shift.float()
        observed = (tokens - 1).clamp_min(0)[..., None]
        background_score = base.log_softmax(-1).gather(-1, observed).squeeze(-1)
        label_score = conditional.log_softmax(-1).gather(
            -1, observed[:, :, None, :].expand(-1, -1, conditional.shape[2], -1)).squeeze(-1)
        return (background_score.masked_fill(~mask, 0.),
                label_score.masked_fill(~mask[..., None], 0.))

    def forward(self, aa, mask, targets=None):
        # Both directions use the same tower. Flip the scores back before pooling.
        directional_mask = torch.cat((mask, mask.flip(1)))
        background, conditional = self.observed_log_probabilities(
            torch.cat((aa, aa.flip(1))), directional_mask)
        forward, backward = (conditional - background.detach()[..., None]).chunk(2)
        evidence = .5 * (forward + backward.flip(1))
        length = mask.sum(1).clamp_min(1)
        positive, negative = evidence.relu(), (-evidence).relu()
        features = torch.stack((positive.sum(1) / length[:, None], negative.sum(1) / length[:, None],
                                positive.amax(1), negative.amax(1)), dim=-1)

        totals = {}
        if targets is not None:
            label_weights = targets.float() * self.positive_weight
            forward_nll, backward_nll = (-conditional.sum(1)).chunk(2)
            per_peptide = (forward_nll + backward_nll) / (2 * length[:, None])
            totals = {
                "order_background": (-background.sum(), directional_mask.sum()),
                "order_conditional": ((per_peptide * label_weights).sum(), label_weights.sum()),
            }
        return self.write(features), totals
