"""Fine39 prediction with a shared sequence encoder and two matched output heads."""

import torch
from torch import nn
from torch.nn import functional as F

from ..consistency import bernoulli_rdrop
from ..multimodal import function_projection
from ..research_layers import masked_mean, observed_label_loss
from . import V7_HEADS
from .sequence_features import FrozenFeatures, ResidueCNN, prior_bias


AXIS_NAMES = (
    "antibacterial", "antiparasitic", "antiviral", "immunoactive",
    "anticancer", "neuropeptide", "metabolic_regulatory", "cell_communication",
)
STATE_NAMES = ("unrecorded", "unknown", "refined")


def axis_columns(labels):
    projection = function_projection(labels, "fine39")
    axes = []
    for name in AXIS_NAMES:
        unknown = labels.index(f"{name}__unknown")
        group = projection[unknown]
        children = tuple(i for i, row in enumerate(projection) if row == group and i != unknown)
        axes.append((unknown, children))
    return tuple(axes)


def axis_observations(targets, axes):
    unknown = targets[:, [column for column, _ in axes]].bool()
    refined = torch.stack([targets[:, children].bool().any(-1) for _, children in axes], dim=-1)
    return torch.where(refined, 2, unknown.long()), unknown & refined


def fit_axis_counts(data):
    targets = torch.as_tensor(data.y[data.indices("train")])
    states, conflicts = axis_observations(targets, axis_columns(data.labels))
    if conflicts.any():
        conflicting = [name for name, present in zip(AXIS_NAMES, conflicts.any(dim=0)) if present]
        raise ValueError(f"V7 Train contains both unknown and specific labels on axes: {conflicting}")
    return F.one_hot(states, num_classes=3).sum(dim=0).tolist()


def nonempty_subsets(size):
    positions = torch.arange(size)
    return ((torch.arange(1, 1 << size)[:, None] >> positions) & 1).bool()


def child_log_odds(log_state, energies, subsets):
    log_not_refined = log_state[:, :2].logsumexp(dim=-1, keepdim=True)
    log_refined = log_state[:, 2:]
    if energies is None:
        return log_refined - log_not_refined
    scores = energies @ subsets.float().T
    log_z = scores.logsumexp(dim=-1, keepdim=True)
    log_yes = scores[..., None].masked_fill(~subsets, -torch.inf).logsumexp(dim=1) - log_z
    log_no = scores[..., None].masked_fill(subsets, -torch.inf).logsumexp(dim=1) - log_z
    return log_refined + log_yes - torch.logaddexp(log_not_refined, log_refined + log_no)


class PeptideV7(nn.Module):
    def __init__(self, input_dim, peptide_input_dim, num_labels, statistics, axis_counts=None,
                 head="structured", dim=256, res_dim=192, heads=8, seeds=4, cnn_channels=64,
                 dropout=.2):
        super().__init__()
        if head not in V7_HEADS:
            raise ValueError("V7 head must be flat or structured")
        self.statistics = statistics
        self.head = head
        local_dim = 4 * cnn_channels
        # Construct the entire shared encoder before either head consumes random numbers.
        self.cnn = ResidueCNN(dropout, cnn_channels)
        self.features = FrozenFeatures(input_dim, res_dim, peptide_input_dim)
        self.fusion = nn.Sequential(nn.Linear(local_dim + res_dim, res_dim), nn.LayerNorm(res_dim), nn.GELU())
        self.queries = nn.Parameter(torch.randn(seeds, res_dim) * .02)
        self.attention = nn.MultiheadAttention(res_dim, heads, dropout=.1, batch_first=True)
        pooled_dim = (seeds + 1) * res_dim + local_dim
        self.projection = nn.Sequential(nn.LayerNorm(pooled_dim), nn.Linear(pooled_dim, dim),
                                        nn.GELU(), nn.Dropout(dropout))

        bias = prior_bias(statistics)
        if head == "flat":
            self.label = nn.Linear(dim, num_labels)
            with torch.no_grad():
                self.label.bias.copy_(bias)
            return

        self.axes = axis_columns(statistics["labels"])
        axis_order = [column for unknown, children in self.axes for column in (unknown, *children)]
        independent = [column for column in range(num_labels) if column not in axis_order]
        self.register_buffer("label_order", torch.tensor(axis_order + independent).argsort(), persistent=False)
        self.state = nn.Linear(dim, len(self.axes) * len(STATE_NAMES))
        self.child = nn.Linear(dim, sum(len(children) for _, children in self.axes if len(children) > 1))
        self.label = nn.Linear(dim, len(independent))
        for size in sorted({len(children) for _, children in self.axes} - {1}):
            self.register_buffer(f"subsets_{size}", nonempty_subsets(size), persistent=False)
        with torch.no_grad():
            self.state.bias.copy_((torch.as_tensor(axis_counts, dtype=torch.float32) + 1).log().flatten())
            self.child.bias.zero_()
            self.label.bias.copy_(bias[independent])

    def encode(self, batch):
        mask = batch["mask"]
        aa = batch["aa"].masked_fill(~mask, 0)
        local = self.cnn({**batch, "aa": aa})
        fused = self.fusion(torch.cat((local, self.features(batch)), dim=-1)).masked_fill(~mask[..., None], 0)
        queries = self.queries.unsqueeze(0).expand(len(mask), -1, -1)
        pooled, _ = self.attention(queries, fused, fused, key_padding_mask=~mask, need_weights=False)
        local_max = local.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        return self.projection(torch.cat((pooled.flatten(1), local_max, masked_mean(fused, mask)), dim=-1))

    def forward(self, batch):
        features = self.encode(batch)
        independent = self.label(features).float()
        if self.head == "flat":
            return {"logits": independent}

        state = self.state(features).float().unflatten(-1, (len(self.axes), len(STATE_NAMES)))
        child = self.child(features).float()
        with torch.autocast(device_type=features.device.type, enabled=False):
            log_state = state.log_softmax(dim=-1)
            axes_logits = []
            offset = 0
            for axis, (_, children) in enumerate(self.axes):
                axis_state = log_state[:, axis]
                unknown = axis_state[:, 1] - axis_state[:, (0, 2)].logsumexp(dim=-1)
                size = len(children)
                if size == 1:
                    specific = child_log_odds(axis_state, None, None)
                else:
                    specific = child_log_odds(axis_state, child[:, offset:offset + size],
                                              getattr(self, f"subsets_{size}"))
                    offset += size
                axes_logits.extend((unknown[:, None], specific))
            logits = torch.cat((*axes_logits, independent), dim=-1).index_select(-1, self.label_order)
        return {"logits": logits, "state_logits": state}

    def compute_loss(self, batch, settings, return_output=False):
        targets = batch["y"]
        state_targets = None
        if self.head == "structured":
            state_targets, _ = axis_observations(targets, self.axes)
        weight = settings.get("rdrop", 0)
        first = self(batch)
        terms = self.loss_terms(first, targets, state_targets, settings)
        agreement = terms["classification"].new_zeros(())
        if self.training and weight > 0:
            second = self(batch)
            second_terms = self.loss_terms(second, targets, state_targets, settings)
            terms = {name: .5 * (value + second_terms[name]) for name, value in terms.items()}
            agreement = bernoulli_rdrop(first["logits"], second["logits"])
        total = terms["classification"] + weight * agreement
        if state_targets is not None:
            total = total + settings.get("state_weight", .2) * terms["state"]
        terms["rdrop"] = agreement
        return (total, terms, first) if return_output else (total, terms)

    def loss_terms(self, output, targets, state_targets, settings):
        terms = {"classification": observed_label_loss(output["logits"], targets, self.statistics,
                                                       settings.get("unlabeled_weight", .3))}
        if state_targets is not None:
            weights = torch.where(state_targets == 0, .3, 1.)
            losses = F.cross_entropy(output["state_logits"].transpose(1, 2), state_targets, reduction="none")
            terms["state"] = ((losses * weights).sum(dim=0) / weights.sum(dim=0)).mean()
        return terms
