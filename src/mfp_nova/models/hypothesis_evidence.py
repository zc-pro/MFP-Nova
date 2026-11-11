"""H6: select residues by function query and re-encode the original token grid."""

import torch
from torch.nn import functional as F


def _top_positions(weights, mask, sizes):
    order = weights.masked_fill(~mask, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    rank = torch.arange(mask.shape[-1], device=mask.device)
    return torch.zeros_like(mask).scatter(-1, order, rank < sizes[:, None])


def _attention_mask(attention, mask):
    lengths = mask.sum(-1)
    sizes = torch.minimum(lengths - 1, (lengths + 1) // 2)
    return _top_positions(attention, mask, sizes)


def match_random_mask(mask, keep, generator=None):
    """Draw a position-matched subset and flag rows with only one possibility."""
    position = torch.arange(mask.shape[-1], device=mask.device)
    span = ((position + 1) * mask).amax(-1, keepdim=True).clamp_min(1)
    segment = (4 * position // span).clamp_max(3)
    weights = torch.rand(mask.shape, device=mask.device, generator=generator)
    random_keep = torch.zeros_like(keep)
    degenerate = torch.ones(mask.shape[0], dtype=torch.bool, device=mask.device)
    for index in range(4):
        valid = mask & (segment == index)
        sizes = (valid & keep).sum(-1)
        random_keep |= _top_positions(weights, valid, sizes)
        degenerate &= (sizes == 0) | (sizes == valid.sum(-1))
    return random_keep, degenerate


def _mean_attention(attention):
    if isinstance(attention, tuple):
        return (attention[0].detach().float() + attention[1].detach().float()) / 2
    return attention.detach().float()


def evidence_selection(attention, mask, targets, observed, method="attention", generator=None):
    """Sample one known positive label per eligible sequence, uniformly."""
    positive = targets.bool() & observed
    rows = torch.where(positive.any(-1) & (mask.sum(-1) > 1))[0]
    selected = positive[rows]
    scores = torch.rand(selected.shape, device=selected.device, generator=generator)
    labels = scores.masked_fill(~selected, -torch.inf).argmax(-1)
    keep = _attention_mask(_mean_attention(attention)[rows, labels], mask[rows])
    if method == "random":
        keep, _ = match_random_mask(mask[rows], keep, generator)
    elif method != "attention":
        raise ValueError(f"Unknown H6 selection: {method}")
    return rows, labels, keep


def evidence_auxiliary(full_logits, kept_logits, rows, labels, reduction="mean"):
    """Supervise the retained view and align it to a stopped full-view target."""
    view = kept_logits[torch.arange(labels.numel(), device=labels.device), labels].float()
    full_probability = full_logits[rows, labels].detach().float().sigmoid()
    keep_loss = F.binary_cross_entropy_with_logits(view, torch.ones_like(view), reduction="sum")
    agreement_loss = F.binary_cross_entropy_with_logits(view, full_probability, reduction="sum")
    loss = 0.1 * (keep_loss + agreement_loss)
    return loss if reduction == "sum" else loss / max(labels.numel(), 1)


def _effect(probability, full_probability, label):
    delta = probability - full_probability
    selected = delta.gather(1, label[:, None]).squeeze(1)
    other = delta.abs().scatter(1, label[:, None], 0).sum(1) / max(delta.shape[1] - 1, 1)
    return torch.stack((selected, other), dim=-1)


@torch.inference_mode()
def evidence_diagnostics(backbone, tokens, mask, targets, observed, repeats=8, generator=None,
                         attention_backbone=None):
    """Score full, kept, removed, and position-matched random views per positive.

    Use an eval-mode backbone. An optional randomized attention backbone can
    select masks while the fitted backbone still supplies all probabilities.
    Each effect holds the signed target change and mean absolute non-target change.
    """
    full = backbone(tokens, mask, return_attention=attention_backbone is None)
    attention = full.attention if attention_backbone is None else attention_backbone(
        tokens, mask, return_attention=True).attention
    attention = _mean_attention(attention)
    rows, labels = torch.where(targets.bool() & observed & (mask.sum(-1) > 1)[:, None])
    full_probability = full.logits.float().sigmoid()
    effects = {"keep_effect": [], "delete_effect": [], "random_keep_effect": [],
               "random_delete_effect": [], "random_degenerate": [], "kept_residues": []}
    for positions in torch.arange(rows.numel(), device=rows.device).split(max(1, tokens.shape[0])):
        if not positions.numel():
            continue
        row, label = rows[positions], labels[positions]
        selected_tokens, full_mask = tokens[row], mask[row]
        keep = _attention_mask(attention[row, label], full_mask)
        original = full_probability[row]

        def score(view_mask):
            probability = backbone(selected_tokens, view_mask).logits.float().sigmoid()
            return _effect(probability, original, label)

        effects["keep_effect"].append(score(keep))
        effects["delete_effect"].append(score(full_mask & ~keep))
        effects["kept_residues"].append(keep.sum(-1))
        random_keep, random_delete = [], []
        for _ in range(repeats):
            random_mask, degenerate = match_random_mask(full_mask, keep, generator)
            random_keep.append(score(random_mask))
            random_delete.append(score(full_mask & ~random_mask))
        effects["random_keep_effect"].append(torch.stack(random_keep, dim=1))
        effects["random_delete_effect"].append(torch.stack(random_delete, dim=1))
        effects["random_degenerate"].append(degenerate)

    result = {"row_indices": rows, "label_indices": labels, "full_target_probability": full_probability[rows, labels]}
    for key, values in effects.items():
        if values:
            result[key] = torch.cat(values)
        elif key == "random_degenerate":
            result[key] = torch.empty(0, dtype=torch.bool, device=rows.device)
        elif key == "kept_residues":
            result[key] = rows
        else:
            shape = (0, repeats, 2) if key.startswith("random_") else (0, 2)
            result[key] = full_probability.new_empty(shape)
    return result
