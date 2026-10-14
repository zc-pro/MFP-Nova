import torch
from torch.nn import functional as F


def asymmetric_loss(logits, targets, gamma_pos=0, gamma_neg=2, clip=0.03):
    p = logits.sigmoid()
    negative = (1 - p + clip).clamp(max=1)
    base = targets * F.logsigmoid(logits) + (1 - targets) * negative.clamp(min=1e-8).log()
    focusing = targets * (1 - p).pow(gamma_pos) + (1 - targets) * (1 - negative).pow(gamma_neg)
    return -(base * focusing.detach()).mean()
