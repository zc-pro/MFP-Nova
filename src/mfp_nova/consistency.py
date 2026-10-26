"""Bernoulli consistency for EMA targets and paired dropout predictions."""
from torch.nn import functional as F


def bernoulli_rdrop(first_logits, second_logits):
    """Mean symmetric KL over independent labels; both views receive gradients.

    For Bernoulli logits a,b, (KL(p||q)+KL(q||p))/2 equals
    (sigmoid(a)-sigmoid(b))*(a-b)/2. FP32 avoids log(0) and
    cancellation between log-probabilities for confident predictions.
    No targets are imputed and no softmax couples different labels.
    """
    first, second = first_logits.float(), second_logits.float()
    return (.5 * (first.sigmoid() - second.sigmoid()) * (first - second)).mean()


def consistency_weight(settings, epoch):
    start = settings["consistency_start_epoch"]
    progress = min(1., max(0., (epoch-start+1)/settings["consistency_ramp_epochs"]))
    return settings["consistency_weight"]*progress


def bernoulli_consistency(student_logits, teacher_logits, targets, mask=None):
    """Label-wise KL, with separate high/low teacher-score budgets (1 and 0.1).

    A score above 0.5 defines a weighting group, not a confirmed positive.
    Known positives are excluded so low teacher scores cannot oppose them.
    """
    student = student_logits.float()
    teacher = teacher_logits.detach().float()
    q = teacher.sigmoid()
    kl = (q*(F.logsigmoid(teacher)-F.logsigmoid(student))
          +(1-q)*(F.logsigmoid(-teacher)-F.logsigmoid(-student))).clamp_min(0)
    unlabeled = targets == 0
    if mask is not None:
        unlabeled = unlabeled & mask.bool()

    def label_mean(selected):
        counts = selected.sum(0)
        values = (kl*selected).sum(0)/counts.clamp_min(1)
        return values.sum()/(counts > 0).sum().clamp_min(1)

    high = label_mean(unlabeled & (q >= .5))
    low = label_mean(unlabeled & (q < .5))
    return high+.1*low, {"consistency_high": high, "consistency_low": low}
