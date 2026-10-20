"""Main-priority reconstruction gradients, confined to shared parameters."""
import math
import torch


def backward_reconstruction(primary_loss, auxiliary_loss, parameters, max_ratio=1.):
    """Losses already include task weights, accumulation divisor and AMP scale.

    Project only a conflicting auxiliary component; then bound its shared norm.
    Private gradients retain their original values. AdamW/clipping can still
    change update directions: this is a gradient-space, per-microbatch rule.
    """
    parameters = tuple(p for p in parameters if p.requires_grad)
    primary = torch.autograd.grad(primary_loss, parameters, retain_graph=True, allow_unused=True)
    auxiliary = torch.autograd.grad(auxiliary_loss, parameters, allow_unused=True)
    shared = [(m, a) for m, a in zip(primary, auxiliary) if m is not None and a is not None]
    zero = primary_loss.detach().float().new_zeros(())
    dot = sum(((m*a).sum() for m, a in shared), zero)
    main_sq = sum((m.square().sum() for m, _ in shared), zero)
    aux_sq = sum((a.square().sum() for _, a in shared), zero)
    values = torch.stack([dot, main_sq, aux_sq]).detach().cpu().tolist()
    finite = all(math.isfinite(v) for v in values)
    coefficient, scale = 0., 1.
    cosine, raw_ratio, applied_ratio = None, None, None
    conflict, capped = False, False
    if finite:
        product, main_norm_sq, aux_norm_sq = values
        if main_norm_sq > 0:
            raw_ratio = (aux_norm_sq/main_norm_sq)**.5
            cosine = product/(main_norm_sq*aux_norm_sq)**.5 if aux_norm_sq > 0 else None
            conflict = product < 0
            coefficient = product/main_norm_sq if conflict else 0.
            projected_sq = max(0., aux_norm_sq-coefficient*product)
            projected_ratio = (projected_sq/main_norm_sq)**.5
            capped = projected_ratio > max_ratio
            scale = max_ratio/projected_ratio if capped else 1.
            applied_ratio = projected_ratio*scale
        elif shared:
            # No shared main signal: do not let reconstruction alone move it.
            scale, applied_ratio, capped = 0., 0., aux_norm_sq > 0
    # Nonfinite scaled gradients are passed through for the existing GradScaler
    # skip or FP32 finite-gradient error, never silently replaced with zeros.
    for p, main, aux in zip(parameters, primary, auxiliary):
        if main is None and aux is None:
            continue
        if main is not None and aux is not None:
            contribution = main+(aux-coefficient*main)*scale if finite else main+aux
        else:
            contribution = main if aux is None else aux
        if p.grad is None:
            p.grad = contribution
        else:
            p.grad.add_(contribution)
    return {'finite':finite, 'conflict':conflict, 'capped':capped,
            'cosine':cosine, 'weighted_aux_main_ratio':raw_ratio,
            'applied_aux_main_ratio':applied_ratio}
