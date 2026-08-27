"""Residual-adaptive random refinement for domain PDE collocation.

RAR never replaces the baseline uniform/SDF-layer samples.  It draws a fresh
candidate pool from the same domain sampler, evaluates the current pointwise
PDE residual, then samples without replacement with probability proportional
to a tempered residual score.  A small uniform mixture prevents zero-probable
regions and preserves exploration.
"""
import torch

import sampling as S
from losses import pde_residual


def residual_scores(field, domain, points, power_scale=1.0,
                    microbatch=2000):
    """Detached squared PDE residual for a candidate tensor."""
    values = []
    for start in range(0, points.shape[0], microbatch):
        pts = points[start:start + microbatch].detach().clone()
        residual = pde_residual(field, domain, pts, power_scale)
        values.append(residual.detach().flatten().square())
    return torch.cat(values)


def sample_rar(field, domain, n, x1, x2, device, power_scale=1.0,
               candidate_factor=4, power=1.0, uniform_mix=0.05,
               score_microbatch=2000, return_info=False):
    """Draw residual-weighted random PDE points from one physical domain."""
    if n <= 0:
        empty = torch.empty((0, 2), device=device)
        return (empty, {}) if return_info else empty
    if candidate_factor < 1 or power < 0 or not 0 <= uniform_mix <= 1:
        raise ValueError("RAR requires factor>=1, power>=0 and 0<=mix<=1")
    candidate_count = max(n, int(candidate_factor * n))
    candidates = S.sample_domain(domain, candidate_count, x1, x2, device)
    score = residual_scores(field, domain, candidates, power_scale,
                            score_microbatch)

    # Normalize before exponentiation to avoid overflow and make `power`
    # independent of the dimensional residual scale of each domain.
    normalized = score / score.mean().clamp_min(1e-30)
    weighted = normalized.clamp_min(1e-12).pow(power)
    probabilities = ((1.0 - uniform_mix) * weighted / weighted.sum()
                     + uniform_mix / candidate_count)
    indices = torch.multinomial(probabilities, n, replacement=False)
    selected = candidates[indices].detach()
    if not return_info:
        return selected
    info = {
        "candidate_mean": float(score.mean()),
        "candidate_max": float(score.max()),
        "selected_mean": float(score[indices].mean()),
        "selected_max": float(score[indices].max()),
    }
    return selected, info
