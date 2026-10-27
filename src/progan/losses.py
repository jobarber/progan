"""WGAN-GP with the drift term used in the ProGAN paper (Section A.1)."""

from __future__ import annotations

import torch

from progan.models import Discriminator


def gradient_penalty(
    D: Discriminator,
    reals: torch.Tensor,
    fakes: torch.Tensor,
    stage: int,
    alpha: float,
    target: float = 1.0,
) -> torch.Tensor:
    """(||grad D(x_hat)|| - target)^2 on random interpolates between reals and fakes."""
    eps = torch.rand(reals.shape[0], 1, 1, 1, device=reals.device)
    mixed = torch.lerp(reals, fakes, eps).requires_grad_(True)
    scores = D(mixed, stage, alpha)
    (grads,) = torch.autograd.grad(scores.sum(), mixed, create_graph=True)
    norms = grads.flatten(1).norm(dim=1)
    return (norms - target).pow(2) / target**2


def discriminator_loss(
    D: Discriminator,
    reals: torch.Tensor,
    fakes: torch.Tensor,
    stage: int,
    alpha: float,
    gp_weight: float = 10.0,
    drift_weight: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    real_scores = D(reals, stage, alpha)
    fake_scores = D(fakes, stage, alpha)
    gp = gradient_penalty(D, reals, fakes, stage, alpha)
    loss = fake_scores - real_scores + gp_weight * gp + drift_weight * real_scores.pow(2)
    stats = {
        "wdist": (real_scores - fake_scores).mean().item(),
        "gp": gp.mean().item(),
    }
    return loss.mean(), stats


def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return -fake_scores.mean()
