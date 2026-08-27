"""A compact, inspectable centralized training loop.

Federated clients can call ``diffusion_loss`` in their local update loop and
send the resulting state dictionary to ``fedavg``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .diffusion import GaussianDiffusion, TabularMLP


def diffusion_loss(
    model: TabularMLP,
    diffusion: GaussianDiffusion,
    x0: Tensor,
    labels: Tensor | None = None,
) -> Tensor:
    """Compute the simplified noise-prediction MSE from Equation 3."""
    if x0.ndim != 2:
        raise ValueError("x0 must have shape [batch, features]")
    timesteps = torch.randint(0, diffusion.steps, (x0.shape[0],), device=x0.device)
    x_t, noise = diffusion.q_sample(x0, timesteps)
    predicted_noise = model(x_t, timesteps, labels)
    return nn.functional.mse_loss(predicted_noise, noise)


def train_step(
    model: TabularMLP,
    optimizer: torch.optim.Optimizer,
    diffusion: GaussianDiffusion,
    x0: Tensor,
    labels: Tensor | None = None,
) -> float:
    """Perform one local optimization step and return a scalar loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = diffusion_loss(model, diffusion, x0, labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
