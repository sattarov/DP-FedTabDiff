"""Diffusion and conditional MLP components used by DP-FedTabDiff."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def timestep_embedding(timesteps: Tensor, dimension: int, max_period: int = 10_000) -> Tensor:
    """Create the sinusoidal timestep embedding used by the denoiser."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if dimension % 2:
        embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
    return embedding


@dataclass
class GaussianDiffusion:
    """DDPM schedule and the closed-form forward/reverse transitions."""

    steps: int = 500
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    def __post_init__(self) -> None:
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device | str) -> GaussianDiffusion:
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def _index(self, values: Tensor, timesteps: Tensor, shape: tuple[int, ...]) -> Tensor:
        if timesteps.dtype not in (torch.int32, torch.int64):
            raise TypeError("timesteps must contain integer indices")
        if torch.any((timesteps < 0) | (timesteps >= self.steps)):
            raise ValueError("timesteps are outside the diffusion schedule")
        return values.gather(0, timesteps).reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Sample x_t from q(x_t | x_0), returning both x_t and the noise."""
        noise = torch.randn_like(x0) if noise is None else noise
        if noise.shape != x0.shape or timesteps.shape != (x0.shape[0],):
            raise ValueError("x0, noise, and timesteps have incompatible shapes")
        alpha_bar = self._index(self.alpha_bars.to(x0.device), timesteps, x0.shape)
        return alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise, noise

    def predict_mean(self, x_t: Tensor, predicted_noise: Tensor, timesteps: Tensor) -> Tensor:
        """Compute the DDPM reverse-process mean from predicted noise."""
        alpha = self._index(self.alphas.to(x_t.device), timesteps, x_t.shape)
        beta = self._index(self.betas.to(x_t.device), timesteps, x_t.shape)
        alpha_bar = self._index(self.alpha_bars.to(x_t.device), timesteps, x_t.shape)
        return (x_t - beta * predicted_noise / (1 - alpha_bar).sqrt()) / alpha.sqrt()


class TabularMLP(nn.Module):
    """Conditional MLP noise predictor for the embedded tabular representation."""

    def __init__(
        self,
        data_dim: int,
        hidden_dims: tuple[int, ...] = (1024, 1024, 1024),
        condition_dim: int = 64,
        num_categories: int | None = None,
        num_labels: int | None = None,
        embedding_dim: int = 2,
    ) -> None:
        super().__init__()
        if data_dim < 1 or not hidden_dims:
            raise ValueError("data_dim and hidden_dims must be non-empty")
        self.time = nn.Sequential(nn.Linear(condition_dim, condition_dim), nn.SiLU())
        self.category = (
            nn.Embedding(num_categories, embedding_dim) if num_categories is not None else None
        )
        self.label = nn.Embedding(num_labels, condition_dim) if num_labels is not None else None
        input_dim = data_dim + condition_dim
        layers: list[nn.Module] = []
        for width in hidden_dims:
            layers.extend((nn.Linear(input_dim, width), nn.SiLU()))
            input_dim = width
        layers.append(nn.Linear(input_dim, data_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor, timesteps: Tensor, labels: Tensor | None = None) -> Tensor:
        condition = self.time(timestep_embedding(timesteps, self.time[0].in_features))
        if labels is not None:
            if self.label is None:
                raise ValueError("labels were supplied but num_labels is not configured")
            condition = condition + self.label(labels)
        return self.network(torch.cat((x, condition), dim=-1))
