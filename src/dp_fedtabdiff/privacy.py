"""Small, explicit DP-SGD primitives.

Use Opacus for production accounting. These functions make the clipping and
noise operations from Equation 6 easy to inspect and test.
"""

from __future__ import annotations

import torch
from torch import Tensor


def clip_per_sample_gradients(gradients: Tensor, max_norm: float) -> Tensor:
    """Clip a tensor shaped ``[batch, ...]`` independently along batch."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if gradients.ndim < 2:
        raise ValueError("gradients must have a batch dimension and parameter dimensions")
    flat = gradients.flatten(start_dim=1)
    norms = flat.norm(2, dim=1)
    scale = (max_norm / norms.clamp_min(1e-12)).clamp_max(1.0)
    return gradients * scale.reshape((-1,) + (1,) * (gradients.ndim - 1))


def add_gaussian_noise(gradient: Tensor, noise_multiplier: float, max_norm: float, batch_size: int) -> Tensor:
    """Add calibrated Gaussian noise to an averaged clipped gradient."""
    if noise_multiplier < 0 or max_norm <= 0 or batch_size <= 0:
        raise ValueError("noise_multiplier, max_norm, and batch_size are invalid")
    standard_deviation = noise_multiplier * max_norm / batch_size
    return gradient + torch.randn_like(gradient) * standard_deviation


class PersistentPrivacyEngine:
    """Keep one Opacus accountant alive across all client local rounds."""

    def __init__(
        self,
        target_delta: float,
        noise_multiplier: float | None,
        max_grad_norm: float,
        target_epsilon: float | None = None,
        epochs: int = 1,
    ) -> None:
        if target_delta <= 0 or target_delta >= 1:
            raise ValueError("target_delta must be between zero and one")
        self.target_delta = target_delta
        if target_epsilon is None and noise_multiplier is None:
            raise ValueError("provide target_epsilon or noise_multiplier")
        if target_epsilon is not None and target_epsilon <= 0:
            raise ValueError("target_epsilon must be positive")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        self.target_epsilon = target_epsilon
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.engine = None
        self.steps = 0

    def attach(self, module, optimizer, data_loader, grad_sample_mode: str = "hooks"):
        try:
            from opacus import PrivacyEngine
        except ImportError as error:
            raise ImportError("install the 'privacy' extra to use Opacus") from error
        if self.engine is not None:
            raise RuntimeError("this privacy engine has already been attached")
        self.engine = PrivacyEngine()
        if self.target_epsilon is not None:
            module, optimizer, data_loader = self.engine.make_private_with_epsilon(
                module=module,
                optimizer=optimizer,
                data_loader=data_loader,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                epochs=self.epochs,
                max_grad_norm=self.max_grad_norm,
                grad_sample_mode=grad_sample_mode,
            )
            self.noise_multiplier = float(optimizer.noise_multiplier)
        else:
            module, optimizer, data_loader = self.engine.make_private(
                module=module,
                optimizer=optimizer,
                data_loader=data_loader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                grad_sample_mode=grad_sample_mode,
            )
        return module, optimizer, data_loader

    def step(self) -> None:
        if self.engine is None:
            raise RuntimeError("attach the privacy engine before recording steps")
        self.steps += 1

    def epsilon(self) -> float:
        if self.engine is None:
            raise RuntimeError("attach the privacy engine before querying epsilon")
        return float(self.engine.get_epsilon(self.target_delta))
