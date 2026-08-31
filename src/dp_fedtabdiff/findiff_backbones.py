"""Neural backbones used by the canonical FinDiff synthesizer."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SinusoidalPositionEmbeddings(nn.Module):
    """The sinusoidal timestep encoding used by the reference FinDiff model."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 2 or dimension % 2:
            raise ValueError("the timestep embedding dimension must be a positive even number")
        self.dimension = dimension

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.dimension // 2
        scale = torch.log(torch.tensor(10_000.0, device=timesteps.device)) / (half - 1)
        frequencies = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        angles = timesteps[:, None].float() * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ResidualBlock(nn.Module):
    """A residual MLP block from the reference implementation."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.layer2(self.activation(self.layer1(inputs)))


class MLPBackbone(nn.Module):
    """Residual MLP backbone used by the original FinDiff release."""

    def __init__(
        self,
        num_features: int,
        condition_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(num_features + condition_dim, hidden_dim)
        self.blocks = nn.ModuleList(ResidualBlock(hidden_dim) for _ in range(num_blocks))
        self.output_proj = nn.Linear(hidden_dim, num_features)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        hidden = self.input_proj(torch.cat((x, condition), dim=1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_proj(hidden)


class TransformerBackbone(nn.Module):
    """Transformer backbone supported by the reference FinDiff package."""

    def __init__(
        self,
        num_features: int,
        condition_dim: int,
        d_model: int = 256,
        n_head: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
    ) -> None:
        super().__init__()
        self.condition_proj = nn.Linear(condition_dim, d_model)
        self.feature_embed = nn.Linear(1, d_model)
        self.position_embed = nn.Embedding(num_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        embedded = self.feature_embed(x.unsqueeze(-1))
        embedded = embedded + self.position_embed.weight.unsqueeze(0)
        embedded = embedded + self.condition_proj(condition).unsqueeze(1)
        return self.output_proj(self.encoder(embedded)).squeeze(-1)


def make_backbone(
    backbone_type: str,
    num_features: int,
    condition_dim: int,
    config: dict[str, object] | None = None,
) -> nn.Module:
    """Construct one of the backbones exposed by canonical FinDiff."""
    config = config or {}
    if backbone_type == "mlp":
        return MLPBackbone(
            num_features,
            condition_dim,
            int(config.get("mlp_hidden_dim", 256)),
            int(config.get("mlp_num_residual_blocks", 2)),
        )
    if backbone_type == "transformer":
        return TransformerBackbone(
            num_features,
            condition_dim,
            int(config.get("d_model", 256)),
            int(config.get("transformer_n_head", 8)),
            int(config.get("transformer_num_layers", 6)),
            int(config.get("transformer_dim_feedforward", 1024)),
        )
    raise ValueError(f"unsupported FinDiff backbone: {backbone_type}")
