"""FinDiff-style mixed-type denoiser and local training helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .diffusion import GaussianDiffusion
from .findiff_backbones import SinusoidalPositionEmbeddings, make_backbone


class FinDiff(nn.Module):
    """Learnable categorical embeddings plus a diffusion denoiser.

    The model follows ``sattarov/FinDiff``: categorical columns share one
    embedding table with per-column offsets, numerical columns remain
    continuous, and a residual backbone denoises the concatenated vector.
    """

    def __init__(
        self,
        category_sizes: Sequence[int],
        numerical_dim: int,
        embedding_dim: int = 2,
        hidden_dims: tuple[int, ...] = (1024, 1024, 1024),
        condition_dim: int = 64,
        num_labels: int | None = None,
        cat_decoding: str = "distance",
        backbone_type: str = "mlp",
        backbone_config: dict[str, object] | None = None,
        embedding_learned: bool = True,
    ) -> None:
        super().__init__()
        if not category_sizes and numerical_dim < 1:
            raise ValueError("the model needs categorical or numerical columns")
        if cat_decoding not in {"distance", "logits"}:
            raise ValueError("cat_decoding must be distance or logits")
        self.cat_decoding = cat_decoding
        self.category_sizes = tuple(category_sizes)
        self.category_offsets = tuple(
            sum(self.category_sizes[:index]) for index in range(len(self.category_sizes))
        )
        total_categories = sum(self.category_sizes)
        self.x_cat_emb = nn.Embedding(total_categories, embedding_dim)
        self.x_cat_emb.weight.requires_grad = embedding_learned or cat_decoding == "logits"
        self.numerical_dim = numerical_dim
        self.embedding_dim = embedding_dim
        self.time_embed_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        self.label_embed = nn.Embedding(num_labels, condition_dim) if num_labels is not None else None
        self.uncond_embed = nn.Parameter(torch.randn(1, condition_dim))
        self.backbone = make_backbone(
            backbone_type,
            len(category_sizes) * embedding_dim + numerical_dim,
            condition_dim,
            {
                "mlp_hidden_dim": hidden_dims[0] if hidden_dims else 256,
                "mlp_num_residual_blocks": len(hidden_dims),
                **(backbone_config or {}),
            },
        )
        if cat_decoding == "logits":
            self.cat_head = nn.Sequential(
                nn.Linear(self.data_dim, condition_dim),
                nn.SiLU(),
                nn.Linear(condition_dim, condition_dim),
            )
            self.x_cat_logits = nn.ModuleList(
                nn.Linear(condition_dim, size) for size in self.category_sizes
            )

    @property
    def data_dim(self) -> int:
        return len(self.category_sizes) * self.embedding_dim + self.numerical_dim

    def encode(self, categorical: Tensor, numerical: Tensor) -> Tensor:
        """Convert integer categories and scaled numerical values to FinDiff x0."""
        if categorical.ndim != 2 or categorical.shape[1] != len(self.category_sizes):
            raise ValueError("categorical tensor has the wrong shape")
        parts = [
            self.x_cat_emb(categorical[:, index] + offset)
            for index, offset in enumerate(self.category_offsets)
        ]
        if self.numerical_dim:
            if numerical.ndim != 2 or numerical.shape[1] != self.numerical_dim:
                raise ValueError("numerical tensor has the wrong shape")
            parts.append(numerical)
        return torch.cat(parts, dim=1)

    def embed_categorical(self, x_cat: Tensor) -> Tensor:
        """Embed categorical columns using the original FinDiff method name."""
        if x_cat.ndim != 2 or x_cat.shape[1] != len(self.category_sizes):
            raise ValueError("categorical tensor has the wrong shape")
        return self.encode(x_cat, x_cat.new_empty((x_cat.shape[0], self.numerical_dim), dtype=torch.float32))[
            :, : len(self.category_sizes) * self.embedding_dim
        ]

    embed_x_cat = embed_categorical

    @property
    def synthesizer(self) -> FinDiff:
        """Compatibility view matching the original FinDiff orchestrator."""
        return self

    @property
    def denoiser(self) -> nn.Module:
        """Compatibility view of the selected FinDiff backbone."""
        return self.backbone

    def embed_timestep(self, timesteps: Tensor) -> Tensor:
        """Return the projected sinusoidal timestep embedding."""
        return self.time_embed_mlp(timesteps)

    def embed_label(self, label: Tensor) -> Tensor:
        """Return conditional label embeddings used by the denoiser."""
        if self.label_embed is None:
            raise ValueError("labels are not configured for this model")
        return self.label_embed(label)

    def get_cat_embeddings(self) -> Tensor:
        """Return all learned categorical embeddings for inspection/decoding."""
        if not self.category_sizes:
            return next(self.parameters()).new_empty((0, self.embedding_dim))
        return self.x_cat_emb.weight

    get_x_cat_emb = get_cat_embeddings

    @property
    def dim_input(self) -> int:
        return self.data_dim

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        labels: Tensor | None = None,
        label: Tensor | None = None,
    ) -> Tensor:
        if labels is not None and label is not None:
            raise ValueError("pass either labels or label, not both")
        labels = labels if labels is not None else label
        condition = self.time_embed_mlp(timesteps)
        if labels is None:
            condition = condition + self.uncond_embed.expand(x.shape[0], -1)
        else:
            if self.label_embed is None:
                raise ValueError("labels were supplied but num_labels is not configured")
            condition = condition + self.label_embed(labels)
        predicted = self.backbone(x, condition)
        if self.cat_decoding == "logits":
            category_condition = self.cat_head(predicted) + self.time_embed_mlp(timesteps)
            return predicted, [head(category_condition) for head in self.x_cat_logits]
        return predicted

    @torch.no_grad()
    def decode_categories(self, embedded: Tensor) -> Tensor:
        """Decode categorical embeddings by nearest learned embedding."""
        categorical = []
        offset = 0
        for index, size in enumerate(self.category_sizes):
            values = embedded[:, offset : offset + self.embedding_dim]
            start = self.category_offsets[index]
            distances = torch.cdist(values, self.x_cat_emb.weight[start : start + size])
            categorical.append(distances.argmin(dim=1))
            offset += self.embedding_dim
        return torch.stack(categorical, dim=1) if categorical else embedded.new_empty((len(embedded), 0), dtype=torch.long)

    @torch.no_grad()
    def generate_findiff_data(
        self,
        diffusion: GaussianDiffusion,
        labels: Tensor | None,
        num_samples: int,
    ) -> Tensor:
        """Generate embedded rows using the original FinDiff sampling name."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        device = next(self.parameters()).device
        samples = torch.randn(num_samples, self.data_dim, device=device)
        sampled_labels = None if labels is None else labels[:num_samples].to(device)
        for step in reversed(range(diffusion.steps)):
            timesteps = torch.full((num_samples,), step, dtype=torch.long, device=device)
            samples = diffusion.predict_mean(samples, self(samples, timesteps, sampled_labels), timesteps)
            if step:
                samples = samples + diffusion.betas[step].sqrt() * torch.randn_like(samples)
        return samples


class FinDiffPrivateWrapper(nn.Module):
    """Single-input adapter for Opacus' per-sample gradient instrumentation.

    Opacus can instrument arbitrary models reliably when their forward method
    receives one tensor. The packed input contains categorical codes, scaled
    numerical values, and the optional class label.
    """

    def __init__(self, model: FinDiff, diffusion: GaussianDiffusion) -> None:
        super().__init__()
        self.model = model
        self.diffusion = diffusion
        self.cat_dim = len(model.category_sizes)
        self.num_dim = model.numerical_dim

    def forward(self, packed: Tensor) -> Tensor:
        cat_end = self.cat_dim
        num_end = cat_end + self.num_dim
        categorical = packed[:, :cat_end].long()
        numerical = packed[:, cat_end:num_end]
        labels = packed[:, num_end].long() if packed.shape[1] > num_end else None
        x0 = self.model.encode(categorical, numerical)
        timesteps = torch.randint(0, self.diffusion.steps, (len(packed),), device=packed.device)
        x_t, noise = self.diffusion.q_sample(x0, timesteps)
        prediction = self.model(x_t, timesteps, labels)
        return (prediction - noise).square().mean(dim=1)


def findiff_loss(
    model: FinDiff,
    diffusion: GaussianDiffusion,
    categorical: Tensor,
    numerical: Tensor,
    labels: Tensor | None = None,
) -> Tensor:
    """Compute Equation 3 after constructing the FinDiff representation."""
    encoder = model if hasattr(model, "encode") else getattr(model, "_module", None)
    if encoder is None or not hasattr(encoder, "encode"):
        raise TypeError("model must provide FinDiff.encode")
    x0 = encoder.encode(categorical, numerical)
    timesteps = torch.randint(0, diffusion.steps, (x0.shape[0],), device=x0.device)
    x_t, noise = diffusion.q_sample(x0, timesteps)
    return nn.functional.mse_loss(model(x_t, timesteps, labels), noise)
