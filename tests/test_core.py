import numpy as np
import pytest
import torch

from dp_fedtabdiff.data import iid_partition
from dp_fedtabdiff.diffusion import GaussianDiffusion, TabularMLP
from dp_fedtabdiff.federated import fedavg
from dp_fedtabdiff.privacy import clip_per_sample_gradients
from dp_fedtabdiff.training import diffusion_loss


def test_diffusion_forward_shapes_and_noise_recovery():
    diffusion = GaussianDiffusion(steps=10)
    x = torch.zeros(4, 3)
    noisy, noise = diffusion.q_sample(x, torch.tensor([0, 1, 5, 9]))
    assert noisy.shape == noise.shape == x.shape


def test_diffusion_rejects_invalid_timestep():
    with pytest.raises(ValueError):
        GaussianDiffusion(steps=10).q_sample(torch.zeros(1, 2), torch.tensor([10]))


def test_mlp_predicts_tabular_noise():
    model = TabularMLP(data_dim=5, hidden_dims=(8, 8), condition_dim=4, num_labels=3)
    output = model(torch.randn(2, 5), torch.tensor([1, 2]), torch.tensor([0, 1]))
    assert output.shape == (2, 5)


def test_fedavg_is_sample_weighted():
    result = fedavg([{"weight": torch.tensor([0.0])}, {"weight": torch.tensor([2.0])}], [1, 3])
    assert torch.equal(result["weight"], torch.tensor([1.5]))


def test_clipping_bounds_each_sample():
    clipped = clip_per_sample_gradients(torch.tensor([[3.0, 4.0], [1.0, 0.0]]), 1.0)
    assert torch.all(clipped.norm(dim=1) <= 1.0 + 1e-6)


def test_iid_partition_is_complete_and_deterministic():
    first = iid_partition(11, 3, seed=7)
    second = iid_partition(11, 3, seed=7)
    assert np.array_equal(np.concatenate(first), np.concatenate(second))
    assert np.array_equal(np.sort(np.concatenate(first)), np.arange(11))


def test_diffusion_loss_is_scalar():
    diffusion = GaussianDiffusion(steps=10)
    model = TabularMLP(data_dim=3, hidden_dims=(8,), condition_dim=4)
    loss = diffusion_loss(model, diffusion, torch.randn(4, 3))
    assert loss.ndim == 0
    assert torch.isfinite(loss)
