import torch

from dp_fedtabdiff.diffusion import GaussianDiffusion
from dp_fedtabdiff.findiff import FinDiff, findiff_loss


def test_findiff_encodes_and_decodes_categories():
    model = FinDiff([3, 2], numerical_dim=2, embedding_dim=2, hidden_dims=(8,), condition_dim=4)
    categorical = torch.tensor([[0, 1], [2, 0]])
    numerical = torch.randn(2, 2)
    encoded = model.encode(categorical, numerical)
    assert encoded.shape == (2, 6)
    assert model.decode_categories(encoded).shape == categorical.shape


def test_findiff_loss_is_scalar():
    model = FinDiff([3], numerical_dim=1, embedding_dim=2, hidden_dims=(8,), condition_dim=4, num_labels=2)
    loss = findiff_loss(
        model,
        GaussianDiffusion(10),
        torch.tensor([[0], [1], [2]]),
        torch.randn(3, 1),
        torch.tensor([0, 1, 0]),
    )
    assert loss.ndim == 0
