import pandas as pd
import pytest
import torch

from dp_fedtabdiff.data import DataTransformer
from dp_fedtabdiff.diffusion import GaussianDiffusion, TabularMLP
from dp_fedtabdiff.federated_flower import DPFedTabDiffClient
from dp_fedtabdiff.privacy import PersistentPrivacyEngine


def test_transformer_fits_and_round_trips():
    frame = pd.DataFrame({"city": ["a", "b", "a"], "amount": [1.0, 2.0, 3.0]})
    transformer = DataTransformer(["city"], ["amount"]).fit(frame.iloc[:2])
    transformed = transformer.transform(frame.iloc[:2])
    restored = transformer.inverse_transform(transformed["cat"], transformed["num"])
    assert restored["city"].tolist() == ["a", "b"]


def test_transformer_requires_fit():
    with pytest.raises(RuntimeError):
        DataTransformer(["city"], ["amount"]).transform(pd.DataFrame({"city": ["a"], "amount": [1]}))


def test_privacy_engine_requires_opacus_attachment():
    engine = PersistentPrivacyEngine(1e-5, 1.0, 1.0)
    with pytest.raises(RuntimeError):
        engine.epsilon()


def test_flower_client_can_be_constructed():
    client = DPFedTabDiffClient(
        TabularMLP(3, (4,), 4),
        GaussianDiffusion(5),
        torch.randn(4, 3),
        torch.tensor([0, 1, 0, 1]),
    )
    assert client.as_numpy_client().get_parameters({})[0].ndim > 0
