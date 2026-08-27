"""Flower client/server adapters for the DP-FedTabDiff training loop."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .diffusion import GaussianDiffusion, TabularMLP
from .privacy import PersistentPrivacyEngine
from .training import diffusion_loss, train_step


def _require_flower():
    try:
        import flwr as fl
    except ImportError as error:
        raise ImportError("install the 'federated' extra to use Flower adapters") from error
    return fl


class DPFedTabDiffClient:
    """A Flower NumPyClient that trains one local model per fit call."""

    def __init__(
        self,
        model: TabularMLP,
        diffusion: GaussianDiffusion,
        x_local: Tensor,
        labels: Tensor | None = None,
        local_steps: int = 1,
        optimizer_factory: Callable[[object], torch.optim.Optimizer] | None = None,
        dp_params: dict[str, float] | None = None,
    ) -> None:
        self.model, self.diffusion = model, diffusion
        self.x_local, self.labels = x_local, labels
        self.local_steps = local_steps
        self.optimizer_factory = optimizer_factory or (
            lambda parameters: torch.optim.Adam(parameters, lr=1e-3)
        )
        self.dp_params = dp_params
        self._private_optimizer = None
        self._private_loader = None
        self._privacy = None

    def _local_fit(self) -> dict[str, float]:
        if self.dp_params is None:
            optimizer = self.optimizer_factory(self.model.parameters())
            for _ in range(self.local_steps):
                train_step(self.model, optimizer, self.diffusion, self.x_local, self.labels)
            return {}
        if self.labels is None:
            raise ValueError("DP training requires labels or an unlabeled tensor dataset")
        if self._private_optimizer is None:
            loader = DataLoader(
                TensorDataset(self.x_local, self.labels),
                batch_size=int(self.dp_params.get("batch_size", 32)),
                shuffle=True,
            )
            self._privacy = PersistentPrivacyEngine(
                target_delta=float(self.dp_params["target_delta"]),
                noise_multiplier=(
                    float(self.dp_params["noise_multiplier"])
                    if "noise_multiplier" in self.dp_params
                    else None
                ),
                max_grad_norm=float(self.dp_params["max_grad_norm"]),
                target_epsilon=(
                    float(self.dp_params["target_epsilon"])
                    if "target_epsilon" in self.dp_params
                    else None
                ),
                epochs=int(self.dp_params.get("epochs", 1)),
            )
            optimizer = self.optimizer_factory(self.model.parameters())
            self.model, self._private_optimizer, self._private_loader = self._privacy.attach(
                self.model, optimizer, loader
            )
        iterator = iter(self._private_loader)
        for _ in range(self.local_steps):
            try:
                batch_x, batch_y = next(iterator)
            except StopIteration:
                iterator = iter(self._private_loader)
                batch_x, batch_y = next(iterator)
            self._private_optimizer.zero_grad(set_to_none=True)
            loss = diffusion_loss(self.model, self.diffusion, batch_x, batch_y)
            loss.backward()
            self._private_optimizer.step()
        return {"epsilon": self._privacy.epsilon()}

    def as_numpy_client(self):
        fl = _require_flower()
        parent = self

        class Client(fl.client.NumPyClient):
            def get_parameters(self, config):
                return [value.detach().cpu().numpy() for value in parent.model.state_dict().values()]

            def set_parameters(self, parameters):
                state = OrderedDict(
                    (key, torch.from_numpy(value))
                    for key, value in zip(parent.model.state_dict(), parameters)
                )
                parent.model.load_state_dict(state, strict=True)

            def fit(self, parameters, config):
                self.set_parameters(parameters)
                metrics = parent._local_fit()
                return self.get_parameters(config), len(parent.x_local), metrics

            def evaluate(self, parameters, config):
                self.set_parameters(parameters)
                return 0.0, len(parent.x_local), {}

        return Client()


def start_simulation(
    client_fn,
    num_clients: int,
    rounds: int,
    strategy=None,
):
    """Start a local Flower simulation using the installed Flower version."""
    fl = _require_flower()
    if strategy is None:
        strategy = fl.server.strategy.FedAvg(min_fit_clients=num_clients, min_available_clients=num_clients)
    return fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


def make_strategy(num_clients: int, strategy_name: str = "fedavg"):
    """Build a Flower server strategy for the supported aggregation methods."""
    fl = _require_flower()
    if num_clients < 1:
        raise ValueError("num_clients must be positive")
    strategies = {
        "fedavg": fl.server.strategy.FedAvg,
        "fedadam": fl.server.strategy.FedAdam,
        "fedprox": fl.server.strategy.FedProx,
        "fedyogi": fl.server.strategy.FedYogi,
    }
    try:
        strategy_type = strategies[strategy_name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported Flower strategy: {strategy_name}") from error
    return strategy_type(min_fit_clients=num_clients, min_available_clients=num_clients)
