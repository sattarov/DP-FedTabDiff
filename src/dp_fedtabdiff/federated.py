"""Framework-independent federated aggregation primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor


def fedavg(
    client_states: Sequence[Mapping[str, Tensor]],
    client_sizes: Sequence[int],
) -> dict[str, Tensor]:
    """Return the sample-weighted average of client model state dictionaries."""
    if not client_states or len(client_states) != len(client_sizes):
        raise ValueError("client states and sizes must be non-empty and aligned")
    if any(size <= 0 for size in client_sizes):
        raise ValueError("client sizes must be positive")
    keys = tuple(client_states[0])
    if any(tuple(state) != keys for state in client_states):
        raise ValueError("all client states must have identical keys")
    total = sum(client_sizes)
    result: dict[str, Tensor] = {}
    for key in keys:
        reference = client_states[0][key]
        if not torch.is_floating_point(reference):
            result[key] = reference.clone()
            continue
        result[key] = sum(
            state[key].to(dtype=reference.dtype) * (size / total)
            for state, size in zip(client_states, client_sizes)
        )
    return result
