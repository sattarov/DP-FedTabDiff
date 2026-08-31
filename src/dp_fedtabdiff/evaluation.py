"""Small dependency-light evaluation helpers."""

from __future__ import annotations

import numpy as np


def marginal_total_variation(real: np.ndarray, synthetic: np.ndarray) -> float:
    """Average total variation distance for aligned categorical columns."""
    if real.ndim != 2 or synthetic.ndim != 2 or real.shape[1] != synthetic.shape[1]:
        raise ValueError("real and synthetic arrays must have the same number of columns")
    distances = []
    for column in range(real.shape[1]):
        values = np.union1d(real[:, column], synthetic[:, column])
        real_counts = np.array([(real[:, column] == value).mean() for value in values])
        syn_counts = np.array([(synthetic[:, column] == value).mean() for value in values])
        distances.append(0.5 * np.abs(real_counts - syn_counts).sum())
    return float(np.mean(distances))
