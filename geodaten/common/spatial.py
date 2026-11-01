"""Small helpers for selecting cells from regular spatial grids."""

from __future__ import annotations

import numpy as np


def intersecting_cell_indices(
    values: np.ndarray,
    lower: float,
    upper: float,
    resolution: float,
) -> np.ndarray:
    """Return source indices for cells with positive-area overlap."""

    order = np.argsort(values)
    sorted_values = values[order]
    half = resolution / 2
    return order[(sorted_values + half > lower) & (sorted_values - half < upper)]
