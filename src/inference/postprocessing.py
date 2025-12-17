"""
Post-processing: Converting raw fields to medical signals.
"""

import numpy as np
from typing import Tuple


def compute_inflammation_series(V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract global inflammation indices from spatial V field.

    Returns:
        (mean_series, max_series) over time.
    """
    # V shape is (steps, N, N) or just (N, N)
    if V.ndim == 3:
        # Time-series mode
        return np.mean(V, axis=(1, 2)), np.max(V, axis=(1, 2))
    else:
        # Single frame mode
        return np.array([np.mean(V)]), np.array([np.max(V)])


def smooth_signal(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Apply simple moving average smoothing.
    """
    return np.convolve(signal, np.ones(window) / window, mode="valid")


def calculate_stability_index(U: np.ndarray, V: np.ndarray) -> float:
    """
    Quantify how 'chaotic' the tissue state is.
    Higher = more unstable/patterned.
    """
    # Simple variance metric
    return float(np.var(V) / (np.mean(U) + 1e-6))
