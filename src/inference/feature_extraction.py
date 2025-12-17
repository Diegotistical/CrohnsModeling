"""
Feature Extraction for ML pipelines.
"""

import numpy as np
from scipy.stats import entropy


def compute_spatial_entropy(frame: np.ndarray, bins: int = 100) -> float:
    """
    Compute Shannon entropy of the inflammation distribution.
    High entropy = complex, widespread pattern.
    Low entropy = localized or uniform state.
    """
    hist, _ = np.histogram(frame, bins=bins, density=True)
    return float(entropy(hist + 1e-9))  # Add epsilon for log(0)


def compute_roughness(frame: np.ndarray) -> float:
    """
    Compute Laplacian variance (measure of texture/roughness).
    """
    # Simple discrete laplacian approximation
    lap = (
        np.roll(frame, 1, 0)
        + np.roll(frame, -1, 0)
        + np.roll(frame, 1, 1)
        + np.roll(frame, -1, 1)
        - 4 * frame
    )
    return float(np.var(lap))


def extract_frame_features(frame: np.ndarray) -> dict:
    """
    Extract single-vector fingerprint for a simulation frame.
    """
    return {
        "mean": float(np.mean(frame)),
        "max": float(np.max(frame)),
        "variance": float(np.var(frame)),
        "entropy": compute_spatial_entropy(frame),
        "roughness": compute_roughness(frame),
    }
