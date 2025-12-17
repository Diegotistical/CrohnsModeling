"""
Visualization Utilities & Theme Engine.

Responsibility:
    - Centralize matplotlib configuration (rcParams).
    - Define consistent colormaps and normalization logic.
    - Provide layout helpers for subplot management.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Tuple, List, Optional, Union

# --- 1. Quant-Grade Style Configuration ---
# We use a clean, sans-serif style optimized for both screen and PDF export.
PLT_STYLE = {
    "font.family": "sans-serif",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.dpi": 140,  # High DPI for clear notebook rendering
    "savefig.dpi": 300,  # Print-quality export
    "axes.spines.top": False,
    "axes.spines.right": False,
    "image.cmap": "viridis",
    "image.interpolation": "nearest",  # Avoid misleading interpolation artifacts
}

plt.rcParams.update(PLT_STYLE)

# --- 2. Semantic Colormaps ---
# U: Substrate/Tissue -> Viridis (Perceptually Uniform)
# V: Inflammation/Fire -> Inferno (High contrast for 'hot' zones)
CMAP_U = "viridis"
CMAP_V = "inferno"


def get_normalization_limit(
    data: np.ndarray, auto_scale: bool = False, hard_limit: float = 1.0
) -> Tuple[float, float]:
    """
    Determine safe vmin/vmax for plotting.

    Args:
        data: The field to normalize.
        auto_scale: If True, uses min/max of current frame.
        hard_limit: If auto_scale is False, uses [0, hard_limit].
    """
    if auto_scale:
        return np.min(data), np.max(data)
    return 0.0, hard_limit


def setup_figure(
    n_panels: int = 2, figsize: Tuple[float, float] = (10, 4)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Factory for creating consistent figure layouts.

    Returns:
        fig, list_of_axes
    """
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    return fig, axes
