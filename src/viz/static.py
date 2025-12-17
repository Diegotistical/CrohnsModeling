"""
Static Plotting Module.

Responsibility:
    - Generate high-resolution snapshots of simulation state.
    - Save figures to disk with publication-ready settings.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
from pathlib import Path

from src.viz.utils import CMAP_U, CMAP_V, setup_figure, get_normalization_limit


def plot_snapshot(
    U: np.ndarray,
    V: np.ndarray,
    step_i: int,
    dt: float = 1.0,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """
    Render a side-by-side heatmap of U (Substrate) and V (Inflammation).

    Args:
        U, V: 2D arrays of shape (Nx, Ny).
        step_i: Current simulation step index.
        dt: Time step size (for title timestamp).
        save_path: If provided, saves the figure to this path.
        show: Whether to call plt.show() (disable for batch processing).
    """
    fig, axes = setup_figure(n_panels=2, figsize=(10, 4))
    time_sim = step_i * dt

    # Plot Substrate U
    im0 = axes[0].imshow(U.T, origin="lower", cmap=CMAP_U, vmin=0, vmax=1.0)
    axes[0].set_title(f"Substrate $U$ (t={time_sim:.2f})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot Inflammation V
    im1 = axes[1].imshow(V.T, origin="lower", cmap=CMAP_V, vmin=0, vmax=1.0)
    axes[1].set_title(f"Inflammation $V$ (t={time_sim:.2f})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        # Ensure parent directory exists
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
