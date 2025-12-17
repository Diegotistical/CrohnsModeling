"""
Animation Engine.

Responsibility:
    - Convert simulation history into MP4/GIF.
    - Use efficient blitting for high-performance rendering.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Tuple, Optional, Union
import logging

from src.viz.utils import CMAP_U, CMAP_V

logger = logging.getLogger(__name__)


def save_animation(
    history: List[Tuple[np.ndarray, np.ndarray]],
    filename: str = "simulation.mp4",
    fps: int = 30,
    dpi: int = 150,
) -> None:
    """
    Compiles a history list into a video file.

    Args:
        history: List of (U, V) tuples. Arrays should be copied, not references.
        filename: Output path. Extension determines writer (.mp4 -> ffmpeg).
        fps: Frames per second.
        dpi: Resolution of output video.
    """
    if not history:
        logger.warning("Animation history is empty. Skipping video generation.")
        return

    # Create Figure once
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)

    # Initial Frame data
    U0, V0 = history[0]

    # Initialize Artists (Images)
    im_u = axes[0].imshow(
        U0.T, origin="lower", cmap=CMAP_U, vmin=0, vmax=1.0, animated=True
    )
    axes[0].set_title("Substrate U")

    im_v = axes[1].imshow(
        V0.T, origin="lower", cmap=CMAP_V, vmin=0, vmax=1.0, animated=True
    )
    axes[1].set_title("Inflammation V")

    # Remove ticks for cleaner video
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    def update(frame_idx):
        """Update function for animation loop."""
        U, V = history[frame_idx]
        im_u.set_data(U.T)
        im_v.set_data(V.T)
        return im_u, im_v

    # Create Animation Object
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(history),
        blit=True,  # Critical for performance
        interval=1000 / fps,
    )

    # Determine Writer
    writer = "ffmpeg" if filename.endswith(".mp4") else "pillow"

    logger.info(f"Rendering animation to {filename} ({len(history)} frames)...")
    try:
        ani.save(filename, writer=writer, fps=fps, dpi=dpi)
        logger.info("Animation saved successfully.")
    except Exception as e:
        logger.error(
            f"Failed to save animation. Ensure ffmpeg is installed if using mp4.\nError: {e}"
        )
    finally:
        plt.close(fig)
