"""
Numerical Operators (Laplacian, etc.).
"""

import numpy as np
from typing import Literal


def laplacian_periodic(Z: np.ndarray, inv_dx2: float) -> np.ndarray:
    """
    5-point stencil Laplacian with Periodic BCs.
    Vectorized implementation.
    """
    return (
        np.roll(Z, 1, 0)
        + np.roll(Z, -1, 0)
        + np.roll(Z, 1, 1)
        + np.roll(Z, -1, 1)
        - 4.0 * Z
    ) * inv_dx2


def laplacian_neumann(Z: np.ndarray, inv_dx2: float) -> np.ndarray:
    """
    5-point stencil Laplacian with Neumann (Reflect) BCs.
    """
    # Pad with reflection to handle boundaries
    Zp = np.pad(Z, 1, mode="reflect")
    # Compute Laplacian on the interior using padded neighbors
    return (
        Zp[:-2, 1:-1] + Zp[2:, 1:-1] + Zp[1:-1, :-2] + Zp[1:-1, 2:] - 4.0 * Z
    ) * inv_dx2


def compute_laplacian(
    Z: np.ndarray, dx: float, bc: Literal["periodic", "neumann"]
) -> np.ndarray:
    """Dispatch method for Laplacian."""
    inv_dx2 = 1.0 / (dx * dx)
    if bc == "periodic":
        return laplacian_periodic(Z, inv_dx2)
    elif bc == "neumann":
        return laplacian_neumann(Z, inv_dx2)
    else:
        raise ValueError(f"Unknown boundary condition: {bc}")
