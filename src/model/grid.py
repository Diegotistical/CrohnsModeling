"""
Spatial Grid Infrastructure.
"""

import numpy as np
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Grid2D:
    """
    Immutable definition of the 2D spatial domain.
    """

    N: int  # Grid resolution (N x N)
    L: float  # Physical length of domain

    @property
    def dx(self) -> float:
        return self.L / self.N

    @property
    def shape(self) -> tuple[int, int]:
        return (self.N, self.N)

    @cached_property
    def spectral_eigenvalues(self) -> np.ndarray:
        """
        Compute eigenvalues for the discrete Laplacian on a periodic grid.
        Used by IMEX/Spectral solvers.
        Returns array corresponding to np.fft.rfft2 format.
        """
        kx = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        ky = 2.0 * np.pi * np.fft.rfftfreq(self.N, d=self.dx)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        return -(KX**2 + KY**2)
