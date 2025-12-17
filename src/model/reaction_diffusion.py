"""
Gray-Scott Reaction-Diffusion Physics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Literal, Optional

from src.model.grid import Grid2D
from src.model.operators import compute_laplacian

BoundaryType = Literal["periodic", "neumann"]


@dataclass(frozen=True)
class RDParameters:
    """Immutable physics parameters."""

    Du: float = 0.16  # Diffusion rate of U
    Dv: float = 0.08  # Diffusion rate of V
    F: float = 0.0545  # Feed rate
    k: float = 0.062  # Kill rate


class ReactionDiffusionModel:
    """
    Manages the state (U, V) and computes rates of change.
    """

    def __init__(
        self,
        grid: Grid2D,
        params: RDParameters,
        bc: BoundaryType = "periodic",
        seed: Optional[int] = None,
    ):
        self.grid = grid
        self.params = params
        self.bc = bc
        self.rng = np.random.default_rng(seed)

        # Initialize State
        self.U = np.ones(grid.shape, dtype=float)
        self.V = np.zeros(grid.shape, dtype=float)

        # Initial Perturbation (Center Square)
        self._initialize_perturbation()

        # Add minimal noise to break symmetry
        self.U += self.rng.normal(0, 0.01, grid.shape)
        self.V += self.rng.normal(0, 0.01, grid.shape)
        self._clip_state()

    def _initialize_perturbation(self):
        """Standard Gray-Scott center square perturbation."""
        N = self.grid.N
        r = max(6, int(0.08 * N))
        s, e = N // 2 - r, N // 2 + r
        self.U[s:e, s:e] = 0.50
        self.V[s:e, s:e] = 0.25

    def _clip_state(self):
        """Enforce physical bounds [0, 1]."""
        np.clip(self.U, 0.0, 1.0, out=self.U)
        np.clip(self.V, 0.0, 1.0, out=self.V)

    # --- Protocol Implementation ---

    @property
    def eigenvalues(self) -> Optional[np.ndarray]:
        """Expose grid eigenvalues for IMEX solvers."""
        if self.bc == "periodic":
            return self.grid.spectral_eigenvalues
        return None

    def diffusion_coefficients(self) -> Tuple[float, float]:
        """Return (Du, Dv)."""
        return self.params.Du, self.params.Dv

    def reaction(self, U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gray-Scott Reaction:
        dU/dt = -UV^2 + F(1-U)
        dV/dt = +UV^2 - (F+k)V
        """
        uvv = U * V * V
        ru = -uvv + self.params.F * (1.0 - U)
        rv = uvv - (self.params.F + self.params.k) * V
        return ru, rv

    def diffusion(self, U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute spatial diffusion using the configured operator."""
        # Laplacian calculation
        lu = compute_laplacian(U, self.grid.dx, self.bc)
        lv = compute_laplacian(V, self.grid.dx, self.bc)

        return self.params.Du * lu, self.params.Dv * lv
