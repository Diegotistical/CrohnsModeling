"""
Time-stepping implementations (Integrators).
"""
import numpy as np
import logging
from typing import Optional, List, Callable, Any, Tuple
from src.solver.base import ModelProtocol

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

class ExplicitEulerSolver:
    """
    Standard Forward Euler integrator.
    Robust but requires small time steps (CFL limited).
    """
    def __init__(self, model: ModelProtocol, flare_source: Optional[Any] = None):
        self.model = model
        self.flare = flare_source

    def step(self, dt: float) -> None:
        # 1. Apply Flares
        if self.flare:
            self.flare.apply_flare(self.model.V)

        # 2. Physics Rates
        du_diff, dv_diff = self.model.diffusion(self.model.U, self.model.V)
        du_react, dv_react = self.model.reaction(self.model.U, self.model.V)

        # 3. Integration
        self.model.U += dt * (du_diff + du_react)
        self.model.V += dt * (dv_diff + dv_react)

        # 4. Safety Constraints
        np.clip(self.model.U, 0.0, 1.0, out=self.model.U)
        np.clip(self.model.V, 0.0, 1.0, out=self.model.V)
        
        # 5. Critical Failure Check
        if np.isnan(self.model.U.sum()):
            logger.critical("NaN detected in simulation state (Explicit Solver).")
            raise RuntimeError("NaN detected in simulation state.")

    def run(self, dt: float, nsteps: int, callbacks: Optional[List[Callable]] = None) -> None:
        iterator = range(nsteps)
        if _TQDM:
            iterator = tqdm(iterator, desc="ExplicitSolver", unit="step")
        
        for i in iterator:
            self.step(dt)
            if callbacks:
                for cb in callbacks:
                    cb(i, self.model.U, self.model.V)


class ImexSolver:
    """
    Implicit-Explicit (IMEX) Solver using Spectral Methods.
    - Diffusion: Solved Implicitly (Exact in Fourier space).
    - Reaction: Solved Explicitly.
    REQUIRES: Model must have periodic BCs and provide eigenvalues.
    """
    def __init__(self, model: ModelProtocol, flare_source: Optional[Any] = None):
        self.model = model
        self.flare = flare_source
        self._cache: dict[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray]] = {} 

    def step(self, dt: float) -> None:
        # Check requirements
        if self.model.eigenvalues is None:
            raise RuntimeError("Model does not provide eigenvalues for IMEX solver.")

        # 1. Apply Flares
        if self.flare:
            self.flare.apply_flare(self.model.V)

        # 2. Explicit Reaction
        ru, rv = self.model.reaction(self.model.U, self.model.V)

        # 3. Implicit Diffusion Solve (Spectral)
        # Equation: (I - dt*D*Lap) U_new = U_old + dt*Reaction
        
        # RHS Calculation
        rhs_u = self.model.U + dt * ru
        rhs_v = self.model.V + dt * rv
        
        # FFT (Transform to Frequency Domain)
        u_hat = np.fft.rfft2(rhs_u)
        v_hat = np.fft.rfft2(rhs_v)

        # Get Denominators (Cached for performance)
        Du, Dv = self.model.diffusion_coefficients()
        key = (dt, Du, Dv)
        
        if key not in self._cache:
            # Precompute spectral operators
            denom_u = 1.0 - dt * Du * self.model.eigenvalues
            denom_v = 1.0 - dt * Dv * self.model.eigenvalues
            self._cache[key] = (denom_u, denom_v)
        
        denom_u, denom_v = self._cache[key]

        # Update in Fourier Space
        u_hat /= denom_u
        v_hat /= denom_v

        # Inverse FFT (Back to Physical Domain)
        self.model.U[:] = np.fft.irfft2(u_hat, s=self.model.U.shape)
        self.model.V[:] = np.fft.irfft2(v_hat, s=self.model.V.shape)

        # 4. Safety
        np.clip(self.model.U, 0.0, 1.0, out=self.model.U)
        np.clip(self.model.V, 0.0, 1.0, out=self.model.V)

    def run(self, dt: float, nsteps: int, callbacks: Optional[List[Callable]] = None) -> None:
        iterator = range(nsteps)
        if _TQDM:
            iterator = tqdm(iterator, desc="ImexSolver", unit="step")
        
        for i in iterator:
            self.step(dt)
            if callbacks:
                for cb in callbacks:
                    cb(i, self.model.U, self.model.V)