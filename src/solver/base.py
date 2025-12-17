"""
Base protocols for Solvers and Models.
"""
from typing import Protocol, Tuple, Optional, Any, Union
import numpy as np

class ModelProtocol(Protocol):
    """
    Interface for a physical model.
    Must manage its own state (U, V) and provide rate calculations.
    """
    U: np.ndarray
    V: np.ndarray
    
    # Optional: For spectral solvers
    eigenvalues: Optional[np.ndarray] 

    def reaction(self, U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return reaction rates (RU, RV)."""
        ...
        
    def diffusion(self, U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return diffusion rates (Du*Lap(U), Dv*Lap(V))."""
        ...
    
    def diffusion_coefficients(self) -> Tuple[float, float]:
        """Return (Du, Dv) for implicit solvers."""
        ...

class SolverProtocol(Protocol):
    """Interface for time-integration strategies."""
    def step(self, dt: float) -> None:
        """Advance system by dt."""
        ...

    def run(self, dt: float, nsteps: int, callbacks: Optional[list] = None) -> None:
        """Execute simulation loop."""
        ...