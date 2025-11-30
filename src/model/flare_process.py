"""
Production-grade Reaction-Diffusion solver (Gray-Scott style) featuring:
 - IMEX spectral solver (unconditionally stable diffusion via FFT)
 - Adaptive Explicit solver (auto-substepping based on CFL condition)
 - Numba/SciPy acceleration for finite differences
 - Robust caching and strictly typed interfaces
 - Progress logging and metadata management

Usage:
    from reaction_diffusion import ReactionDiffusion
    
    sim = ReactionDiffusion(
        N=256, 
        bc='periodic', 
        flare_source={'rate': 0.01, 'intensity_mean': 0.5}
    )
    sim.run(dt=1.0, nsteps=100, method='imex') 
    # Note: dt=1.0 is large, but IMEX handles diffusion. 
    # Explicit would auto-substep this into stable increments.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Literal, Any, Union, Dict, Protocol, runtime_checkable
import warnings
import argparse
import json
import os
import sys
import time

# --- Optional Dependencies ---

# 1. Progress Bar
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# 2. FlareProcess model
try:
    from src.model.flare_process import FlareProcess
except ImportError:
    try:
        from flare_process import FlareProcess
    except ImportError:
        FlareProcess = None

# 3. Numba (JIT)
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# 4. SciPy (Convolution)
try:
    from scipy.ndimage import convolve
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


# ---------------------------
# Protocols / Interfaces
# ---------------------------

@runtime_checkable
class FlareSourceProtocol(Protocol):
    """Protocol ensuring the flare source implements the required API."""
    def apply_flare(self, V: np.ndarray) -> None:
        """Apply inflammation logic directly to the V field in-place."""
        ...


# ---------------------------
# Numerical Kernels
# ---------------------------

def _make_rfft_laplacian_eig(nx: int, ny: int, Lx: float, Ly: float) -> np.ndarray:
    """
    Compute eigenvalues of the discrete Laplacian for a periodic grid.
    Returns array compatible with rfft2 (shape: nx, ny//2 + 1).
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2.0 * np.pi * np.fft.rfftfreq(ny, d=Ly / ny)
    KX = kx[:, None]
    KY = ky[None, :]
    return -(KX ** 2 + KY ** 2)

# --- Laplacian Implementations ---

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _laplacian_periodic_numba(Z: np.ndarray) -> np.ndarray:
        nx, ny = Z.shape
        out = np.empty_like(Z)
        for i in range(nx):
            ip = (i + 1) % nx
            im = (i - 1) % nx
            for j in range(ny):
                jp = (j + 1) % ny
                jm = (j - 1) % ny
                out[i, j] = Z[im, j] + Z[ip, j] + Z[i, jm] + Z[i, jp] - 4.0 * Z[i, j]
        return out

def _laplacian_scipy(Z: np.ndarray, mode: str = 'wrap') -> np.ndarray:
    """5-point stencil via SciPy convolution."""
    # Define kernel once? In C-based scipy, passing new array is cheap, 
    # but could be module-level constant.
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=Z.dtype)
    return convolve(Z, kernel, mode=mode)

def _laplacian_numpy_roll(Z: np.ndarray) -> np.ndarray:
    """Naive fallback. Slow."""
    return (np.roll(Z, 1, 0) + np.roll(Z, -1, 0) + 
            np.roll(Z, 1, 1) + np.roll(Z, -1, 1) - 4.0 * Z)


# ---------------------------
# Main Solver
# ---------------------------

Boundary = Literal["periodic", "neumann"]

class ReactionDiffusion:
    """
    Robust 2D Reaction-Diffusion solver (Gray-Scott).
    
    Attributes:
        N (int): Grid dimension (square).
        L (float): Domain physical size.
        Du, Dv, F, k (float): Model parameters.
        bc (str): Boundary condition.
        flare_source (FlareSourceProtocol): Optional stochastic forcing.
    """

    def __init__(
        self,
        N: int = 200,
        L: float = 1.0,
        Du: float = 0.16,
        Dv: float = 0.08,
        F: float = 0.0545,
        k: float = 0.062,
        bc: Boundary = "periodic",
        seed: Optional[int] = None,
        noise: float = 0.05,
        flare_source: Union[FlareSourceProtocol, Dict[str, Any], None] = None,
    ):
        self.N = int(N)
        self.L = float(L)
        self.Du = float(Du)
        self.Dv = float(Dv)
        self.F = float(F)
        self.k = float(k)
        
        if bc not in ("periodic", "neumann"):
            raise ValueError(f"Invalid bc '{bc}'. Must be 'periodic' or 'neumann'.")
        self.bc = bc

        self.dx = self.L / self.N
        self.dy = self.L / self.N

        # State Initialization
        self.rng = np.random.default_rng(seed)
        self.U = np.ones((self.N, self.N), dtype=float)
        self.V = np.zeros((self.N, self.N), dtype=float)

        # Standard Gray-Scott Initialization (Center Square)
        r = max(6, int(0.08 * self.N))
        s, e = self.N // 2 - r, self.N // 2 + r
        self.U[s:e, s:e] = 0.50
        self.V[s:e, s:e] = 0.25

        # Add Noise
        if noise > 0:
            self.U += self.rng.random(self.U.shape) * float(noise)
            self.V += self.rng.random(self.V.shape) * float(noise)
        
        # Flare Setup
        if isinstance(flare_source, dict):
            if FlareProcess is None:
                raise ImportError("FlareProcess module not found; cannot instantiate from dict.")
            self.flare_source = FlareProcess(**flare_source, seed=seed)
        else:
            self.flare_source = flare_source

        # Interface Check
        if self.flare_source is not None:
            if not isinstance(self.flare_source, FlareSourceProtocol):
                # We check via isinstance against Protocol to be safe, 
                # but fallback to hasattr if static check fails at runtime? 
                # Ideally, just trust Protocol check or explicit hasattr.
                if not hasattr(self.flare_source, 'apply_flare'):
                    raise TypeError("flare_source must implement 'apply_flare(V)'.")

        # Optimization / Caching
        self._eig: Optional[np.ndarray] = None
        # Cache key: (dt, Du, Dv) -> (denomU, denomV)
        self._imex_cache: Dict[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray]] = {}
        
        if self.bc == "periodic":
            self._eig = _make_rfft_laplacian_eig(self.N, self.N, self.L, self.L)

    # ---------------------------
    # Core Logic
    # ---------------------------

    def _reaction(self, U: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute reaction rates (Gray-Scott)."""
        # Reaction: 2V + U -> 3V (P = U*V^2)
        UVV = U * (V * V)
        d_u = -UVV + self.F * (1.0 - U)
        d_v = UVV - (self.F + self.k) * V
        return d_u, d_v

    def _compute_laplacian(self, Z: np.ndarray) -> np.ndarray:
        """Dispatches best available Laplacian implementation."""
        if self.bc == "periodic":
            if _NUMBA_AVAILABLE:
                return _laplacian_periodic_numba(Z)
            elif _SCIPY_AVAILABLE:
                return _laplacian_scipy(Z, mode='wrap')
            return _laplacian_numpy_roll(Z)
        
        # Neumann
        if _SCIPY_AVAILABLE:
            return _laplacian_scipy(Z, mode='reflect')
        
        # NumPy Pad Fallback (Reflect)
        Z_pad = np.pad(Z, 1, mode="reflect")
        return (Z_pad[:-2, 1:-1] + Z_pad[2:, 1:-1] + 
                Z_pad[1:-1, :-2] + Z_pad[1:-1, 2:] - 4.0 * Z_pad[1:-1, 1:-1])

    def _apply_bounds(self):
        """Constrains state to physical bounds [0, 1]."""
        np.clip(self.U, 0.0, 1.0, out=self.U)
        np.clip(self.V, 0.0, 1.0, out=self.V)

    # ---------------------------
    # Explicit Solver (Adaptive)
    # ---------------------------

    def _explicit_single_substep(self, dt: float, inv_dx2: float) -> None:
        """Performs a single explicit Euler step."""
        Lu = self._compute_laplacian(self.U) * inv_dx2
        Lv = self._compute_laplacian(self.V) * inv_dx2
        
        ru, rv = self._reaction(self.U, self.V)
        
        self.U += dt * (self.Du * Lu + ru)
        self.V += dt * (self.Dv * Lv + rv)
        self._apply_bounds()

    def _explicit_step_adaptive(self, dt: float) -> None:
        """
        Adaptive Explicit Step.
        Checks CFL condition. If requested dt is unstable, subdivides into
        smaller stable steps automatically.
        """
        limit = self.explicit_cfl_limit()
        # Safety factor 0.9 to be conservative
        safe_limit = 0.9 * limit
        
        if dt <= safe_limit:
            self._explicit_single_substep(dt, 1.0/(self.dx**2))
        else:
            # Sub-stepping required
            n_sub = int(np.ceil(dt / safe_limit))
            dt_sub = dt / n_sub
            inv_dx2 = 1.0 / (self.dx ** 2)
            
            # We assume reaction timescale is generally slower than diffusion limit,
            # so diffusion CFL drives the sub-stepping.
            for _ in range(n_sub):
                self._explicit_single_substep(dt_sub, inv_dx2)

    # ---------------------------
    # IMEX Solver
    # ---------------------------

    def _imex_step_fft(self, dt: float) -> None:
        """
        Semi-Implicit Method (IMEX).
        Diffusion handled implicitly in spectral domain (unconditionally stable).
        Reaction handled explicitly.
        """
        if self._eig is None:
            raise RuntimeError("IMEX requires periodic BCs and precomputed eigenvalues.")

        # 1. Explicit Reaction
        ru, rv = self._reaction(self.U, self.V)
        
        # Intermediate state (U*, V*)
        rhsU = self.U + dt * ru
        rhsV = self.V + dt * rv

        # 2. Implicit Diffusion Solve
        rhsU_hat = np.fft.rfft2(rhsU)
        rhsV_hat = np.fft.rfft2(rhsV)

        # Retrieve/Compute Denominators (Cached)
        key = (dt, self.Du, self.Dv)
        if key in self._imex_cache:
            denomU, denomV = self._imex_cache[key]
        else:
            denomU = 1.0 - dt * self.Du * self._eig
            denomV = 1.0 - dt * self.Dv * self._eig
            self._imex_cache[key] = (denomU, denomV)

        self.U = np.fft.irfft2(rhsU_hat / denomU, s=(self.N, self.N))
        self.V = np.fft.irfft2(rhsV_hat / denomV, s=(self.N, self.N))
        
        self._apply_bounds()

    # ---------------------------
    # Public Interface
    # ---------------------------

    def step(self, dt: float = 1.0, method: Literal["imex", "explicit"] = "imex") -> None:
        """Advance simulation by one global timestep."""
        
        if method == "imex":
            if self.bc != "periodic":
                warnings.warn("IMEX requires periodic BCs. Switching to Explicit.", RuntimeWarning)
                self._explicit_step_adaptive(dt)
            else:
                self._imex_step_fft(dt)
        elif method == "explicit":
            self._explicit_step_adaptive(dt)
        else:
            raise ValueError(f"Unknown method: {method}")

        if self.flare_source:
            self.flare_source.apply_flare(self.V)

    def run(
        self,
        dt: float,
        nsteps: int,
        method: Literal["imex", "explicit"] = "imex",
        callback: Optional[callable] = None,
        use_tqdm: bool = True
    ) -> None:
        """
        Run simulation loop.
        
        Args:
            dt: Time step size.
            nsteps: Number of steps.
            method: Solver method.
            callback: fn(step_i, U, V) -> None.
            use_tqdm: Whether to show progress bar.
        """
        iterator = range(nsteps)
        if _TQDM_AVAILABLE and use_tqdm:
            iterator = tqdm(iterator, desc="Simulating", unit="step")
        
        for i in iterator:
            self.step(dt, method)
            if callback:
                callback(i, self.U, self.V)

    def explicit_cfl_limit(self) -> float:
        """Calculate max stable dt for explicit diffusion: dx^2 / (4 * Dmax)."""
        return (self.dx ** 2) / (4.0 * max(self.Du, self.Dv))

    def to_dict(self):
        return {
            "N": self.N, "L": self.L, 
            "Du": self.Du, "Dv": self.Dv, 
            "F": self.F, "k": self.k, 
            "bc": self.bc
        }

# ---------------------------
# CLI & Entry Point
# ---------------------------

def _run_cli():
    parser = argparse.ArgumentParser(description="Reaction-Diffusion Solver CLI")
    parser.add_argument("--N", type=int, default=200)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--Du", type=float, default=0.16)
    parser.add_argument("--Dv", type=float, default=0.08)
    parser.add_argument("--F", type=float, default=0.0545)
    parser.add_argument("--k", type=float, default=0.062)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--method", default="imex", choices=["imex", "explicit"])
    parser.add_argument("--bc", default="periodic", choices=["periodic", "neumann"])
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flare-rate", type=float, default=0.0)
    
    args = parser.parse_args()

    # Configure Flare
    flare = None
    if args.flare_rate > 0:
        if FlareProcess:
            flare = FlareProcess(rate=args.flare_rate, seed=args.seed)
        else:
            print("Warning: FlareProcess module not found. Ignoring flares.")

    sim = ReactionDiffusion(
        N=args.N, L=args.L, 
        Du=args.Du, Dv=args.Dv, F=args.F, k=args.k, 
        bc=args.bc, seed=args.seed, flare_source=flare
    )

    os.makedirs(args.outdir, exist_ok=True)
    
    # Callback to save frames
    def cb(i, U, V):
        if i % args.save_every == 0:
            # Placeholder for saving frames if needed, 
            # or could accumulate in list.
            pass

    print(f"Starting simulation: {args.method}, dt={args.dt}, steps={args.steps}")
    sim.run(args.dt, args.steps, method=args.method, callback=cb)
    
    # Save final
    np.save(os.path.join(args.outdir, "V_final.npy"), sim.V)
    with open(os.path.join(args.outdir, "meta.json"), "w") as f:
        json.dump(sim.to_dict(), f)
    print("Simulation complete.")

if __name__ == "__main__":
    _run_cli()

# ---------------------------
# Test Helpers (for Unit Tests)
# ---------------------------

def default_params():
    nx = 64
    dx = 2*np.pi/nx
    D = 1.0
    dt_cfl = (dx**2)/(4*D)
    # Return unstable dt to verify sub-stepping/stability logic handles it
    return {"D": D, "dt": dt_cfl * 2.0, "dx": dx, "dy": dx}

def simulate_explicit(u0, D, dt, dx, dy, steps, noise_scale=1e-12, seed=42):
    """
    Standalone explicit integrator for testing numerical properties 
    in isolation from class overhead.
    """
    u = u0.astype(float).copy()
    if noise_scale > 0:
        rng = np.random.default_rng(seed)
        u += rng.normal(0, noise_scale, u.shape)
    
    inv_d2 = 1.0 / (dx**2)
    
    for _ in range(int(steps)):
        lap = (np.roll(u, -1, 0) + np.roll(u, 1, 0) - 2*u) * inv_d2 + \
              (np.roll(u, -1, 1) + np.roll(u, 1, 1) - 2*u) * inv_d2
        u += dt * D * lap
        
    return u