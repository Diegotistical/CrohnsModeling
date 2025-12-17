import numpy as np
from src.solver.reaction_diffusion import ReactionDiffusion


def test_imex_runs_and_shapes():
    sim = ReactionDiffusion(
        N=64, L=1.0, Du=0.1, Dv=0.05, F=0.04, k=0.06, bc="periodic", seed=42, noise=0.0
    )
    dt = 1e-2
    sim.step(dt=dt, method="imex")
    U, V = sim.state()
    assert U.shape == (64, 64)
    assert V.shape == (64, 64)
    assert not np.any(np.isnan(U))
    assert not np.any(np.isnan(V))
