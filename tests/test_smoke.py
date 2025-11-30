import numpy as np
from src.solver.reaction_diffusion import simulate_explicit, default_params


def test_smoke_runs():
    p = default_params()
    nx, ny = 32, 32

    u0 = np.ones((nx, ny))
    u_final = simulate_explicit(
        u0=u0,
        D=p["D"],
        dt=p["dt"],
        dx=p["dx"],
        dy=p["dy"],
        steps=5,
    )

    assert u_final.shape == (nx, ny)
    assert np.isfinite(u_final).all()
