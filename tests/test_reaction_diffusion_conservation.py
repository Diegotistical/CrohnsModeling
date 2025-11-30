import numpy as np
from src.solver.reaction_diffusion import simulate_explicit, default_params


def test_mass_conservation():
    p = default_params()
    nx, ny = 64, 64

    rng = np.random.default_rng(0)
    u0 = rng.random((nx, ny))

    mass_initial = np.sum(u0)
    u_final = simulate_explicit(
        u0=u0,
        D=p["D"],
        dt=p["dt"],
        dx=p["dx"],
        dy=p["dy"],
        steps=20,
    )
    mass_final = np.sum(u_final)

    # Explicit diffusion with periodic BC should preserve mass very tightly
    assert abs(mass_final - mass_initial) < 1e-6
