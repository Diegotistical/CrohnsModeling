import numpy as np
from src.solver.reaction_diffusion import simulate_explicit, default_params


def test_numerical_stability_with_smaller_dt():
    p = default_params()
    nx, ny = 64, 64

    # Sinusoidal perturbation
    x = np.linspace(0, 2 * np.pi, nx)
    y = np.linspace(0, 2 * np.pi, ny)
    X, Y = np.meshgrid(x, y)
    u0 = np.sin(X) * np.sin(Y)

    # Large dt
    u_large_dt = simulate_explicit(
        u0=u0,
        D=p["D"],
        dt=p["dt"] * 2.0,  # more unstable
        dx=p["dx"],
        dy=p["dy"],
        steps=10,
    )

    # Smaller dt
    u_small_dt = simulate_explicit(
        u0=u0,
        D=p["D"],
        dt=p["dt"] * 0.25,  # more stable
        dx=p["dx"],
        dy=p["dy"],
        steps=40,  # adjust so physical time matches
    )

    # Energy should decay more consistently for small dt
    energy_large_dt = np.sum(u_large_dt**2)
    energy_small_dt = np.sum(u_small_dt**2)

    # Explicit scheme: smaller dt gives smoother, lower-energy solution
    assert energy_small_dt < energy_large_dt
