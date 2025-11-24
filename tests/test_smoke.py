import numpy as np
from src.solver.reaction_diffusion import simulate_explicit, default_params

def test_smoke():
nx = 32
dx = 1.0/(nx-1)
x = np.linspace(0,1,nx)
u0 = 0.01*np.ones(nx)
p = default_params()
u = simulate_explicit(u0, p, dx, dt=1e-4, nsteps=2)
assert u.shape == u0.shape
