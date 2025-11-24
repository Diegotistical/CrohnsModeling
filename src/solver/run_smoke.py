"""Run a tiny smoke test to validate installation."""
import numpy as np
from src.solver.reaction_diffusion import simulate_explicit, default_params

def main():
nx = 128
L = 1.0
dx = L / (nx-1)
x = np.linspace(0, L, nx)
u0 = 0.01 * np.exp(-100*(x-0.5)**2)
params = default_params()
u_final = simulate_explicit(u0, params, dx, dt=1e-4, nsteps=10)
print('smoke run done, u_mean=', u_final.mean())

if name == 'main':
main()
