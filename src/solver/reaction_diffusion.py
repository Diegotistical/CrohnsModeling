"""1D reaction-diffusion solver skeleton.

Fill this with a tested FD IMEX or Crank-Nicolson implementation.
"""
import numpy as np

def default_params():
return dict(D=0.1, alpha=1.0, K=1.0, beta=0.2)

def reaction(u, v, params):
# placeholder reaction term: logistic - damping
return params['alpha'] * u * (1 - u/params['K']) - params['beta']*u

def simulate_explicit(u0, params, dx, dt, nsteps):
# VERY simple explicit scheme (replace with IMEX / CN for stability)
u = u0.copy()
nx = u.size
for t in range(nsteps):
lap = np.zeros_like(u)
lap[1:-1] = (u[2:] - 2*u[1:-1] + u[0:-2]) / dx**2
u = u + dt * (params['D'] * lap + reaction(u, None, params))
return u
