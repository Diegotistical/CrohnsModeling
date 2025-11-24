import numpy as np
import torch
from src.solver.reaction_diffusion import simulate_explicit, default_params
from src.solver import solver_2d_torch

def test_explicit_mass_conservation_zero_reaction():
# zero reaction and zero-flux boundary => mass should remain approx constant for small dt
nx = 64
L = 1.0
dx = L/(nx-1)
x = np.linspace(0,L,nx)
u0 = np.zeros(nx)
# single bump
u0[int(nx/2)-2:int(nx/2)+2] = 1.0
p = default_params()
p['alpha'] = 0.0
p['beta'] = 0.0
u_final = simulate_explicit(u0, p, dx, dt=1e-5, nsteps=10)
mass0 = u0.sum()
mass1 = u_final.sum()
assert abs(mass1 - mass0) < 1e-6

def test_2d_solver_runs_and_respects_device():
device = 'cuda' if torch.cuda.is_available() else 'cpu'
nx = 64
ny = 64
dx = 1.0/(nx-1)
u0 = torch.zeros((1,1,nx,ny))
u0[:,:,nx//2-2:nx//2+2, ny//2-2:ny//2+2] = 1.0
params = dict(D=0.1, alpha=0.0, beta=0.0)
u_final = solver_2d_torch.simulate_2d(u0.to(device), params, dx=dx, dt=1e-4, nsteps=5, device=device)
assert isinstance(u_final, torch.Tensor)
assert u_final.shape == u0.to(device).shape
