"""2D finite-difference PDE solver using PyTorch (GPU-aware).

Provides:

laplacian(u): 2D Laplacian using torch.roll (periodic) or Neumann via padding

simulate_2d(u0, params, dx, dt, nsteps, device): explicit diffusion + reaction demo

Notes:

This is an explicit solver; for stiff problems use implicit / IMEX.

Designed to be simple, readable, and fast on GPU.
"""
import torch
import torch.nn.functional as F

def laplacian(u, dx, periodic=True):
# u shape: (batch, channel, nx, ny)
if periodic:
up = torch.roll(u, shifts=-1, dims=2)
down = torch.roll(u, shifts=1, dims=2)
left = torch.roll(u, shifts=-1, dims=3)
right = torch.roll(u, shifts=1, dims=3)
lap = (up + down + left + right - 4u) / (dxdx)
return lap
else:
# Neumann (zero-flux) via reflection padding
u_padded = F.pad(u, (1,1,1,1), mode='replicate') # pad left,right,top,bottom
lap = (u_padded[:,:,2:,1:-1] + u_padded[:,:,0:-2,1:-1] + u_padded[:,:,1:-1,2:] + u_padded[:,:,1:-1,0:-2] - 4u) / (dxdx)
return lap

def reaction_term(u, params):
# u in [0,1] ideally
alpha = params.get('alpha', 1.0)
K = params.get('K', 1.0)
beta = params.get('beta', 0.2)
return alpha * u * (1 - u / K) - beta * u

def simulate_2d(u0, params, dx=1.0/127, dt=1e-4, nsteps=100, device='cpu', periodic=True):
"""Simulate PDE: u_t = D laplacian(u) + reaction(u) + optional S(t,x)
u0: torch.Tensor shape (B,C,Nx,Ny)
returns final u (on device)
"""
dev = torch.device(device)
u = u0.to(dev).float()
D = params.get('D', 0.1)
for i in range(nsteps):
lap = laplacian(u, dx, periodic=periodic)
r = reaction_term(u, params)
u = u + dt * (D * lap + r)
# clamp to physical range to avoid numerical blow-up in explicit runs
u = torch.clamp(u, -10.0, 10.0)
return u

if name == 'main':
# quick local demo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
nx = ny = 128
dx = 1.0/(nx-1)
u0 = torch.zeros((1,1,nx,ny))
u0[:,:,nx//2-4:nx//2+4, ny//2-4:ny//2+4] = 1.0
params = dict(D=0.1, alpha=0.5, beta=0.2)
u_final = simulate_2d(u0, params, dx=dx, dt=1e-4, nsteps=200, device=device)
print('demo done; device=', device, 'mean=', u_final.mean().item())
