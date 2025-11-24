# CrohnsModeling

[![CI](https://img.shields.io/badge/ci-pytests-blue)](https://github.com/yourname/CrohnsModeling/actions)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#)
[![Python](https://img.shields.io/badge/python-3.11-blue)](#)

**CrohnsModeling** â€” mechanistic, reproducible models of localized intestinal inflammation using PDEs, stochastic flares, agent-based layers and modern ML (PINNs).  
**Author:** [Your Name] â€” _MSFE-focused portfolio project_.  
**Created:** 2025-11-24

## Quick pitch
This repo demonstrates quantitative modeling skills relevant for MSFE applications:
- PDE/SPDE modeling (reactionâ€“diffusion with stochastic flares)
- Numerical methods (FD, IMEX, stability analysis, convergence tests)
- Probabilistic inference & UQ (MCMC/VI, PINNs)
- Software engineering (tests, CI, Docker, notebooks)
- GPU-accelerated 2D solver (PyTorch) for fast experiments and visualization

## Repo structure
CrohnsModeling/
â”œâ”€ notebooks/
â”œâ”€ src/
â”‚ â”œâ”€ solver/
â”‚ â”œâ”€ model/
â”‚ â”œâ”€ inference/
â”‚ â””â”€ viz/
â”œâ”€ tests/
â”œâ”€ data/
â”œâ”€ results/
â”œâ”€ paper/
â””â”€ .github/workflows/ci.yml

bash
Copy code

## Install (recommended)
Use conda for the full environment:

`ash
conda env create -f environment.yml
conda activate crohnsmodeling
pip install -r requirements-extras.txt   # optional extras (mesa, numpyro, optuna)
Quickstart (smoke)
bash
Copy code
# from repo root
python -m src.solver.run_smoke
# run the GPU demo notebook
jupyter lab notebooks/06_2d_pde_gpu.ipynb
Reproducibility
environment.yml included for conda

Dockerfile included for containerized runs

Minimal CI runs the test suite (pytest -q)

Limitations & ethics
This is a mechanistic research / portfolio project, not a clinical tool. See paper/technical_summary.md for assumptions and limitations.

