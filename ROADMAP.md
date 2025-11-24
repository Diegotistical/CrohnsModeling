
ROADMAP CrohnsModeling
Goal: produce a compact, reproducible, mathematically rigorous project that showcases skills admission committees care about (PDEs/SPDEs, numerics, ML, uncertainty quantification, software engineering).

Milestones (6â€“8 weeks realistic schedule)
Week 0: Repo skeleton, smoke tests, README, CI (done).

Week 1: 1D deterministic PDE solver (IMEX / Crankâ€“Nicolson). Unit tests for convergence.

Week 2: Add stochastic flares (compartment SDE + SPDE approximation). Ensemble diagnostics and plots.

Week 3: Agent-based coupling (Mesa) to show emergent patchiness and lesion formation.

Week 4: PINN template to infer 1â€“2 parameters from sparse observations (PyTorch).

Week 5: UQ pipeline (NumPyro/NumPyro or PyMC3), surrogate model for fast MCMC.

Week 6: 2D GPU-accelerated solver, parameter sweeps, nice animations; write technical summary and prepare PDF.

Optional: brief experiment with inflammationâ†’tumor initiation (mechanistic term) and sensitivity analysis.

Deliverables for MSFE application
1â€“2 page technical summary PDF explaining math, numerics, and findings (paper/technical_summary.md -> PDF)

3 polished notebooks: PDE baseline, SPDE + UQ, 2D GPU demo (animated)

Test suite + CI to show reproducibility

Short README bullet: "What this is NOT" (no clinical claims)

A concise results folder with 3 figures and one animation (.mp4 or gif)

How to present in your application
One-line elevator pitch in CV: "CrohnsModeling â€” mechanistic PDE/SPDE modeling of intestinal inflammation; code + technical summary; emphasis on numerics and UQ."

In SOP/essay: 1 paragraph about mathematical contributions (convergence tests, stability analysis), 1 paragraph about engineering (CI, Docker), 1 paragraph about future work.

