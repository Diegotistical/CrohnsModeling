
Technical Summary â€” CrohnsModeling
Author: [Your Name]
Date: 2025-11-24

Problem statement
We model chronic localized intestinal inflammation (Crohn's-scale lesions) at the tissue level using reactionâ€“diffusion equations with stochastic flare forcing. The model aims to study conditions under which transient flares lead to sustained chronic inflammation, how spatial coupling spreads lesions, and how parameter uncertainty affects outcomes.

Core model (1D, minimal)
State variable 
ð‘¢
(
ð‘¥
,
ð‘¡
)
u(x,t) âˆˆ [0,1] = inflammation intensity.

âˆ‚
ð‘¡
ð‘¢
=
ð·
â€‰
âˆ‚
ð‘¥
ð‘¥
ð‘¢
+
ð›¼
ð‘¢
(
1
âˆ’
ð‘¢
/
ð¾
)
âˆ’
ð›½
ð‘¢
+
ð‘†
(
ð‘¥
,
ð‘¡
)
âˆ‚ 
t
â€‹
 u=Dâˆ‚ 
xx
â€‹
 u+Î±u(1âˆ’u/K)âˆ’Î²u+S(x,t)
ð·
D: diffusion coefficient (cytokine/immune cell migration)

ð›¼
Î±: activation/amplification rate

ð¾
K: saturation level

ð›½
Î²: healing/damping

ð‘†
(
ð‘¥
,
ð‘¡
)
S(x,t): stochastic flare process (Poisson impulses in time, spatial kernel in x)

Numerical methods
1D solver: finite differences in space, IMEX time-stepping (implicit linear diffusion, explicit nonlinear reaction).

2D solver: finite differences implemented in PyTorch for GPU acceleration; explicit/Strang-splitting options.

SPDE handling: spatially-correlated noise (exponential kernel) or compartmental Langevin approximation; Eulerâ€“Maruyama for SDE layer.

Stability & validation: Von Neumann-style checks, grid-refinement convergence tests, mass-conservation checks for zero-reaction limit.

Inference & UQ
Parameter discovery via PINNs (PyTorch) from sparse synthetic observations.

Bayesian calibration (NumPyro/PyMC3) with surrogate model (small NN) to accelerate posterior sampling.

Posterior predictive checks and identifiability discussion.

Limitations & ethics
Simplified tissue model â€” omits explicit cell types and detailed microbiome dynamics.

Not clinical â€” no patient-level claims. Intended as proof-of-concept computational research.

Key results (to be filled after experiments)
Example: parameter sweep suggests chronic regime when 
ð›¼
/
ð›½
>
ð‘‹
Î±/Î²>X and D < Y.

Example: stochastic flares with mean rate Î» trigger persistent lesions in Z% of runs.

