"""
Pipeline: GIF Version (Fixed Initialization).
"""
import sys
import os

# --- PATH FIX ---
sys.path.append(os.getcwd())

import argparse
import logging
import numpy as np
from pathlib import Path
from time import perf_counter

# --- Absolute Imports ---
from src.model import Grid2D, RDParameters, ReactionDiffusionModel, StochasticFlare, FlareConfig
from src.solver import ImexSolver, Monitor, save_checkpoint
from src.viz import save_animation, plot_snapshot
from src.inference import extract_frame_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=1.0)  # Larger dt is safe for IMEX
    parser.add_argument("--outdir", type=str, default="results/gif_run_fixed")
    args = parser.parse_args()

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Model - Using "Robust Spot" Parameters
    logger.info("Initializing Physics...")
    grid = Grid2D(N=args.N, L=2.5) # Slightly larger physical size
    
    # F=0.055, k=0.062 is the classic "Coral/Spot" pattern
    params = RDParameters(Du=2e-5, Dv=1e-5, F=0.055, k=0.062)
    model = ReactionDiffusionModel(grid, params, bc='periodic', seed=42)

    # --- THE FIX: MANUAL KICKSTART ---
    # We force a square in the center to be V=0.25 to start the reaction
    logger.info("Injecting initial inflammation bolus...")
    mid = args.N // 2
    r = 10
    model.U[mid-r:mid+r, mid-r:mid+r] = 0.50
    model.V[mid-r:mid+r, mid-r:mid+r] = 0.25
    # ---------------------------------

    # 2. Solver (IMEX)
    solver = ImexSolver(model)
    history = []
    
    # 3. Run
    logger.info(f"Running {args.steps} steps...")
    t_start = perf_counter()
    
    for step in range(args.steps):
        solver.step(args.dt)
        
        # Log health every 100 steps
        if step % 100 == 0:
            max_v = np.max(model.V)
            logger.info(f"Step {step}: Max Inflammation = {max_v:.4f}")

        # Save frame every 10 steps
        if step % 10 == 0:
            history.append((model.U.copy(), model.V.copy()))
            
    logger.info(f"Done in {perf_counter()-t_start:.2f}s")

    # 4. Viz
    vid_path = out_path / "simulation.gif"
    logger.info(f"Saving GIF to {vid_path}...")
    save_animation(history, filename=str(vid_path), fps=20)
    
    logger.info("Pipeline Complete. Check results/gif_run_fixed/simulation.gif")

if __name__ == "__main__":
    main()