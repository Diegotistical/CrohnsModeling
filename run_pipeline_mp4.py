"""
Pipeline: MP4 Version (High Quality).
Requires: ffmpeg installed.
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
from src.inference import extract_frame_features, FlareDetector
from src.viz import save_animation, plot_snapshot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="MP4 Pipeline")
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--outdir", type=str, default="results/mp4_run")
    parser.add_argument("--flare-prob", type=float, default=0.02)
    args = parser.parse_args()

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Model
    logger.info("Initializing Physics Model...")
    grid = Grid2D(N=args.N, L=1.0)
    params = RDParameters(Du=0.16, Dv=0.08, F=0.04, k=0.06)
    model = ReactionDiffusionModel(grid, params, bc='periodic', seed=42)
    
    flare = None
    if args.flare_prob > 0:
        flare_conf = FlareConfig(probability=args.flare_prob, intensity=0.3)
        flare = StochasticFlare(flare_conf, shape=grid.shape, seed=101)

    # 2. Solver (IMEX for Stability)
    logger.info("Initializing Solver (IMEX)...")
    solver = ImexSolver(model, flare_source=flare)
    monitor = Monitor(abort_on_nan=True)
    history = []
    
    # 3. Run
    logger.info(f"Running {args.steps} steps...")
    t_start = perf_counter()
    for step in range(args.steps):
        solver.step(args.dt)
        monitor.check(step, model.U, model.V)
        
        if step % 5 == 0: # Higher framerate for MP4
            history.append((model.U.copy(), model.V.copy()))
            
    logger.info(f"Done in {perf_counter()-t_start:.2f}s")

    # 4. Viz
    vid_path = out_path / "simulation.mp4"
    logger.info(f"Saving MP4 to {vid_path}...")
    save_animation(history, filename=str(vid_path), fps=30)
    
    snap_path = out_path / "final_state.png"
    plot_snapshot(model.U, model.V, step_i=args.steps, dt=args.dt, save_path=snap_path, show=False)
    
    logger.info("MP4 Pipeline Complete.")

if __name__ == "__main__":
    main()