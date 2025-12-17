"""
End-to-End Simulation Pipeline.

Usage:
    python test_pipeline.py --steps 500 --dt 0.1 --outdir ./results/experiment_01

Workflow:
    1. Model: Initialize Reaction-Diffusion physics (Gray-Scott).
    2. Solver: Run time-integration with explicit Euler.
    3. Inference: Extract entropy, roughness, and detect flares.
    4. Viz: Generate high-res snapshots and MP4 animation.
"""
import sys
import os

# --- PATH FIX: Ensures 'src' is visible even if run from weird paths ---
sys.path.append(os.getcwd())

import argparse
import logging
import numpy as np
from pathlib import Path
from time import perf_counter

# --- Absolute Imports (Production Standard) ---
from src.model import Grid2D, RDParameters, ReactionDiffusionModel, StochasticFlare, FlareConfig
from src.solver import ExplicitEulerSolver, Monitor, save_checkpoint
from src.inference import extract_frame_features, FlareDetector
from src.viz import save_animation, plot_snapshot

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="CrohnsModeling Production Pipeline")
    parser.add_argument("--N", type=int, default=128, help="Grid resolution")
    parser.add_argument("--steps", type=int, default=1000, help="Simulation steps")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step size")
    parser.add_argument("--outdir", type=str, default="results/pipeline_run", help="Output directory")
    parser.add_argument("--flare-prob", type=float, default=0.02, help="Flare probability per step")
    args = parser.parse_args()

    out_path = Path(args.outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Initialize Model (Physics Layer)
    # ---------------------------------------------------------
    logger.info("Initializing Physics Model...")
    
    grid = Grid2D(N=args.N, L=1.0)
    
    # Using Standard Gray-Scott "Splitting" Regime parameters
    params = RDParameters(Du=0.16, Dv=0.08, F=0.04, k=0.06)
    
    model = ReactionDiffusionModel(grid, params, bc='periodic', seed=42)
    
    # Configure Stochastic Flares
    flare = None
    if args.flare_prob > 0:
        flare_conf = FlareConfig(probability=args.flare_prob, intensity=0.3, radius=5)
        flare = StochasticFlare(flare_conf, shape=grid.shape, seed=101)
        logger.info(f"Flare process enabled (p={args.flare_prob}).")

    # ---------------------------------------------------------
    # 2. Initialize Solver (Engine Layer)
    # ---------------------------------------------------------
    logger.info("Initializing Solver...")
    solver = ExplicitEulerSolver(model, flare_source=flare)
    monitor = Monitor(abort_on_nan=True)
    
    # Data containers for Viz/Inference
    history = []      # For animation
    time_series_v = [] # For flare detection

    # ---------------------------------------------------------
    # 3. Execution Loop
    # ---------------------------------------------------------
    logger.info(f"Starting simulation ({args.steps} steps)...")
    t_start = perf_counter()

    for step in range(args.steps):
        # Step the physics
        solver.step(args.dt)
        
        # Monitor health
        monitor.check(step, model.U, model.V)
        
        # Record Data (Viz)
        if step % 5 == 0:  # Save every 5th frame for video
            history.append((model.U.copy(), model.V.copy()))
            
        # Record Data (Inference)
        time_series_v.append(np.max(model.V))

    duration = perf_counter() - t_start
    logger.info(f"Simulation completed in {duration:.2f}s ({args.steps / duration:.1f} steps/s)")

    # ---------------------------------------------------------
    # 4. Inference (Analysis Layer)
    # ---------------------------------------------------------
    logger.info("Running Inference Pipeline...")
    
    # A. Feature Extraction (Last Frame)
    features = extract_frame_features(model.V)
    logger.info(f"Final Frame Stats -> Entropy: {features['entropy']:.3f}, Roughness: {features['roughness']:.3f}")

    # B. Flare Detection
    detector = FlareDetector(threshold=0.25, min_duration=5)
    flares = detector.detect(np.array(time_series_v))
    
    if flares:
        logger.info(f"Detected {len(flares)} flare events.")
        for i, f in enumerate(flares):
            logger.info(f"  Flare {i+1}: Duration={f.duration*args.dt:.1f}s, Peak={f.max_intensity:.2f}")
    else:
        logger.info("No distinct flare events detected.")

    # ---------------------------------------------------------
    # 5. Visualization (Presentation Layer)
    # ---------------------------------------------------------
    logger.info("Generating Artifacts...")
    
    # A. Save Final Snapshot
    snap_path = out_path / "final_state.png"
    plot_snapshot(model.U, model.V, step_i=args.steps, dt=args.dt, save_path=snap_path, show=False)
    logger.info(f"Saved snapshot: {snap_path}")

    # B. Generate Animation
    vid_path = out_path / "simulation.mp4"
    save_animation(history, filename=str(vid_path), fps=30)
    
    # C. Save Checkpoint (for reproducibility)
    ckpt_path = out_path / "checkpoint.npz"
    save_checkpoint(str(ckpt_path), model.U, model.V, meta={"params": str(params), "steps": args.steps})
    logger.info(f"Saved checkpoint: {ckpt_path}")

    logger.info("Pipeline Finished Successfully. 🚀")

if __name__ == "__main__":
    main()