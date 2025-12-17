"""
Simulation health monitoring.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Monitor:
    def __init__(self, abort_on_nan: bool = True):
        self.abort = abort_on_nan
        self.stats: dict[str, list] = {'max_v': [], 'mean_u': []}

    def check(self, step: int, U: np.ndarray, V: np.ndarray):
        # Calculate metrics
        v_max = np.max(V)
        u_max = np.max(U)
        u_mean = np.mean(U)
        
        # Store for analysis
        self.stats['max_v'].append(v_max)
        self.stats['mean_u'].append(u_mean)

        # Safety Check
        if self.abort:
            is_unstable = (
                np.isnan(u_mean) or 
                np.isinf(u_max) or 
                np.isnan(v_max) or 
                np.isinf(v_max)
            )
            
            if is_unstable:
                logger.critical(f"Simulation exploded at step {step}")
                raise RuntimeError(f"Numerical Instability Detected at step {step}")