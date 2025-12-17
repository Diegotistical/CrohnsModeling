"""
Stochastic Flare Process.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class FlareConfig:
    probability: float = 0.01  # Prob of flare per step
    intensity: float = 0.2     # Magnitude of inflammation spike
    radius: int = 4            # Spatial radius of flare event

class StochasticFlare:
    """
    Applies random, localized inflammation spikes to the V field.
    """
    def __init__(self, config: FlareConfig, shape: Tuple[int, int], seed: Optional[int] = None):
        self.config = config
        self.shape = shape
        self.rng = np.random.default_rng(seed)

    def apply_flare(self, V: np.ndarray) -> None:
        """
        In-place modification of V with probability `p`.
        """
        if self.rng.random() < self.config.probability:
            self._trigger_flare(V)

    def _trigger_flare(self, V: np.ndarray):
        # Pick a random center
        cx = self.rng.integers(0, self.shape[0])
        cy = self.rng.integers(0, self.shape[1])
        
        # Define a simple box
        r = self.config.radius
        
        # Slicing with boundary checks (clamping indices)
        x_min = max(0, cx - r)
        x_max = min(self.shape[0], cx + r)
        y_min = max(0, cy - r)
        y_max = min(self.shape[1], cy + r)
        
        # Add inflammation
        V[x_min:x_max, y_min:y_max] += self.config.intensity