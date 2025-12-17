"""
Data Loading Utilities.
"""
import numpy as np
import json
import os
from src.inference.schemas import SimulationData

def load_simulation_data(path: str) -> SimulationData:
    """
    Load a compressed simulation checkpoint (.npz).
    
    Args:
        path: Path to the .npz file.
        
    Returns:
        SimulationData object with validated structure.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        data = np.load(path)
        
        # Parse metadata (stored as JSON string in solver)
        meta_raw = data.get('metadata', '{}')
        if isinstance(meta_raw, np.ndarray):
            meta_raw = str(meta_raw)
        
        metadata = json.loads(str(meta_raw)) if meta_raw else {}

        return SimulationData(
            U=data['U'],
            V=data['V'],
            metadata=metadata
        )
    except Exception as e:
        raise ValueError(f"Failed to load or validate simulation data at {path}: {e}")