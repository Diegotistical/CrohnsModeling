"""
Checkpointing utilities.
"""
import numpy as np
import json
import os
import time
from typing import Dict, Any, Tuple

def save_checkpoint(path: str, U: np.ndarray, V: np.ndarray, meta: Dict[str, Any] = None):
    """Save simulation state to compressed .npz with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    full_meta = {'timestamp': time.time()}
    if meta:
        full_meta.update(meta)
        
    np.savez_compressed(
        path,
        U=U,
        V=V,
        metadata=json.dumps(full_meta)
    )

def load_checkpoint(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load simulation state and metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint at {path}")
        
    data = np.load(path)
    # Ensure we return native python dict for metadata
    meta = json.loads(str(data['metadata']))
    return data['U'], data['V'], meta