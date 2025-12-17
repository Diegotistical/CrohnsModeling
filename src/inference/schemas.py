"""
Data Schemas for Inference.

Strictly typed definitions for inputs and outputs.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import numpy as np


class SimulationData(BaseModel):
    """Raw output from the solver loader."""

    U: np.ndarray
    V: np.ndarray
    metadata: Dict[str, Any]

    # Allow numpy arrays
    model_config = ConfigDict(arbitrary_types_allowed=True)


class FlareEvent(BaseModel):
    """Structured record of a detected flare."""

    start_step: int
    end_step: int
    duration: int
    max_intensity: float
    avg_intensity: float
    peak_step: int


class TimeSeriesFeatures(BaseModel):
    """Aggregated scalar metrics over time."""

    time: List[float]
    mean_inflammation: List[float]
    max_inflammation: List[float]
    entropy: List[float]


class InferenceResult(BaseModel):
    """Final container for all analysis."""

    simulation_id: str
    flares: List[FlareEvent]
    global_risk_score: float
    stability_index: float
