"""
Flare Detection Logic.
"""
import numpy as np
from typing import List
from src.inference.schemas import FlareEvent

class FlareDetector:
    def __init__(self, threshold: float = 0.4, min_duration: int = 5):
        self.threshold = threshold
        self.min_duration = min_duration

    def detect(self, time_series: np.ndarray) -> List[FlareEvent]:
        """
        Scan a 1D inflammation time-series for flare events.
        """
        # Boolean mask of active flare steps
        is_flare = time_series > self.threshold
        
        # Find edges (0->1 is start, 1->0 is end)
        diff = np.diff(is_flare.astype(int), prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        events = []
        for s, e in zip(starts, ends):
            duration = e - s
            if duration >= self.min_duration:
                # Extract segment for stats
                segment = time_series[s:e]
                peak_idx = s + np.argmax(segment)
                
                events.append(FlareEvent(
                    start_step=int(s),
                    end_step=int(e),
                    duration=int(duration),
                    max_intensity=float(np.max(segment)),
                    avg_intensity=float(np.mean(segment)),
                    peak_step=int(peak_idx)
                ))
        return events