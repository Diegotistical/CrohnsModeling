from src.inference.schemas import SimulationData, FlareEvent, InferenceResult
from src.inference.loaders import load_simulation_data
from src.inference.postprocessing import compute_inflammation_series
from src.inference.feature_extraction import extract_frame_features
from src.inference.flare_detection import FlareDetector
from src.inference.model_inference import HeuristicRiskScorer