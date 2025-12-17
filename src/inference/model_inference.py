"""
Predictive Model Wrappers.
"""
from abc import ABC, abstractmethod
import numpy as np

class RiskModelProtocol(ABC):
    @abstractmethod
    def predict_risk(self, features: dict) -> float:
        pass

class HeuristicRiskScorer(RiskModelProtocol):
    """
    Baseline model: Risks is proportional to max inflammation + entropy.
    Used before ML model is trained.
    """
    def predict_risk(self, features: dict) -> float:
        # Simple weighted sum of features
        risk = 0.7 * features.get('max', 0.0) + 0.3 * features.get('entropy', 0.0)
        return min(1.0, risk)

# Placeholder for future Neural Surrogate
class SurrogateModel(RiskModelProtocol):
    def predict_risk(self, features: dict) -> float:
        # TODO: Load ONNX/PyTorch model here
        raise NotImplementedError("ML model not trained yet.")