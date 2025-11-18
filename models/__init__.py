"""
Enhanced ML Models Module for Fresh Supply Chain Intelligence System

Advanced model architectures with:
- Multi-model ensembles for improved accuracy
- Uncertainty quantification for reliable predictions  
- Attention mechanisms for interpretability
- Production-ready training and deployment
- Comprehensive evaluation and monitoring
"""

from .vision_model import FreshProduceVisionModel, ModelConfig as VisionConfig
from .forecasting_model import TemporalFusionTransformer, DemandForecaster, ForecastConfig
from .gnn_optimizer import SupplyChainGNN, SupplyChainOptimizer
from .simple_optimizer import SimpleSupplyChainOptimizer

__version__ = "2.0.0"
__author__ = "Fresh Supply Chain Intelligence Team"

# Enhanced model capabilities
ENHANCED_MODEL_FEATURES = {
    "vision_model": {
        "architecture": "EfficientNet-B4 + Attention + Ensemble",
        "capabilities": [
            "Multi-model ensemble (3 models)",
            "Spatial attention mechanism", 
            "Uncertainty quantification via Monte Carlo Dropout",
            "Test-time augmentation (TTA)",
            "Advanced data augmentation pipeline",
            "Grad-CAM visualization for explainability",
            "5-class quality prediction with confidence scores",
            "Production-ready inference with <200ms latency"
        ]
    },
    "forecasting_model": {
        "architecture": "Enhanced Temporal Fusion Transformer",
        "capabilities": [
            "Multi-horizon demand forecasting (1-30 days)",
            "Variable selection networks for feature importance",
            "Multi-head attention for temporal dependencies", 
            "Quantile regression for uncertainty bounds",
            "Gated residual networks for feature processing",
            "Static and temporal feature fusion",
            "Interpretable predictions with attention weights"
        ]
    },
    "optimization_model": {
        "architecture": "Graph Neural Network + Gurobi Solver",
        "capabilities": [
            "Multi-objective route optimization",
            "Graph-based supply chain modeling",
            "Real-time constraint handling",
            "Cost and waste minimization"
        ]
    }
}

def get_model_summary():
    """Get summary of all enhanced model capabilities"""
    return {
        "version": __version__,
        "total_models": len(ENHANCED_MODEL_FEATURES),
        "enhanced_features": ENHANCED_MODEL_FEATURES,
        "production_ready": True,
        "key_improvements": [
            "Ensemble methods for improved accuracy and robustness",
            "Uncertainty quantification for reliable decision making", 
            "Attention mechanisms for model interpretability",
            "Advanced training strategies (mixed precision, scheduling)",
            "Production-optimized inference pipelines"
        ]
    }

__all__ = [
    'FreshProduceVisionModel', 'TemporalFusionTransformer', 'DemandForecaster',
    'SupplyChainGNN', 'SupplyChainOptimizer', 'SimpleSupplyChainOptimizer',
    'VisionConfig', 'ForecastConfig', 'get_model_summary', 'ENHANCED_MODEL_FEATURES'
]