"""
Political Stability Prediction using ML
Predicting FSI (Fragile States Index) using economic and social indicators
"""

__version__ = "0.1.0"
__author__ = "Jonathan Kissack"

# Export all models for easy import
from .models import (
    KNNPredictor,
    MLPPredictor,
    PanelAnalyzer,
    RandomForestPredictor,
    SVRPredictor,
    XGBoostPredictor,
)

__all__ = [
    "RandomForestPredictor",
    "XGBoostPredictor",
    "KNNPredictor",
    "SVRPredictor",
    "MLPPredictor",
    "PanelAnalyzer",
]
