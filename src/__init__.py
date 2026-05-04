"""
CIFAR-10 Human Disagreement Predictor

A deep learning system that predicts human annotator disagreement on CIFAR-10 images.
Rather than predicting a single hard class label, the system predicts the full probability
distribution over labels that reflects how human annotators disagree about image classification.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from . import data_pipeline
from . import model
from . import losses
from . import training
from . import evaluation
from . import visualization

__all__ = [
    "data_pipeline",
    "model",
    "losses",
    "training",
    "evaluation",
    "visualization",
]
