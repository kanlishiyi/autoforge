"""
AutoForge - An Intelligent Machine Learning Training & Hyperparameter Tuning Platform

AutoForge provides a comprehensive toolkit for automated hyperparameter optimization,
experiment tracking, and model training, inspired by Karpathy's autoresearch project.
"""

__version__ = "0.1.0"
__author__ = "AutoForge Team"
__license__ = "MIT"

from mltune.core.config import Config
from mltune.core.experiment import Experiment
from mltune.optim.study import Study
from mltune.optim.tuner import Tuner

__all__ = [
    "Config",
    "Experiment",
    "Tuner",
    "Study",
    "__version__",
]
