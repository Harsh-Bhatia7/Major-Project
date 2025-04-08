"""
Iris Classification Package

A comprehensive tool for iris flower species classification using
various machine learning techniques.
"""

# Import key classes to make them available directly from the package
from .logger import Logger
from .classifier import IrisClassifier
from .data_loading import DataLoader
from .feature_proc import FeatureProcessor
from .model_train import ModelTrainer
from .model_eval import ModelEvaluator
from .model_storage import ModelSaver

# Define what gets imported with "from package import *"
__all__ = [
    'Logger',
    'IrisClassifier',
    'DataLoader',
    'FeatureProcessor',
    'ModelTrainer',
    'ModelEvaluator',
    'ModelSaver',
]

# Package metadata
__version__ = '0.1.0'
__author__ = 'Harsh Bhatia, Krish Talwar'
