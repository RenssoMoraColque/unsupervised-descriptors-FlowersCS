"""
Unsupervised Descriptors for STL-10 Dataset
==========================================

This package contains implementations of classical (non-deep learning) 
image descriptors for the STL-10 hackathon challenge.

Modules:
--------
- descriptors: Various image descriptor implementations
- evaluation: Evaluation metrics and robustness testing
- utils: Utility functions for data loading and preprocessing
"""

__version__ = "1.0.0"
__author__ = "Team FlowersCS"

from . import descriptors
from . import evaluation
from . import utils

__all__ = ["descriptors", "evaluation", "utils"]