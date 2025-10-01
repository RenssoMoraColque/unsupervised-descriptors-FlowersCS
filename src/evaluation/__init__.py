"""
Evaluation Module
================

This module contains evaluation metrics, robustness testing,
and classification utilities for the descriptor evaluation pipeline.
"""

from .metrics import (
    ClassificationMetrics,
    ClusteringMetrics,
    RetrievalMetrics
)

from .robustness import (
    RobustnessEvaluator,
    ImageTransforms
)

from .classifiers import (
    DescriptorClassifier,
    CrossValidationEvaluator
)

__all__ = [
    "ClassificationMetrics",
    "ClusteringMetrics", 
    "RetrievalMetrics",
    "RobustnessEvaluator",
    "ImageTransforms",
    "DescriptorClassifier",
    "CrossValidationEvaluator"
]