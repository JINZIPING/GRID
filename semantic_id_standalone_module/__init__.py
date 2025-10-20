"""
Semantic ID Standalone Module

A self-contained implementation of semantic ID generation using K-means clustering
and residual quantization, without external dependencies beyond PyTorch and NumPy.
"""

__version__ = "1.0.0"
__author__ = "GRID Team"

from .src.semantic_id_module import (
    ResidualQuantization,
    MiniBatchKMeans,
    SquaredEuclideanDistance,
    KMeansPlusPlusInitInitializer,
    WeightedSquaredError
)

__all__ = [
    "ResidualQuantization",
    "MiniBatchKMeans", 
    "SquaredEuclideanDistance",
    "KMeansPlusPlusInitInitializer",
    "WeightedSquaredError"
]
