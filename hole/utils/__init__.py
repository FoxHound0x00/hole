"""
Utility functions for topological data analysis.

This package contains helper functions for data preprocessing and other utilities.
Distance computations have been moved to hole.core.distance_metrics.
"""

# All distance functions have been moved to core.distance_metrics
from ..core.distance_metrics import (
    chebyshev,
    distance_matrix,
    euclidean,
    manhattan,
    validate_distance_matrix,
)

__all__ = [
    "euclidean",
    "manhattan",
    "chebyshev",
    "distance_matrix",
    "validate_distance_matrix",
]
