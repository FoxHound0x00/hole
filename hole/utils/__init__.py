"""
Utility functions for topological data analysis.

This package contains helper functions for data preprocessing, distance
computations, and TDA evaluation metrics.
"""

# Distance functions live in core.distance_metrics; re-exposed for convenience.
from ..core.distance_metrics import (
    chebyshev,
    distance_matrix,
    euclidean,
    manhattan,
    validate_distance_matrix,
)
from .metrics import (
    compute_clustering_metrics,
    compute_distance_matrix_properties,
    compute_homogeneity,
    compute_persistence_entropy,
    compute_persistence_stability,
    compute_purity,
    compute_robustness_metrics,
    compute_topological_features,
    evaluate_clustering_evolution,
    summarize_metrics,
)

__all__ = [
    # Distance helpers
    "euclidean",
    "manhattan",
    "chebyshev",
    "distance_matrix",
    "validate_distance_matrix",
    # TDA & clustering metrics
    "compute_persistence_entropy",
    "compute_persistence_stability",
    "compute_topological_features",
    "compute_distance_matrix_properties",
    "compute_clustering_metrics",
    "compute_purity",
    "compute_homogeneity",
    "evaluate_clustering_evolution",
    "compute_robustness_metrics",
    "summarize_metrics",
]
