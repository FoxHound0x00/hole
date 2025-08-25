"""
Core functions for topological data analysis.

This package contains the fundamental algorithms and computations
needed by the visualization and analysis functions.
"""

from .persistence import (
    compute_cluster_evolution,
    compute_persistence,
    compute_persistence_statistics,
    extract_death_thresholds,
    select_meaningful_thresholds,
    track_cluster_flows,
)

__all__ = [
    "compute_persistence",
    "extract_death_thresholds",
    "compute_cluster_evolution",
    "select_meaningful_thresholds",
    "track_cluster_flows",
    "compute_persistence_statistics",
]
