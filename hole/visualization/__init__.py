"""
Visualization package for HOLE - Homological Observation of Latent Embeddings.

This package provides high-quality visualization classes and functions for analyzing 
point clouds, distance matrices, and persistence diagrams. The main HOLEVisualizer 
class is located in the parent hole package.
"""

# Distance functions are now in core
from ..core.distance_metrics import distance_matrix, euclidean
from .cluster_flow import ClusterFlowAnalyzer
from .cluster_flow import ComponentEvolutionVisualizer as ClusterFlow
from .heatmap_dendrograms import PersistenceDendrogram as HeatmapDendrograms
from .pers_vis import PersVis

# Individual visualization functions (legacy support)
from .persistence_vis import (
    plot_dimensionality_reduction,
    plot_persistence_barcode,
    plot_persistence_diagram,
)

# High-quality visualization classes
from .scatter_hull import BlobVisualizer as ScatterHull

__all__ = [
    # High-quality visualization classes
    "ScatterHull",
    "ClusterFlow",
    "ClusterFlowAnalyzer",
    "HeatmapDendrograms",
    "PersVis",
    # Legacy functions (backward compatibility)
    "plot_persistence_diagram",
    "plot_persistence_barcode",
    "plot_dimensionality_reduction",
    # Utility functions
    "euclidean",
    "distance_matrix",
]
