"""
Visualization package for HOLE - Homological Observation of Latent Embeddings.

This package provides high-quality visualization classes and functions for analyzing 
point clouds, distance matrices, and persistence diagrams. The main HOLEVisualizer 
class is located in the parent hole package.
"""

# Distance functions are now in core
from ..core.distance_metrics import distance_matrix, euclidean
from .cluster_flow import ClusterFlowAnalyzer, ComponentEvolutionVisualizer
from .heatmap_dendrograms import PersistenceDendrogram
from .pers_vis import PersVis

# Individual visualization functions (legacy support)
from .persistence_vis import (
    plot_dimensionality_reduction,
    plot_persistence_barcode,
    plot_persistence_diagram,
)
from .scatter_hull import BlobVisualizer

__all__ = [
    # Main visualization classes (no confusing aliases)
    "BlobVisualizer",
    "ComponentEvolutionVisualizer",
    "ClusterFlowAnalyzer",
    "PersistenceDendrogram",
    "PersVis",
    # Legacy functions (backward compatibility)
    "plot_persistence_diagram",
    "plot_persistence_barcode",
    "plot_dimensionality_reduction",
    # Utility functions
    "euclidean",
    "distance_matrix",
]
