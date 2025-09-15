"""
HOLE: Homological Observation of Latent Embeddings

A library for topological analysis and
visualization of deep learning representations.
"""

from loguru import logger

# Configure loguru for the HOLE package
logger.add(
    sink=lambda msg: print(msg, end=""),  # Print to stdout
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

# Core functionality
from . import core, utils, visualization

# Import commonly used functions to top level for convenience
from .core.distance_metrics import (
    chebyshev_distance,
    cosine_distance,
    density_normalized_distance,
    euclidean_distance,
    geodesic_distances,
    mahalanobis_distance,
    manhattan_distance,
)
from .core.mst_processor import MSTProcessor
from .visualization.cluster_flow import ClusterFlowAnalyzer
from .visualization.heatmap_dendrograms import PersistenceDendrogram
from .visualization.scatter_hull import BlobVisualizer
from .visualizer import HOLEVisualizer

__version__ = "0.1.0"
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2024, HOLE Development Team"

__all__ = [
    # Main classes
    "HOLEVisualizer",
    "MSTProcessor",
    "ClusterFlowAnalyzer",
    "BlobVisualizer",
    "PersistenceDendrogram",
    # Distance metrics
    "euclidean_distance",
    "cosine_distance",
    "mahalanobis_distance",
    "manhattan_distance",
    "chebyshev_distance",
    "geodesic_distances",
    "density_normalized_distance",
    # Submodules
    "core",
    "utils",
    "visualization",
]


def get_version():
    """Return the version of HOLE."""
    return __version__


def get_info():
    """Return basic information about HOLE."""
    return {
        "name": "HOLE",
        "version": __version__,
        "description": "Homological Observation of Latent Embeddings",
        "author": "Sudhanva M Athreya, University of Utah",
        "license": __license__,
    }
