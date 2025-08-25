"""
HOLE: Homological Observation of Latent Embeddings

A library for topological analysis and
visualization of deep learning representations.
"""

# Visualization functions (for advanced users)
# Core functionality
from . import core, utils, visualization
from .visualizer import HOLEVisualizer

__version__ = "0.1.0"
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2024, HOLE Development Team"

__all__ = ["HOLEVisualizer", "core", "utils", "visualization"]


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
