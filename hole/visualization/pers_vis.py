"""
PersVis - Persistence Visualization Class

This module provides a clean class interface for persistence diagrams,
barcodes, and dimensionality reduction visualizations.
"""

from typing import Dict, List, Optional, Tuple, Union

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

# Import the existing functions
from .persistence_vis import (
    plot_dimensionality_reduction,
    plot_persistence_barcode,
    plot_persistence_diagram,
)


class PersVis:
    """
    Persistence Visualization class for persistence diagrams, barcodes, and dimensionality reduction.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        Initialize the persistence visualizer.

        Args:
            figsize: Figure size for plots
            dpi: DPI for saved plots
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_persistence_diagram(
        self,
        persistence: List,
        ax: Optional[plt.Axes] = None,
        title: str = "Persistence Diagram",
        pts: int = 10,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot persistence diagram.

        Args:
            persistence: List of persistence pairs
            ax: Optional matplotlib axes to plot on
            title: Plot title
            pts: Number of most persistent features to highlight
            **kwargs: Additional arguments

        Returns:
            matplotlib Axes object
        """
        return plot_persistence_diagram(
            persistence, ax=ax, title=title, pts=pts, **kwargs
        )

    def plot_persistence_barcode(
        self,
        persistence: List,
        ax: Optional[plt.Axes] = None,
        title: str = "Persistence Barcode",
        pts: int = 10,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot persistence barcode.

        Args:
            persistence: List of persistence pairs
            ax: Optional matplotlib axes to plot on
            title: Plot title
            pts: Number of most persistent features to highlight
            **kwargs: Additional arguments

        Returns:
            matplotlib Axes object
        """
        return plot_persistence_barcode(
            persistence, ax=ax, title=title, pts=pts, **kwargs
        )

    def plot_dimensionality_reduction(
        self,
        point_cloud: np.ndarray,
        method: str = "pca",
        ax: Optional[plt.Axes] = None,
        true_labels: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot dimensionality reduction visualization.

        Args:
            point_cloud: Input data array
            method: Dimensionality reduction method ('pca', 'tsne', 'mds')
            ax: Optional matplotlib axes to plot on
            true_labels: True labels for coloring points
            title: Plot title
            **kwargs: Additional arguments

        Returns:
            matplotlib Axes object
        """
        return plot_dimensionality_reduction(
            point_cloud,
            method=method,
            ax=ax,
            true_labels=true_labels,
            title=title,
            **kwargs,
        )
