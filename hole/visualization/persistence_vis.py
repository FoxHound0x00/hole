"""
Persistence visualizations including diagrams, barcodes, and dimensionality reduction.

This module provides functions for visualizing persistent homology results
and performing dimensionality reduction for data exploration.
"""

import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

# Import from core
from ..core.distance_metrics import distance_matrix, euclidean


def plot_persistence_barcode(
    persistence: List[Tuple],
    pts: int = 10,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> plt.Axes:
    """
    Plot persistence barcode from persistence data.

    Parameters
    ----------
    persistence : list
        List of persistence pairs from GUDHI
    pts : int, optional
        Number of persistence points to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size if creating new figure

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.set_style("whitegrid")

    birth_times = [birth for _, (birth, death) in persistence[:pts]]
    death_times = [death for _, (birth, death) in persistence[:pts]]

    if not birth_times:
        ax.text(
            0.5,
            0.5,
            "No persistence data to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    min_birth = min(birth_times)
    max_death = max(d for d in death_times if d < float("inf"))
    delta = (max_death - min_birth) * 0.1
    infinity = max_death + delta
    axis_start = min_birth - delta
    axis_end = max_death + delta * 2

    dimensions = sorted(set(dim for dim, _ in persistence[:pts]))
    palette = sns.color_palette("Set1", n_colors=len(dimensions))
    color_map = {dim: palette[i] for i, dim in enumerate(dimensions)}

    for i, (dim, (birth, death)) in enumerate(persistence[:pts]):
        bar_length = (death - birth) if death != float("inf") else (infinity - birth)
        ax.barh(i, bar_length, left=birth, color=color_map[dim], alpha=0.7)

    legend_patches = [
        mpatches.Patch(color=color_map[dim], label=f"H{dim}") for dim in dimensions
    ]
    ax.legend(handles=legend_patches, loc="best", fontsize=10)

    if title is None:
        title = "Persistence Barcode"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Filtration Value", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_yticks([])
    ax.invert_yaxis()

    if birth_times:
        ax.set_xlim((axis_start, axis_end))

    return ax


def plot_persistence_diagram(
    persistence: List[Tuple],
    pts: int = 10,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: tuple = (6, 6),
) -> plt.Axes:
    """
    Plot persistence diagram from persistence data.

    Parameters
    ----------
    persistence : list
        List of persistence pairs from GUDHI
    pts : int, optional
        Number of persistence points to plot
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size if creating new figure

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.set_style("whitegrid")

    birth_times = [birth for _, (birth, death) in persistence[:pts]]
    death_times = [death for _, (birth, death) in persistence[:pts]]

    if not birth_times:
        ax.text(
            0.5,
            0.5,
            "No persistence data to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ax

    min_birth = min(birth_times)
    max_death = max(d for d in death_times if d < float("inf"))

    delta = (max_death - min_birth) * 0.1
    infinity = max_death + 3 * delta
    axis_end = max_death + delta
    axis_start = min_birth - delta

    dimensions = sorted(set(dim for dim, _ in persistence[:pts]))
    palette = sns.color_palette("Set1", n_colors=len(dimensions))
    color_map = {dim: palette[i] for i, dim in enumerate(dimensions)}

    x = [birth for (dim, (birth, death)) in persistence[:pts]]
    y = [
        death if death != float("inf") else infinity
        for (dim, (birth, death)) in persistence[:pts]
    ]
    c = [color_map[dim] for (dim, (birth, death)) in persistence[:pts]]

    sizes = [
        20 + 80 * ((death - birth) / (max(1e-5, max_death - min_birth)))
        for (_, (birth, death)) in persistence[:pts]
    ]
    ax.scatter(x, y, s=sizes, alpha=0.7, c=c, edgecolors="k")

    # Diagonal line
    ax.fill_between(
        [axis_start, axis_end],
        [axis_start, axis_end],
        axis_start,
        color="lightgrey",
        alpha=0.5,
    )

    # Handle infinite death times
    if any(death == float("inf") for (_, (birth, death)) in persistence[:pts]):
        ax.scatter(
            [min_birth],
            [infinity],
            s=150,
            color="black",
            marker="*",
            label="Infinite Death",
        )
        ax.plot(
            [axis_start, axis_end],
            [infinity, infinity],
            linewidth=1.0,
            color="k",
            alpha=0.6,
        )

        yt = np.array(ax.get_yticks())
        yt = yt[yt < axis_end]  # Avoid out-of-bounds y-ticks
        yt = np.append(yt, infinity)
        ytl = ["%.3f" % e for e in yt]
        ytl[-1] = r"$+\infty$"
        ax.set_yticks(yt)
        ax.set_yticklabels(ytl)

    ax.legend(
        handles=[
            mpatches.Patch(color=color_map[dim], label=f"H{dim}") for dim in dimensions
        ],
        title="Dimension",
        loc="lower right",
    )

    ax.set_xlabel("Birth", fontsize=12)
    ax.set_ylabel("Death", fontsize=12)

    if title is None:
        title = "Persistence Diagram"
    ax.set_title(title, fontsize=12)

    ax.set_xlim(axis_start, axis_end)
    ax.set_ylim(min_birth, infinity + delta / 2)

    return ax


def plot_dimensionality_reduction(
    data: Union[np.ndarray, tuple],
    method: str = "pca",
    labels: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    point_size: float = 50,
    alpha: float = 0.7,
    show_legend: bool = True,
    **kwargs,
) -> plt.Axes:
    """
    Plot dimensionality reduction visualization.

    Parameters
    ----------
    data : np.ndarray or tuple
        Input data. Can be:
        - 2D array of features for dimensionality reduction
        - Distance matrix (will be converted using MDS)
        - Tuple of (x, y) coordinates for direct plotting
    method : str, optional
        Dimensionality reduction method ('pca', 'tsne', 'mds')
    labels : np.ndarray, optional
        Labels for coloring points
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.
    title : str, optional
        Title for the plot
    figsize : tuple, optional
        Figure size if creating new figure
    point_size : float, optional
        Size of scatter points
    alpha : float, optional
        Alpha transparency for points
    show_legend : bool, optional
        Whether to show legend
    **kwargs : dict
        Additional plotting arguments

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Handle different data input types
    if isinstance(data, tuple) and len(data) == 2:
        # Direct coordinates provided
        coords_2d = np.column_stack(data)
    else:
        # Need dimensionality reduction
        coords_2d = _perform_dimensionality_reduction(data, method)

    # Generate colors based on labels
    if labels is not None:
        unique_labels = sorted(set(labels))
        n_colors = len(unique_labels)

        # Choose appropriate discrete colormap
        if n_colors <= 10:
            cmap = plt.cm.tab10
            colors = [cmap(i) for i in range(n_colors)]
        elif n_colors <= 20:
            cmap = plt.cm.tab20
            colors = [cmap(i) for i in range(n_colors)]
        else:
            # For many labels, use multiple discrete colormaps
            discrete_colormaps = [
                plt.cm.tab20,
                plt.cm.tab20b,
                plt.cm.tab20c,
                plt.cm.Set1,
                plt.cm.Set2,
                plt.cm.Set3,
            ]
            colors = []
            for i in range(n_colors):
                cmap_idx = i // 20
                color_idx = i % 20
                if cmap_idx < len(discrete_colormaps):
                    cmap = discrete_colormaps[cmap_idx]
                    colors.append(cmap(color_idx % cmap.N))
                else:
                    # Fallback to HSV generation
                    hue = (i * 0.618033988749895) % 1.0
                    colors.append(mcolors.hsv_to_rgb([hue, 0.7, 0.9]) + (1.0,))

        label_to_color = {}
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Special gray color for noise
                label_to_color[label] = (0.5, 0.5, 0.5, 1.0)
            else:
                label_to_color[label] = colors[i]

        point_colors = [label_to_color[label] for label in labels]
    else:
        point_colors = "blue"

    # Create scatter plot
    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=point_colors,
        s=point_size,
        alpha=alpha,
        edgecolors="black",
        linewidth=0.5,
        **kwargs,
    )

    # Add legend if labels provided
    if show_legend and labels is not None:
        legend_elements = []
        for label in unique_labels:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=label_to_color[label],
                    markeredgecolor="black",
                    markersize=8,
                    label=f"Class {label}",
                )
            )
        ax.legend(
            handles=legend_elements,
            title="Labels",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    # Styling
    ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
    ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)

    if title is None:
        title = f"{method.upper()} Visualization"
    ax.set_title(title, fontsize=14, pad=20)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def _perform_dimensionality_reduction(
    data: np.ndarray, method: str = "pca", n_components: int = 2, random_state: int = 42
) -> np.ndarray:
    """
    Perform dimensionality reduction on data.

    Parameters
    ----------
    data : np.ndarray
        Input data for dimensionality reduction
    method : str
        Method to use ('pca', 'tsne', 'mds')
    n_components : int
        Number of components for output
    random_state : int
        Random state for reproducibility

    Returns
    -------
    np.ndarray
        Reduced data
    """
    if method.lower() == "pca":
        if data.shape[0] == data.shape[1] and np.allclose(data, data.T):
            # Distance matrix - convert to coordinates using MDS first
            warnings.warn(
                "PCA requested but distance matrix detected. Using MDS instead."
            )
            reducer = MDS(
                n_components=n_components,
                random_state=random_state,
                dissimilarity="precomputed",
                n_init=4,
                max_iter=1000,
            )
            return reducer.fit_transform(data)
        else:
            reducer = PCA(n_components=n_components, random_state=random_state)
            return reducer.fit_transform(data)

    elif method.lower() == "tsne":
        if data.shape[0] == data.shape[1] and np.allclose(data, data.T):
            # Distance matrix
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                metric="precomputed",
                perplexity=min(30, (data.shape[0] - 1) // 3),
                n_iter=1000,
            )
        else:
            # Feature matrix
            perplexity = min(30, (data.shape[0] - 1) // 3)
            perplexity = max(5, perplexity)  # Ensure minimum perplexity
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=perplexity,
                max_iter=1000,
            )
        return reducer.fit_transform(data)

    elif method.lower() == "mds":
        if data.shape[0] == data.shape[1] and np.allclose(data, data.T):
            # Distance matrix
            reducer = MDS(
                n_components=n_components,
                random_state=random_state,
                dissimilarity="precomputed",
                n_init=4,
                max_iter=1000,
            )
        else:
            # Feature matrix
            reducer = MDS(
                n_components=n_components,
                random_state=random_state,
                dissimilarity="euclidean",
                n_init=4,
                max_iter=1000,
            )
        return reducer.fit_transform(data)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'mds'")
