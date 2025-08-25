"""
Core persistent homology computation functions.

This module contains the fundamental functions for computing persistent homology
that are used throughout the library.
"""

from collections import defaultdict
from typing import List, Optional, Tuple, Union

import gudhi as gd
import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances


def compute_persistence(
    distance_matrix: np.ndarray, max_dimension: int = 1, max_edge_length: float = np.inf
) -> List[Tuple]:
    """
    Compute persistent homology from distance matrix using GUDHI.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix of shape (n_points, n_points)
    max_dimension : int, optional
        Maximum dimension for persistence computation. Default is 1.
    max_edge_length : float, optional
        Maximum edge length for Rips complex. Default is np.inf.

    Returns
    -------
    list
        List of persistence pairs from GUDHI
    """
    # Create Rips complex
    rc = gd.RipsComplex(
        distance_matrix=distance_matrix, max_edge_length=max_edge_length
    )
    st = rc.create_simplex_tree(max_dimension=max_dimension)
    persistence = st.persistence()

    return persistence


def extract_death_thresholds(
    persistence: List[Tuple], dimension: int = 0
) -> List[float]:
    """
    Extract death thresholds for a specific dimension from persistence data.

    Parameters
    ----------
    persistence : list
        List of persistence pairs from GUDHI
    dimension : int, optional
        Dimension to extract thresholds for. Default is 0 (connected components).

    Returns
    -------
    list
        Sorted list of death thresholds
    """
    death_thresholds = []
    for dim, (birth, death) in persistence:
        if dim == dimension and death != float("inf"):
            death_thresholds.append(death)

    return sorted(set(death_thresholds))


def compute_cluster_evolution(
    distance_matrix: np.ndarray, thresholds: List[float]
) -> dict:
    """
    Compute cluster evolution through different filtration thresholds.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix of shape (n_points, n_points)
    thresholds : list
        List of filtration thresholds to analyze

    Returns
    -------
    dict
        Dictionary mapping thresholds to cluster information
    """
    n_points = distance_matrix.shape[0]
    cluster_evolution = {}

    for threshold in thresholds:
        # Create adjacency matrix for this threshold
        adj_matrix = (distance_matrix <= threshold).astype(int)
        np.fill_diagonal(adj_matrix, 0)

        # Find connected components
        graph = nx.from_numpy_array(adj_matrix)
        components = list(nx.connected_components(graph))

        # Create cluster labels
        cluster_labels = np.zeros(n_points, dtype=int)
        for cluster_id, component in enumerate(components):
            for node in component:
                cluster_labels[node] = cluster_id

        cluster_evolution[threshold] = {
            "n_clusters": len(components),
            "labels": cluster_labels,
            "components": components,
        }

    return cluster_evolution


def select_meaningful_thresholds(
    all_thresholds: List[float], max_thresholds: int = 8, strategy: str = "uniform"
) -> List[float]:
    """
    Select meaningful thresholds from a list for visualization.

    Parameters
    ----------
    all_thresholds : list
        Complete list of available thresholds
    max_thresholds : int, optional
        Maximum number of thresholds to select. Default is 8.
    strategy : str, optional
        Strategy for selection ('uniform', 'logarithmic'). Default is 'uniform'.

    Returns
    -------
    list
        Selected thresholds
    """
    if len(all_thresholds) <= max_thresholds:
        return all_thresholds

    if strategy == "uniform":
        # Select thresholds at regular intervals
        indices = np.linspace(0, len(all_thresholds) - 1, max_thresholds, dtype=int)
        return [all_thresholds[i] for i in indices]

    elif strategy == "logarithmic":
        # Select thresholds with logarithmic spacing
        if all_thresholds[0] <= 0:
            # Handle case where minimum threshold is 0 or negative
            min_thresh = max(all_thresholds[0], 1e-10)
        else:
            min_thresh = all_thresholds[0]

        max_thresh = all_thresholds[-1]
        log_thresholds = np.logspace(
            np.log10(min_thresh), np.log10(max_thresh), max_thresholds
        )

        # Find closest actual thresholds
        selected = []
        for target in log_thresholds:
            closest_idx = np.argmin(np.abs(np.array(all_thresholds) - target))
            selected.append(all_thresholds[closest_idx])

        return sorted(list(set(selected)))

    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Use 'uniform' or 'logarithmic'"
        )


def track_cluster_flows(
    cluster_evolution: dict, thresholds: Optional[List[float]] = None
) -> List[dict]:
    """
    Track how clusters flow between thresholds.

    Parameters
    ----------
    cluster_evolution : dict
        Dictionary mapping thresholds to cluster information
    thresholds : list, optional
        Ordered list of thresholds. If None, uses sorted keys from cluster_evolution.

    Returns
    -------
    list
        List of flow information between consecutive thresholds
    """
    if thresholds is None:
        thresholds = sorted(cluster_evolution.keys())

    flows = []

    for i in range(len(thresholds) - 1):
        current_threshold = thresholds[i]
        next_threshold = thresholds[i + 1]

        current_labels = cluster_evolution[current_threshold]["labels"]
        next_labels = cluster_evolution[next_threshold]["labels"]

        # Track transitions
        transition_counts = defaultdict(int)
        for point_idx in range(len(current_labels)):
            current_cluster = current_labels[point_idx]
            next_cluster = next_labels[point_idx]
            transition_counts[(current_cluster, next_cluster)] += 1

        flows.append(
            {
                "from_threshold": current_threshold,
                "to_threshold": next_threshold,
                "transitions": dict(transition_counts),
            }
        )

    return flows


def compute_persistence_statistics(persistence: List[Tuple]) -> dict:
    """
    Compute summary statistics from persistence data.

    Parameters
    ----------
    persistence : list
        List of persistence pairs from GUDHI

    Returns
    -------
    dict
        Dictionary of statistics
    """
    stats = {
        "total_features": len(persistence),
        "dimensions": {},
        "infinite_features": 0,
        "finite_features": 0,
    }

    # Analyze by dimension
    for dim, (birth, death) in persistence:
        if dim not in stats["dimensions"]:
            stats["dimensions"][dim] = {
                "count": 0,
                "birth_range": [float("inf"), float("-inf")],
                "death_range": [float("inf"), float("-inf")],
                "lifespans": [],
            }

        dim_stats = stats["dimensions"][dim]
        dim_stats["count"] += 1

        # Update birth range
        dim_stats["birth_range"][0] = min(dim_stats["birth_range"][0], birth)
        dim_stats["birth_range"][1] = max(dim_stats["birth_range"][1], birth)

        if death != float("inf"):
            # Finite feature
            stats["finite_features"] += 1
            lifespan = death - birth
            dim_stats["lifespans"].append(lifespan)

            # Update death range
            if dim_stats["death_range"][0] == float("inf"):
                dim_stats["death_range"] = [death, death]
            else:
                dim_stats["death_range"][0] = min(dim_stats["death_range"][0], death)
                dim_stats["death_range"][1] = max(dim_stats["death_range"][1], death)
        else:
            # Infinite feature
            stats["infinite_features"] += 1

    # Compute lifespan statistics for each dimension
    for dim in stats["dimensions"]:
        lifespans = stats["dimensions"][dim]["lifespans"]
        if lifespans:
            stats["dimensions"][dim]["lifespan_stats"] = {
                "mean": np.mean(lifespans),
                "std": np.std(lifespans),
                "min": np.min(lifespans),
                "max": np.max(lifespans),
                "median": np.median(lifespans),
            }
        else:
            stats["dimensions"][dim]["lifespan_stats"] = None

    return stats
