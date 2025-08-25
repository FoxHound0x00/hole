"""
Metrics and evaluation utilities for topological data analysis.

This module provides various metrics for evaluating clustering quality,
topological features, and model performance in the context of TDA.
"""

from typing import Dict, List, Optional, Tuple, Union

import gudhi as gd
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def compute_persistence_entropy(persistence_diagram: List[Tuple]) -> float:
    """
    Compute the entropy of a persistence diagram.

    Args:
        persistence_diagram: List of (birth, death) tuples

    Returns:
        Persistence entropy value
    """
    if not persistence_diagram:
        return 0.0

    # Extract lifetimes (death - birth)
    lifetimes = []
    for birth, death in persistence_diagram:
        if death != float("inf") and death > birth:
            lifetimes.append(death - birth)

    if not lifetimes:
        return 0.0

    # Normalize lifetimes to form probability distribution
    lifetimes = np.array(lifetimes)
    total_lifetime = np.sum(lifetimes)

    if total_lifetime == 0:
        return 0.0

    probabilities = lifetimes / total_lifetime

    # Compute entropy
    return entropy(probabilities)


def compute_persistence_stability(
    persistence1: List[Tuple], persistence2: List[Tuple]
) -> float:
    """
    Compute stability between two persistence diagrams using bottleneck distance.

    Args:
        persistence1: First persistence diagram
        persistence2: Second persistence diagram

    Returns:
        Bottleneck distance between diagrams
    """
    try:
        # Convert to format expected by GUDHI
        diag1 = [
            (birth, death) for birth, death in persistence1 if death != float("inf")
        ]
        diag2 = [
            (birth, death) for birth, death in persistence2 if death != float("inf")
        ]

        if not diag1 or not diag2:
            return float("inf")

        return gd.bottleneck_distance(diag1, diag2)
    except Exception:
        return float("inf")


def compute_clustering_metrics(
    data: np.ndarray,
    cluster_labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive clustering evaluation metrics.

    Args:
        data: Original data points
        cluster_labels: Predicted cluster labels
        true_labels: Optional ground truth labels

    Returns:
        Dictionary of clustering metrics
    """
    metrics = {}

    # Internal clustering metrics (don't require true labels)
    if len(np.unique(cluster_labels)) > 1:
        try:
            metrics["silhouette_score"] = silhouette_score(data, cluster_labels)
        except:
            metrics["silhouette_score"] = -1.0

        try:
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(
                data, cluster_labels
            )
        except:
            metrics["calinski_harabasz_score"] = 0.0

        try:
            metrics["davies_bouldin_score"] = davies_bouldin_score(data, cluster_labels)
        except:
            metrics["davies_bouldin_score"] = float("inf")
    else:
        metrics["silhouette_score"] = -1.0
        metrics["calinski_harabasz_score"] = 0.0
        metrics["davies_bouldin_score"] = float("inf")

    # External clustering metrics (require true labels)
    if true_labels is not None:
        try:
            metrics["adjusted_rand_score"] = adjusted_rand_score(
                true_labels, cluster_labels
            )
        except:
            metrics["adjusted_rand_score"] = 0.0

        try:
            metrics["normalized_mutual_info"] = normalized_mutual_info_score(
                true_labels, cluster_labels
            )
        except:
            metrics["normalized_mutual_info"] = 0.0

        # Purity and homogeneity
        metrics["purity"] = compute_purity(true_labels, cluster_labels)
        metrics["homogeneity"] = compute_homogeneity(true_labels, cluster_labels)

    return metrics


def compute_purity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """
    Compute clustering purity.

    Args:
        true_labels: Ground truth labels
        cluster_labels: Predicted cluster labels

    Returns:
        Purity score (0-1)
    """
    if len(true_labels) != len(cluster_labels):
        return 0.0

    total_correct = 0
    total_points = len(true_labels)

    for cluster_id in np.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = true_labels[cluster_mask]

        if len(cluster_true_labels) > 0:
            # Most common true label in this cluster
            unique_labels, counts = np.unique(cluster_true_labels, return_counts=True)
            max_count = np.max(counts)
            total_correct += max_count

    return total_correct / total_points if total_points > 0 else 0.0


def compute_homogeneity(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """
    Compute clustering homogeneity.

    Args:
        true_labels: Ground truth labels
        cluster_labels: Predicted cluster labels

    Returns:
        Homogeneity score (0-1)
    """
    if len(true_labels) != len(cluster_labels):
        return 0.0

    total_correct = 0
    total_points = len(true_labels)

    for true_label in np.unique(true_labels):
        true_mask = true_labels == true_label
        true_cluster_labels = cluster_labels[true_mask]

        if len(true_cluster_labels) > 0:
            # Most common cluster for this true label
            unique_clusters, counts = np.unique(true_cluster_labels, return_counts=True)
            max_count = np.max(counts)
            total_correct += max_count

    return total_correct / total_points if total_points > 0 else 0.0


def compute_topological_features(
    distance_matrix: np.ndarray, max_dimension: int = 1
) -> Dict[str, Union[float, int, List]]:
    """
    Compute topological features from a distance matrix.

    Args:
        distance_matrix: 2D distance matrix
        max_dimension: Maximum dimension for homology computation

    Returns:
        Dictionary of topological features
    """
    features = {}

    try:
        # Create Rips complex
        rips_complex = gd.RipsComplex(distance_matrix=distance_matrix)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

        # Compute persistence
        persistence = simplex_tree.persistence()

        # Extract features by dimension
        for dim in range(max_dimension + 1):
            dim_features = [
                (birth, death) for d, (birth, death) in persistence if d == dim
            ]

            # Basic statistics
            features[f"dim_{dim}_count"] = len(dim_features)

            if dim_features:
                lifetimes = [
                    death - birth
                    for birth, death in dim_features
                    if death != float("inf")
                ]
                if lifetimes:
                    features[f"dim_{dim}_mean_lifetime"] = np.mean(lifetimes)
                    features[f"dim_{dim}_std_lifetime"] = np.std(lifetimes)
                    features[f"dim_{dim}_max_lifetime"] = np.max(lifetimes)
                    features[f"dim_{dim}_sum_lifetime"] = np.sum(lifetimes)
                else:
                    features[f"dim_{dim}_mean_lifetime"] = 0.0
                    features[f"dim_{dim}_std_lifetime"] = 0.0
                    features[f"dim_{dim}_max_lifetime"] = 0.0
                    features[f"dim_{dim}_sum_lifetime"] = 0.0

                # Persistence entropy
                features[f"dim_{dim}_entropy"] = compute_persistence_entropy(
                    dim_features
                )
            else:
                features[f"dim_{dim}_mean_lifetime"] = 0.0
                features[f"dim_{dim}_std_lifetime"] = 0.0
                features[f"dim_{dim}_max_lifetime"] = 0.0
                features[f"dim_{dim}_sum_lifetime"] = 0.0
                features[f"dim_{dim}_entropy"] = 0.0

    except Exception as e:
        print(f"Error computing topological features: {e}")
        # Return default values
        for dim in range(max_dimension + 1):
            features[f"dim_{dim}_count"] = 0
            features[f"dim_{dim}_mean_lifetime"] = 0.0
            features[f"dim_{dim}_std_lifetime"] = 0.0
            features[f"dim_{dim}_max_lifetime"] = 0.0
            features[f"dim_{dim}_sum_lifetime"] = 0.0
            features[f"dim_{dim}_entropy"] = 0.0

    return features


def compute_distance_matrix_properties(distance_matrix: np.ndarray) -> Dict[str, float]:
    """
    Compute properties of a distance matrix.

    Args:
        distance_matrix: 2D distance matrix

    Returns:
        Dictionary of matrix properties
    """
    properties = {}

    # Basic statistics
    upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]

    properties["mean_distance"] = np.mean(upper_triangle)
    properties["std_distance"] = np.std(upper_triangle)
    properties["min_distance"] = np.min(upper_triangle)
    properties["max_distance"] = np.max(upper_triangle)
    properties["median_distance"] = np.median(upper_triangle)

    # Percentiles
    for p in [10, 25, 75, 90]:
        properties[f"distance_p{p}"] = np.percentile(upper_triangle, p)

    # Matrix properties
    properties["matrix_size"] = distance_matrix.shape[0]
    properties["matrix_density"] = np.count_nonzero(upper_triangle) / len(
        upper_triangle
    )

    # Condition number (for numerical stability assessment)
    try:
        eigenvals = np.linalg.eigvals(distance_matrix)
        eigenvals = eigenvals[eigenvals > 1e-10]  # Remove near-zero eigenvalues
        if len(eigenvals) > 0:
            properties["condition_number"] = np.max(eigenvals) / np.min(eigenvals)
        else:
            properties["condition_number"] = float("inf")
    except:
        properties["condition_number"] = float("inf")

    return properties


def evaluate_clustering_evolution(
    cluster_evolution: Dict[str, Dict], true_labels: Optional[np.ndarray] = None
) -> Dict[str, Dict]:
    """
    Evaluate clustering quality across different thresholds.

    Args:
        cluster_evolution: Cluster evolution data
        true_labels: Optional ground truth labels

    Returns:
        Dictionary of evaluation metrics for each threshold
    """
    evaluation = {}

    components_ = cluster_evolution.get("components_", {})
    labels_ = cluster_evolution.get("labels_", {})

    for distance_metric in components_.keys():
        evaluation[distance_metric] = {}

        thresholds = sorted([float(t) for t in components_[distance_metric].keys()])

        for threshold in thresholds:
            threshold_str = str(threshold)
            n_clusters = components_[distance_metric][threshold_str]
            cluster_labels = labels_[distance_metric][threshold_str]

            metrics = {"n_clusters": n_clusters, "threshold": threshold}

            # Basic cluster statistics
            if n_clusters > 0:
                unique_clusters, cluster_counts = np.unique(
                    cluster_labels, return_counts=True
                )
                metrics["mean_cluster_size"] = np.mean(cluster_counts)
                metrics["std_cluster_size"] = np.std(cluster_counts)
                metrics["max_cluster_size"] = np.max(cluster_counts)
                metrics["min_cluster_size"] = np.min(cluster_counts)
                metrics["cluster_size_entropy"] = entropy(
                    cluster_counts / np.sum(cluster_counts)
                )

            # External metrics if true labels available
            if true_labels is not None:
                metrics["purity"] = compute_purity(true_labels, cluster_labels)
                metrics["homogeneity"] = compute_homogeneity(
                    true_labels, cluster_labels
                )

                try:
                    metrics["adjusted_rand_score"] = adjusted_rand_score(
                        true_labels, cluster_labels
                    )
                    metrics["normalized_mutual_info"] = normalized_mutual_info_score(
                        true_labels, cluster_labels
                    )
                except:
                    metrics["adjusted_rand_score"] = 0.0
                    metrics["normalized_mutual_info"] = 0.0

            evaluation[distance_metric][threshold_str] = metrics

    return evaluation


def compute_robustness_metrics(
    baseline_features: Dict[str, float], perturbed_features: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute robustness metrics by comparing baseline and perturbed features.

    Args:
        baseline_features: Features from baseline (clean) condition
        perturbed_features: Features from perturbed condition

    Returns:
        Dictionary of robustness metrics
    """
    robustness = {}

    # Feature stability (relative changes)
    for key in baseline_features.keys():
        if key in perturbed_features:
            baseline_val = baseline_features[key]
            perturbed_val = perturbed_features[key]

            if baseline_val != 0:
                relative_change = abs(perturbed_val - baseline_val) / abs(baseline_val)
                robustness[f"{key}_relative_change"] = relative_change
            else:
                robustness[f"{key}_relative_change"] = (
                    float("inf") if perturbed_val != 0 else 0.0
                )

    # Overall stability score (mean relative change)
    relative_changes = [
        v
        for k, v in robustness.items()
        if k.endswith("_relative_change") and np.isfinite(v)
    ]
    if relative_changes:
        robustness["overall_stability"] = 1.0 / (1.0 + np.mean(relative_changes))
    else:
        robustness["overall_stability"] = 0.0

    return robustness


def summarize_metrics(metrics_dict: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """
    Summarize metrics across multiple conditions.

    Args:
        metrics_dict: Dictionary of {condition: {metric: value}}

    Returns:
        Summary statistics for each metric
    """
    summary = {}

    # Collect all metric names
    all_metrics = set()
    for condition_metrics in metrics_dict.values():
        all_metrics.update(condition_metrics.keys())

    # Compute summary statistics for each metric
    for metric in all_metrics:
        values = []
        for condition_metrics in metrics_dict.values():
            if metric in condition_metrics and np.isfinite(condition_metrics[metric]):
                values.append(condition_metrics[metric])

        if values:
            summary[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
                "count": len(values),
            }
        else:
            summary[metric] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "count": 0,
            }

    return summary
