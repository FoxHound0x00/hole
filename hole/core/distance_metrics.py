"""
Distance metrics for topological data analysis.

This module provides optimized distance metrics for HOLE,
including Euclidean, Manhattan, Chebyshev, Cosine, and Mahalanobis variants.
"""

from typing import Callable, Optional, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances between points.

    Parameters
    ----------
    a : np.ndarray
        First set of points
    b : np.ndarray
        Second set of points

    Returns
    -------
    np.ndarray
        Euclidean distances
    """
    diff = a - b
    ssd = np.sum(diff**2, axis=1)
    return np.sqrt(ssd)


def manhattan(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute Manhattan (L1) distances between points.

    Parameters
    ----------
    a : np.ndarray
        First set of points
    b : np.ndarray
        Second set of points

    Returns
    -------
    np.ndarray
        Manhattan distances
    """
    diff = np.abs(a - b)
    return np.sum(diff, axis=1)


def chebyshev(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev (L∞) distances between points.

    Parameters
    ----------
    a : np.ndarray
        First set of points
    b : np.ndarray
        Second set of points

    Returns
    -------
    np.ndarray
        Chebyshev distances
    """
    diff = np.abs(a - b)
    return np.max(diff, axis=1)


def euclidean_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance matrix efficiently.

    Args:
        X: Input data array of shape (n_samples, n_features)

    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples)
    """
    n_points, n_features = X.shape
    i, j = np.triu_indices(n_points, k=1)  # Upper triangular indices without diagonal

    a = X[i]  # Points for upper triangular computation
    b = X[j]  # Points for upper triangular computation

    upper_triangle_distances = euclidean(a, b)

    # Create symmetric distance matrix
    distance_mat = np.zeros((n_points, n_points))
    distance_mat[i, j] = upper_triangle_distances
    distance_mat = distance_mat + distance_mat.T  # Make symmetric

    return distance_mat


def manhattan_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute Manhattan distance matrix efficiently.

    Args:
        X: Input data array of shape (n_samples, n_features)

    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples)
    """
    n_points, n_features = X.shape
    i, j = np.triu_indices(n_points, k=1)

    a = X[i]
    b = X[j]

    upper_triangle_distances = manhattan(a, b)

    distance_mat = np.zeros((n_points, n_points))
    distance_mat[i, j] = upper_triangle_distances
    distance_mat = distance_mat + distance_mat.T

    return distance_mat


def chebyshev_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute Chebyshev distance matrix efficiently.

    Args:
        X: Input data array of shape (n_samples, n_features)

    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples)
    """
    n_points, n_features = X.shape
    i, j = np.triu_indices(n_points, k=1)

    a = X[i]
    b = X[j]

    upper_triangle_distances = chebyshev(a, b)

    distance_mat = np.zeros((n_points, n_points))
    distance_mat[i, j] = upper_triangle_distances
    distance_mat = distance_mat + distance_mat.T

    return distance_mat


def cosine_distance(X: np.ndarray) -> np.ndarray:
    """
    Compute cosine distance matrix.

    Args:
        X: Input data array of shape (n_samples, n_features)

    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples)
    """
    distances = pdist(X, metric="cosine")
    return squareform(distances)


def mahalanobis_distance(
    X: np.ndarray,
    cov_inv: Optional[np.ndarray] = None,
    pca_components: int = 10,
) -> np.ndarray:
    """
    Compute Mahalanobis distance matrix with optional PCA preprocessing.

    Args:
        X: Input data array of shape (n_samples, n_features)
        cov_inv: Inverse covariance matrix. If None, computed from data
        pca_components: Number of PCA components for preprocessing

    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples)
    """
    # Apply PCA if data is high-dimensional or if we have too few samples
    # Use fewer components to avoid numerical instability
    n_samples, n_features = X.shape
    max_components = min(pca_components, n_samples - 2, n_features)

    if n_features > max_components or n_samples < n_features * 2:
        if max_components < 2:
            # Too few samples, fallback to Euclidean
            print("Warning: Too few samples for Mahalanobis distance, using Euclidean")
            return euclidean_distance(X)

        pca = PCA(n_components=max_components)
        X_processed = pca.fit_transform(X)
    else:
        X_processed = X.copy()

    # Compute covariance matrix and its inverse with better regularization
    if cov_inv is None:
        cov_matrix = np.cov(X_processed.T)

        # Add stronger regularization based on data scale
        reg_factor = max(1e-6, np.trace(cov_matrix) / cov_matrix.shape[0] * 1e-3)
        cov_matrix += np.eye(cov_matrix.shape[0]) * reg_factor

        # Check condition number
        try:
            cond_num = np.linalg.cond(cov_matrix)
            if cond_num > 1e12:
                print(
                    f"Warning: Covariance matrix is poorly conditioned (cond={cond_num:.2e}), using regularization"
                )
                cov_matrix += np.eye(cov_matrix.shape[0]) * reg_factor * 100

            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse with stronger regularization
            print("Warning: Using pseudo-inverse for covariance matrix")
            cov_inv = np.linalg.pinv(cov_matrix, rcond=1e-10)

    try:
        # Use chunked computation for large datasets to avoid memory issues
        if n_samples > 1000:
            print(
                "Warning: Large dataset detected, using chunked Mahalanobis computation"
            )
            return _chunked_mahalanobis_distance(X_processed, cov_inv)
        else:
            distances = pdist(X_processed, metric="mahalanobis", VI=cov_inv)
            return squareform(distances)
    except (np.linalg.LinAlgError, ValueError, MemoryError) as e:
        # Fallback to Euclidean distance
        print(
            f"Warning: Mahalanobis distance computation failed ({e}), falling back to Euclidean"
        )
        return euclidean_distance(X_processed)


def _chunked_mahalanobis_distance(
    X: np.ndarray, cov_inv: np.ndarray, chunk_size: int = 100
) -> np.ndarray:
    """
    Compute Mahalanobis distance matrix in chunks to avoid memory issues.
    """
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))

    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        for j in range(i, n_samples, chunk_size):
            end_j = min(j + chunk_size, n_samples)

            # Compute distances for this chunk
            chunk_i = X[i:end_i]
            chunk_j = X[j:end_j]

            # Compute pairwise distances manually
            for ii, x_i in enumerate(chunk_i):
                for jj, x_j in enumerate(chunk_j):
                    if i + ii <= j + jj:  # Only compute upper triangle
                        diff = x_i - x_j
                        dist = np.sqrt(diff.T @ cov_inv @ diff)
                        distance_matrix[i + ii, j + jj] = dist
                        if i + ii != j + jj:  # Make symmetric
                            distance_matrix[j + jj, i + ii] = dist

    return distance_matrix


def floyd_warshall(dist_matrix):
    n = dist_matrix.shape[0]
    dist = dist_matrix.copy()
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])
    return dist


def geodesic_distances(X, k=10):
    D = euclidean_distance(X)
    n = D.shape[0]

    # Build sparse kNN graph manually
    knn_graph = np.full_like(D, np.inf)
    for i in range(n):
        idx = np.argsort(D[i])[1 : k + 1]  # Exclude self
        knn_graph[i, idx] = D[i, idx]

    # Symmetrize
    knn_graph = np.minimum(knn_graph, knn_graph.T)

    # Floyd-Warshall for shortest paths
    geo_dist = floyd_warshall(knn_graph)
    return geo_dist


def distance_matrix(
    points: np.ndarray, metric: Union[str, Callable] = "euclidean"
) -> np.ndarray:
    """
    Compute distance matrix for a set of points using optimized implementations.

    Parameters
    ----------
    points : np.ndarray
        Points array of shape (n_points, n_features)
    metric : str or callable, optional
        Distance metric to use. Can be 'euclidean', 'manhattan', 'chebyshev',
        'cosine', 'mahalanobis', or a callable. Default is 'euclidean'.

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_points, n_points)
    """
    if isinstance(metric, str):
        if metric == "euclidean":
            return euclidean_distance(points)
        elif metric == "manhattan":
            return manhattan_distance(points)
        elif metric == "chebyshev":
            return chebyshev_distance(points)
        elif metric == "cosine":
            return cosine_distance(points)
        elif metric == "mahalanobis":
            return mahalanobis_distance(points)
        else:
            # Use sklearn for other metrics
            return pairwise_distances(points, metric=metric)
    else:
        # Use sklearn for callable metrics
        return pairwise_distances(points, metric=metric)


def density_normalized_distance(
    X: np.ndarray, dists: np.ndarray, k: int = 5
) -> np.ndarray:
    """
    Apply density normalization to a distance matrix.

    This normalizes distances by the k-th nearest neighbor distance,
    making the metric adaptive to local density variations.

    Args:
        X: Input data array (used for shape reference)
        dists: Distance matrix to normalize
        k: Number of nearest neighbors for normalization

    Returns:
        Density-normalized distance matrix
    """
    n_points = dists.shape[0]

    # For each point, find its k-th nearest neighbor distance
    # Sort each row and get the k-th smallest distance (excluding self)
    sorted_dists = np.sort(dists, axis=1)
    kth_neighbor_dists = sorted_dists[
        :, k
    ]  # k-th nearest neighbor (k=1 means 1st neighbor, etc.)

    # Avoid division by very small numbers that cause explosive normalization
    # Use a conservative threshold - at least 1% of the median k-th distance or 0.01, whichever is larger
    median_kth_dist = np.median(kth_neighbor_dists)
    min_threshold = max(0.01, median_kth_dist * 0.1)
    kth_neighbor_dists = np.maximum(kth_neighbor_dists, min_threshold)

    # Create normalized distance matrix
    norm_dist = np.zeros_like(dists)

    # For each pair (i,j), normalize by the geometric mean of their k-th neighbor distances
    # This makes the normalization symmetric
    for i in range(n_points):
        for j in range(i + 1, n_points):
            # Geometric mean of the k-th neighbor distances
            norm_factor = np.sqrt(kth_neighbor_dists[i] * kth_neighbor_dists[j])
            normalized_distance = dists[i, j] / norm_factor
            norm_dist[i, j] = normalized_distance
            norm_dist[j, i] = normalized_distance  # Symmetric

    return norm_dist


def validate_distance_matrix(dist_matrix: np.ndarray) -> bool:
    """
    Validate that a matrix is a proper distance matrix.

    Args:
        dist_matrix: Matrix to validate

    Returns:
        True if valid distance matrix, False otherwise
    """
    # Check if square matrix
    if dist_matrix.shape[0] != dist_matrix.shape[1]:
        return False

    # Check if symmetric (within tolerance)
    if not np.allclose(dist_matrix, dist_matrix.T, rtol=1e-10):
        return False

    # Check if diagonal is zero
    if not np.allclose(np.diag(dist_matrix), 0, atol=1e-10):
        return False

    # Check if non-negative
    if np.any(dist_matrix < 0):
        return False

    return True
