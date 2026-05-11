"""
Minimum Spanning Tree (MST) processing utilities for topological data analysis.

This module provides tools for computing MSTs, analyzing connected components,
and performing various distance calculations on high-dimensional data.
"""

from typing import Tuple

import numpy as np
from loguru import logger
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm import tqdm

from ..config import DEFAULT_MST_THRESHOLD, DEFAULT_RANDOM_STATE


class MSTProcessor:
    """
    A class for processing data using Minimum Spanning Trees and various distance metrics.

    This class provides functionality for:
    - Computing minimum spanning trees from distance matrices
    - Filtering MSTs based on distance thresholds
    - Analyzing connected components
    - Computing various distance metrics (Euclidean, Mahalanobis, Cosine)
    - Density normalization of distance matrices
    """

    def __init__(self, threshold: float = DEFAULT_MST_THRESHOLD):
        """
        Initialize the MST processor.

        Args:
            threshold: Distance threshold for filtering MST edges. Defaults to
                ``hole.config.DEFAULT_MST_THRESHOLD``.

        Note: Construction no longer mutates the global ``np.random`` seed —
        callers needing reproducibility should set the seed themselves, or use
        the ``random_state`` parameters on the methods that consume it.
        """
        self.threshold = threshold

    def create_mst(self, X: np.ndarray, distance_matrix: bool = False, return_sparse: bool = False) -> np.ndarray:
        """
        Create minimum spanning tree from data or distance matrix.

        Args:
            X: Input data array or distance matrix
            distance_matrix: If True, X is treated as a distance matrix
            return_sparse: If True, returns sparse matrix (more memory efficient)

        Returns:
            MST adjacency matrix (sparse or dense based on return_sparse parameter)
        """
        if distance_matrix:
            dist_matrix = X
        else:
            dist_matrix = squareform(pdist(X))

        mst_sparse = minimum_spanning_tree(dist_matrix)
        
        # Log MST properties for verification
        n_nodes = dist_matrix.shape[0]
        n_edges = mst_sparse.nnz // 2  # Each edge counted twice in undirected graph
        expected_edges = n_nodes - 1
        
        logger.debug(f"MST created: {n_nodes} nodes, {n_edges} edges (expected: {expected_edges})")
        
        if n_edges != expected_edges:
            logger.warning(f"MST has {n_edges} edges, expected {expected_edges}. Graph may be disconnected.")
        
        if return_sparse:
            # Return sparse matrix for memory efficiency
            return mst_sparse
        else:
            # Return dense matrix for backward compatibility
            # Note: This can be memory intensive for large graphs
            if n_nodes > 1000:
                logger.warning(f"Converting sparse MST to dense matrix for {n_nodes} nodes. Consider using return_sparse=True for memory efficiency.")
            return mst_sparse.toarray()

    def filter_mst(self, mst: np.ndarray, threshold: float) -> np.ndarray:
        """
        Filter MST by removing edges above threshold.

        Args:
            mst: MST adjacency matrix
            threshold: Distance threshold for filtering

        Returns:
            Filtered MST adjacency matrix
        """
        filtered = mst.copy()
        filtered[filtered > threshold] = 0
        return filtered

    def ncomps_(self, mst: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Count connected components in MST.

        Args:
            mst: MST adjacency matrix

        Returns:
            Tuple of (number of components, component labels)
        """
        return connected_components(mst, directed=False)

    def pca_utils(self, X: np.ndarray, n_components: int = 50) -> np.ndarray:
        """
        Apply PCA dimensionality reduction.

        Uses the randomized SVD solver when projecting to far fewer
        components than the input has features — far cheaper on 768-dim+
        embeddings (BERT/ViT scale) than the default full SVD.

        Args:
            X: Input data array
            n_components: Number of principal components

        Returns:
            PCA-transformed data
        """
        n_components = min(n_components, X.shape[0] - 1, X.shape[1])

        # Randomized SVD is cheaper when output dim is much smaller than input
        # dim; sklearn's heuristic threshold is ~10x but we only need it for
        # the high-dim case the library actually targets.
        solver = "randomized" if X.shape[1] > 2 * n_components else "auto"
        pca = PCA(
            n_components=n_components,
            svd_solver=solver,
            random_state=DEFAULT_RANDOM_STATE,
        )
        return pca.fit_transform(X)

    @staticmethod
    def filter(persistence_, dists_matrices, k_deaths):
        """
        Filter persistence components based on distance matrices.

        Args:
            persistence_: Dictionary of persistence diagrams for each distance metric
            dists_matrices: Dictionary of distance matrices for each metric
            k_deaths: Limit the number of deaths that are being processed

        Returns:
            components_: Dictionary of connected components for each death threshold
            labels_: Dictionary of component labels for each death threshold
        """
        ### change the number of deaths that are being processed here
        components_ = {k: {} for k in persistence_}
        labels_ = {k: {} for k in persistence_}

        for k, v in persistence_.items():
            # Check if the key exists in dists_matrices
            if k not in dists_matrices:
                logger.warning(f"Key '{k}' not found in dists_matrices")
                continue

            persistence = v[
                :k_deaths
            ]  # Limit to first k_deaths for computational efficiency
            deaths = [death for _, (birth, death) in persistence if death != np.inf]

            # Get the distance matrix for this specific key
            dist_matrix = dists_matrices[k]

            logger.info(f"Processing {len(deaths)} death thresholds for {k}")

            n_points = dist_matrix.shape[0]

            for i in tqdm(range(len(deaths)), desc=f"Processing {k}"):
                death_threshold = deaths[i]

                # Build sparse adjacency directly — avoids the n^2 dense int
                # array that dominated memory at scale.
                mask = (dist_matrix <= death_threshold) & (~np.eye(n_points, dtype=bool))
                adj_sparse = csr_matrix(mask)

                n_components, cluster_labels = connected_components(
                    adj_sparse, directed=False
                )

                components_[k][f"{death_threshold}"] = n_components
                labels_[k][f"{death_threshold}"] = cluster_labels

        return components_, labels_

    def __call__(
        self, X: np.ndarray, distance_matrix: bool = False
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Complete MST processing pipeline.

        Args:
            X: Input data or distance matrix
            distance_matrix: If True, X is treated as distance matrix

        Returns:
            Tuple of (n_components, labels, filtered_mst)
        """
        mst_dense = self.create_mst(X, distance_matrix)
        filtered_mst = self.filter_mst(mst_dense, self.threshold)
        n_components, labels = self.ncomps_(filtered_mst)
        return n_components, labels, filtered_mst
