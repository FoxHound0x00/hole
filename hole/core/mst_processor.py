"""
Minimum Spanning Tree (MST) processing utilities for topological data analysis.

This module provides tools for computing MSTs, analyzing connected components,
and performing various distance calculations on high-dimensional data.
"""

from typing import Tuple

import networkx as nx
import numpy as np
from loguru import logger
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm import tqdm


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

    def __init__(self, threshold: float = 35):
        """
        Initialize the MST processor.

        Args:
            threshold: Distance threshold for filtering MST edges
        """
        np.random.seed(42)  # For reproducibility
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

        Args:
            X: Input data array
            n_components: Number of principal components

        Returns:
            PCA-transformed data
        """
        # Ensure n_components doesn't exceed data dimensions
        n_components = min(n_components, X.shape[0] - 1, X.shape[1])

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return X_pca

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

            for i in tqdm(range(len(deaths)), desc=f"Processing {k}"):
                death_threshold = deaths[i]

                # Create adjacency matrix for this threshold
                adj_matrix = (dist_matrix <= death_threshold).astype(int)
                np.fill_diagonal(adj_matrix, 0)  # Remove self-loops

                # Find connected components using NetworkX
                G = nx.from_numpy_array(adj_matrix)
                conn_comp = list(nx.connected_components(G))

                # Create cluster labels
                n_points = dist_matrix.shape[0]
                cluster_labels = np.zeros(n_points, dtype=int)
                for cluster_id, component in enumerate(conn_comp):
                    for node in component:
                        cluster_labels[node] = cluster_id

                # Store results
                components_[k][f"{death_threshold}"] = len(conn_comp)
                labels_[k][f"{death_threshold}"] = cluster_labels

                # Optional: Use MST processor for additional analysis
                try:
                    mst_obj = MSTProcessor(threshold=death_threshold)
                    n_components, mst_labels, filtered_mst = mst_obj(
                        X=dist_matrix, distance_matrix=True
                    )
                    # Could store additional MST-based analysis here if needed
                except Exception as e:
                    logger.error(f"MST processing failed for threshold {death_threshold}: {e}")

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
