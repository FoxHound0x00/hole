import os
from collections import defaultdict, deque

import gudhi as gd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.spatial.distance import squareform
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

# Import distance_matrix from core
from ..core.distance_metrics import distance_matrix


class UnionFind:
    """Union-Find data structure to track cluster merges."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


class PersistenceDendrogram:
    def __init__(self, distance_matrix=None, points=None):
        """
        Initialize with either a distance matrix or points.
        If points are provided, we'll compute the distance matrix.
        """
        if distance_matrix is not None:
            self.distance_matrix = distance_matrix
            self.n_points = distance_matrix.shape[0]
        elif points is not None:
            self.points = points
            self.distance_matrix = pairwise_distances(points)
            self.n_points = len(points)
        else:
            raise ValueError("Must provide either distance_matrix or points")

        self.persistence = None
        self.death_thresholds = None
        self.linkage_matrix = None

    def compute_persistence(self, max_dimension=1):
        """Compute persistence homology using GUDHI Rips complex."""
        print(f"Computing persistence homology for {self.n_points} points...")

        # Create Rips complex
        rips_complex = gd.RipsComplex(distance_matrix=self.distance_matrix)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

        # Compute persistence
        self.persistence = simplex_tree.persistence()

        # Extract death thresholds for 0-dimensional features (connected components)
        # These represent when clusters merge
        death_thresholds = []
        for dim, (birth, death) in self.persistence:
            if dim == 0 and death != float("inf"):  # Connected components that die
                death_thresholds.append(death)

        # Sort death thresholds
        self.death_thresholds = sorted(set(death_thresholds))
        print(f"Found {len(self.death_thresholds)} unique death thresholds")

        return self.persistence

    def _compute_rcm_reordering(self, fallback_order=None):
        """
        Compute RCM (Reverse Cuthill-McKee) reordering indices for the distance matrix.

        Parameters
        ----------
        fallback_order : array-like, optional
            Fallback ordering to use if RCM fails. If None, uses original order.

        Returns
        -------
        reorder_indices : np.ndarray
            Indices for reordering the distance matrix
        title_suffix : str
            Suffix to add to plot titles indicating reordering status
        """
        try:
            # Create sparse matrix from distance matrix (use inverse distances for connectivity)
            # RCM works on connectivity, so we threshold the distance matrix
            threshold = np.percentile(
                self.distance_matrix, 20
            )  # Connect closest 20% of pairs
            adjacency = (self.distance_matrix <= threshold).astype(int)
            np.fill_diagonal(adjacency, 0)  # Remove self-loops

            sparse_adj = csr_matrix(adjacency)
            rcm_order = reverse_cuthill_mckee(sparse_adj)

            # Use RCM ordering if it gives good results
            if len(rcm_order) == self.distance_matrix.shape[0]:
                return rcm_order, " (RCM Reordered)"
            else:
                # Fall back to provided order or original order
                fallback = (
                    fallback_order
                    if fallback_order is not None
                    else np.arange(self.distance_matrix.shape[0])
                )
                return fallback, " (No Reordering)"

        except Exception as e:
            fallback_name = "dendrogram" if fallback_order is not None else "original"
            print(f"RCM reordering failed: {e}, using {fallback_name} ordering")
            fallback = (
                fallback_order
                if fallback_order is not None
                else np.arange(self.distance_matrix.shape[0])
            )
            return fallback, " (No Reordering)"

    def _plot_reordered_heatmap(self, ax, reorder_indices, cmap="viridis", labels=None):
        """
        Plot heatmap with reordered distance matrix.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        reorder_indices : array-like
            Indices for reordering the distance matrix
        cmap : str, optional
            Colormap for the heatmap
        labels : list, optional
            Labels for the data points

        Returns
        -------
        im : matplotlib.image.AxesImage
            The image object from imshow
        """
        # Reorder distance matrix
        reordered_dist_matrix = self.distance_matrix[
            np.ix_(reorder_indices, reorder_indices)
        ]

        # Plot heatmap
        im = ax.imshow(
            reordered_dist_matrix, aspect="auto", cmap=cmap, interpolation="nearest"
        )

        # Add labels if not too many
        if labels and len(labels) <= 30:
            reordered_labels = [labels[i] for i in reorder_indices]
            ax.set_xticks(range(len(reordered_labels)))
            ax.set_yticks(range(len(reordered_labels)))
            ax.set_xticklabels(reordered_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(reordered_labels, fontsize=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        return im

    def _add_colorbar(self, fig, ax, im, label="Distance"):
        """
        Add colorbar to heatmap.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object
        im : matplotlib.image.AxesImage
            Image object from imshow
        label : str, optional
            Label for the colorbar

        Returns
        -------
        cbar : matplotlib.colorbar.Colorbar
            The colorbar object
        """
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(label, rotation=270, labelpad=15)
        return cbar

    def build_linkage_matrix_from_persistence(self):
        """Build linkage matrix directly from persistence homology death thresholds."""
        if self.persistence is None:
            self.compute_persistence()

        print("Building linkage matrix from persistence...")

        # Get all edges (pairs of points) with their distances
        edges = []
        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                edges.append((self.distance_matrix[i, j], i, j))

        # Sort edges by distance
        edges.sort()

        # Track cluster mappings
        # cluster_map[old_id] = current_id
        cluster_map = {
            i: i for i in range(self.n_points)
        }  # Initially each point is its own cluster
        cluster_sizes = {i: 1 for i in range(self.n_points)}
        next_cluster_id = self.n_points

        linkage = []

        # Process edges in order of increasing distance
        for distance, i, j in edges:
            # Get current cluster IDs for points i and j
            cluster_i = cluster_map[i]
            cluster_j = cluster_map[j]

            # If they are in different clusters, merge them
            if cluster_i != cluster_j:
                # Get sizes
                size_i = cluster_sizes[cluster_i]
                size_j = cluster_sizes[cluster_j]

                # Create linkage entry: [cluster1, cluster2, distance, new_size]
                linkage.append([cluster_i, cluster_j, distance, size_i + size_j])

                # Update all points in both clusters to point to the new cluster
                for point_id in range(self.n_points):
                    if (
                        cluster_map[point_id] == cluster_i
                        or cluster_map[point_id] == cluster_j
                    ):
                        cluster_map[point_id] = next_cluster_id

                # Store new cluster size
                cluster_sizes[next_cluster_id] = size_i + size_j

                # Clean up old cluster sizes
                if cluster_i in cluster_sizes:
                    del cluster_sizes[cluster_i]
                if cluster_j in cluster_sizes:
                    del cluster_sizes[cluster_j]

                next_cluster_id += 1

                # If we have made n-1 merges, we're done
                if len(linkage) == self.n_points - 1:
                    break

        self.linkage_matrix = (
            np.array(linkage) if linkage else self._create_simple_linkage()
        )
        return self.linkage_matrix

    def _create_simple_linkage(self):
        """Create a simple linkage matrix when other methods fail."""
        print("Creating simple linkage matrix...")

        linkage = []
        remaining_clusters = list(range(self.n_points))
        cluster_sizes = [1] * self.n_points
        next_id = self.n_points

        while len(remaining_clusters) > 1:
            # Find the two closest clusters
            min_dist = float("inf")
            best_pair = None

            for i in range(len(remaining_clusters)):
                for j in range(i + 1, len(remaining_clusters)):
                    c1, c2 = remaining_clusters[i], remaining_clusters[j]
                    if c1 < self.n_points and c2 < self.n_points:
                        dist = self.distance_matrix[c1, c2]
                    else:
                        # For merged clusters, use a reasonable distance
                        dist = min_dist + 0.1

                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (i, j, c1, c2)

            if best_pair:
                i, j, c1, c2 = best_pair
                new_size = cluster_sizes[c1] + cluster_sizes[c2]

                linkage.append([c1, c2, min_dist, new_size])

                # Remove merged clusters and add new one
                remaining_clusters.remove(c1)
                remaining_clusters.remove(c2)
                remaining_clusters.append(next_id)

                cluster_sizes.append(new_size)
                next_id += 1

        return np.array(linkage) if linkage else np.array([[0, 1, 1.0, 2]])

    def plot_dendrogram(
        self, labels=None, title="Persistence Dendrogram", figsize=(12, 8)
    ):
        """Plot the dendrogram (simple version without heatmap)."""
        if self.linkage_matrix is None:
            self.build_linkage_matrix_from_persistence()

        plt.figure(figsize=figsize)

        # Create dendrogram
        dendrogram_result = dendrogram(
            self.linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=10
        )

        plt.title(title)
        plt.xlabel("Data Points")
        plt.ylabel("Distance (Death Threshold)")
        plt.tight_layout()

        return dendrogram_result

    def plot_dendrogram_with_heatmap(
        self,
        labels=None,
        # title="Persistence Dendrogram",
        title=None,
        figsize=(16, 8),
        cmap="viridis",
    ):
        """Plot dendrogram with distance matrix heatmap using RCM reordering."""
        if self.linkage_matrix is None:
            self.build_linkage_matrix_from_persistence()

        fig, (ax_dendro, ax_heatmap) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1]}
        )

        # Compute dendrogram for ordering
        dendro_result = dendrogram(
            self.linkage_matrix,
            labels=labels,
            ax=ax_dendro,
            orientation="left",
            leaf_font_size=8,
            no_labels=True if len(labels or []) > 50 else False,
        )
        ax_dendro.set_title("Dendrogram")
        ax_dendro.set_xlabel("Distance")
        ax_dendro.set_yticks([])

        # Get dendrogram ordering
        dendro_order = dendro_result["leaves"]

        # Apply RCM reordering with dendrogram as fallback
        reorder_indices, _ = self._compute_rcm_reordering(fallback_order=dendro_order)

        # Plot heatmap using helper method
        im = self._plot_reordered_heatmap(
            ax_heatmap, reorder_indices, cmap=cmap, labels=labels
        )

        ax_heatmap.set_title("Distance Matrix (RCM Reordered)")
        ax_heatmap.set_xlabel("Data Points")
        ax_heatmap.set_ylabel("Data Points")

        # Add colorbar using helper method
        self._add_colorbar(fig, ax_heatmap, im)

        # Set main title
        fig.suptitle(title, fontsize=14, y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Add space for main title
        return fig, dendro_result

    def plot_rcm_heatmap(
        self,
        labels=None,
        title="Distance Matrix (RCM Reordered)",
        figsize=(10, 8),
        cmap="viridis",
        ax=None,
    ):
        """
        Plot standalone distance matrix heatmap using RCM reordering.

        Parameters
        ----------
        labels : list, optional
            Labels for the data points
        title : str, optional
            Title for the heatmap
        figsize : tuple, optional
            Figure size (width, height)
        cmap : str, optional
            Colormap for the heatmap
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the heatmap
        ax : matplotlib.axes.Axes
            The axes object containing the heatmap
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = ax.figure

        # Apply RCM reordering
        reorder_indices, title_suffix = self._compute_rcm_reordering()

        # Plot heatmap using helper method
        im = self._plot_reordered_heatmap(ax, reorder_indices, cmap=cmap, labels=labels)

        ax.set_title(title + title_suffix)
        ax.set_xlabel("Data Points")
        ax.set_ylabel("Data Points")

        # Add colorbar using helper method
        self._add_colorbar(fig, ax, im)

        plt.tight_layout()
        return fig, ax

    def analyze_cluster_evolution(self):
        """Analyze how clusters evolve at different death thresholds."""
        if self.persistence is None:
            self.compute_persistence()

        print("Analyzing cluster evolution...")

        # Track cluster assignments at different thresholds
        cluster_evolution = {}

        # Test at various thresholds
        test_thresholds = (
            [0.0] + self.death_thresholds[:10] + [max(self.death_thresholds) * 2]
        )

        for threshold in test_thresholds:
            # Create graph with edges where distance <= threshold
            graph = nx.Graph()
            graph.add_nodes_from(range(self.n_points))

            for i in range(self.n_points):
                for j in range(i + 1, self.n_points):
                    if self.distance_matrix[i, j] <= threshold:
                        graph.add_edge(i, j)

            # Find connected components
            components = list(nx.connected_components(graph))

            # Create cluster labels
            cluster_labels = np.zeros(self.n_points, dtype=int)
            for cluster_id, component in enumerate(components):
                for node in component:
                    cluster_labels[node] = cluster_id

            cluster_evolution[threshold] = {
                "labels": cluster_labels,
                "n_clusters": len(components),
                "components": components,
            }

        return cluster_evolution


def analyze_activation_persistence(
    activation_file, output_dir, model_name, condition_name, max_points=100
):
    """
    Analyze ViT activations using persistence dendrograms for all 5 distance metrics.

    Args:
        activation_file: Path to the .npy file containing activations
        output_dir: Directory to save results
        model_name: Name of the model (e.g., 'global_pruned_30pct')
        condition_name: Name of the condition (e.g., 'gaussian', 'inference')
        max_points: Maximum number of points to use for analysis (for speed)
    """
    print(f"Analyzing persistence for {model_name} - {condition_name}")

    # Load activations
    try:
        all_activations = np.load(activation_file, allow_pickle=True).item()
        if not isinstance(all_activations, dict):
            print(f"Warning: Expected dictionary, got {type(all_activations)}")
            return
    except Exception as e:
        print(f"Error loading {activation_file}: {e}")
        return

    # Import MST processor for distance calculations
    from mst_proc import MSTProcessor

    from hole.visualization.pers_bcd import distance_matrix

    # Process each layer
    for layer_name, activation_data in all_activations.items():
        print(f"  Processing layer: {layer_name}")

        # Handle different activation shapes
        if len(activation_data.shape) == 3:
            # [batch_size, seq_len, hidden_dim] - use class token
            pc = activation_data[:, 0, :]
        elif len(activation_data.shape) == 2:
            # [batch_size, hidden_dim] - already flattened
            pc = activation_data
        else:
            print(f"    Warning: Unexpected shape {activation_data.shape}, skipping...")
            continue

        # Subsample if too many points
        if pc.shape[0] > max_points:
            indices = np.random.choice(pc.shape[0], max_points, replace=False)
            pc = pc[indices]
            print(f"    Subsampled to {max_points} points")

        # Clean layer name for filename
        clean_layer_name = layer_name.replace("/", "_").replace(".", "_")

        # Initialize MST processor
        mst_obj = MSTProcessor()

        try:
            # Compute all 5 distance matrices
            X_pca = mst_obj.pca_utils(pc)
            euclid_dist = distance_matrix(pc)
            maha_dist = mst_obj.fast_maha(X_pca)
            density_maha = mst_obj.density_normalizer(X=X_pca, dists=maha_dist, k=5)
            density_euclid = mst_obj.density_normalizer(X=X_pca, dists=euclid_dist, k=5)
            dists_cosine = mst_obj.cosine_gen(pc)

            # Store distance matrices
            dists_matrices = {
                "Mahalanobis": maha_dist,
                "Euclidean": euclid_dist,
                "Density_Normalized_Mahalanobis": density_maha,
                "Density_Normalized_Euclidean": density_euclid,
                "Cosine": dists_cosine,
            }

            # Process each distance metric
            for dist_name, dist_matrix in dists_matrices.items():
                print(f"    Processing {dist_name} distance metric...")

                # Create persistence dendrogram for this distance metric
                try:
                    pd = PersistenceDendrogram(distance_matrix=dist_matrix)
                    pd.compute_persistence()

                    # Create point labels
                    labels = [f"P{i}" for i in range(len(pc))]

                    # Create output directory for this distance metric
                    metric_output_dir = (
                        f"{output_dir}/{model_name}_{condition_name}_{dist_name}"
                    )
                    os.makedirs(metric_output_dir, exist_ok=True)

                    # Model name for titles
                    if condition_name.lower() in ["inference", "clean"]:
                        if model_name.lower() == "original":
                            title_prefix = "Model"
                        else:
                            title_prefix = f"{model_name.replace('_', ' ').title()}"
                    else:
                        if model_name.lower() == "original":
                            title_prefix = (
                                f"Model - {condition_name.replace('_', ' ').title()}"
                            )
                        else:
                            title_prefix = f"{model_name.replace('_', ' ').title()} - {condition_name.replace('_', ' ').title()}"

                    # Generate dendrogram WITH RCM-reordered heatmap
                    pd.plot_dendrogram_with_heatmap(
                        labels=labels,
                        title=f"ViT {title_prefix} - {dist_name} - {layer_name}",
                        figsize=(16, 8),
                        cmap="viridis",
                    )

                    # Save dendrogram+heatmap version
                    output_file = f"{metric_output_dir}/ViT_{model_name}_{condition_name}_{clean_layer_name}_{dist_name}_dendrogram_rcm_heatmap.png"
                    plt.savefig(output_file, dpi=300, bbox_inches="tight")
                    plt.close()

                    print(f"      Saved: {output_file}")

                except Exception as e:
                    print(f"      Error processing {dist_name} for {layer_name}: {e}")
                    continue

        except Exception as e:
            print(f"    Error processing layer {layer_name}: {e}")
            continue


def run_persistence_analysis_on_results(
    results_dir="results_compression", max_points=100
):
    """
    Run persistence dendrogram analysis on all activation files in results directory.

    Args:
        results_dir: Directory containing model results
        max_points: Maximum points per analysis (for computational efficiency)
    """
    print(f"Running persistence analysis on {results_dir}...")

    persistence_output_dir = f"{results_dir}/persistence_dendrograms"
    os.makedirs(persistence_output_dir, exist_ok=True)

    # Process each model directory
    for model_name in os.listdir(results_dir):
        model_path = f"{results_dir}/{model_name}"
        if os.path.isdir(model_path) and model_name not in [
            "visualizations",
            "model_stats",
            "tda_analysis",
            "persistence_dendrograms",
        ]:
            activations_dir = f"{model_path}/activations"
            if os.path.exists(activations_dir):
                print(f"Processing {model_name}...")

                # Process each activation file
                activation_files = [
                    f
                    for f in os.listdir(activations_dir)
                    if f.endswith("_all_layers.npy")
                ]

                for activation_file in activation_files:
                    condition_name = activation_file.replace("_all_layers.npy", "")
                    activation_path = f"{activations_dir}/{activation_file}"

                    analyze_activation_persistence(
                        activation_path,
                        persistence_output_dir,
                        model_name,
                        condition_name,
                        max_points,
                    )


if __name__ == "__main__":
    print("PersistenceDendrogram: Persistence homology-based hierarchical clustering")
    print("Usage:")
    print("  from vis.persistence_dendrogram import PersistenceDendrogram")
    print("  pd = PersistenceDendrogram(points=your_data)")
    print("  pd.compute_persistence()")
    print("  pd.plot_dendrogram_with_heatmap()")
    print("\nTo run tests: make test-persistence")
