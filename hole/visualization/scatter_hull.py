"""
Blob Visualization for Cluster Separation Analysis

This module provides visualization of cluster separation at specific distance thresholds
from persistent homology cluster evolution. Shows t-SNE, PCA, and MDS plots with 
nodes colored by true labels and convex hulls around cluster assignments.
"""

import os
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from scipy.spatial import distance_matrix as scipy_distance_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA


def generate_consistent_colors(n_colors: int = 20, include_noise: bool = True) -> List:
    """
    Generate consistent discrete colors for use across different visualizations.

    Args:
        n_colors: Number of colors to generate
        include_noise: Whether to include gray color for noise (-1)

    Returns:
        List of RGBA color tuples
    """
    colors = []

    # Use multiple discrete colormaps for better variety
    discrete_colormaps = [
        plt.cm.tab10,  # 10 distinct colors
        plt.cm.tab20,  # 20 distinct colors
        plt.cm.tab20b,  # 20 more distinct colors
        plt.cm.tab20c,  # 20 more distinct colors
        plt.cm.Set1,  # 9 distinct colors
        plt.cm.Set2,  # 8 distinct colors
        plt.cm.Set3,  # 12 distinct colors
        plt.cm.Paired,  # 12 paired colors
    ]

    # Add gray for noise if requested
    if include_noise:
        colors.append((0.5, 0.5, 0.5, 1.0))
        n_colors -= 1

    # Fill colors from discrete colormaps
    cmap_idx = 0
    while len(colors) < n_colors + (1 if include_noise else 0) and cmap_idx < len(
        discrete_colormaps
    ):
        cmap = discrete_colormaps[cmap_idx]

        # Get the number of colors in this colormap
        if hasattr(cmap, "N"):
            n_cmap_colors = cmap.N
        else:
            n_cmap_colors = 20  # Default for tab-like maps

        # Sample discrete colors from the colormap
        for i in range(
            min(n_cmap_colors, n_colors - len(colors) + (1 if include_noise else 0))
        ):
            color = cmap(i / max(1, n_cmap_colors - 1))
            colors.append(color)

        cmap_idx += 1

    # If we still need more colors, use HSV generation
    if len(colors) < n_colors + (1 if include_noise else 0):
        golden_ratio = 0.618033988749895
        start_idx = len(colors) - (1 if include_noise else 0)
        for i in range(start_idx, n_colors):
            hue = (i * golden_ratio) % 1.0
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 2) * 0.1

            rgb = mcolors.hsv_to_rgb([hue, saturation, value])
            rgba = (*rgb, 1.0)
            colors.append(rgba)

    return colors[: n_colors + (1 if include_noise else 0)]


class BlobVisualizer:
    """
    Visualizes cluster separation at specific persistent homology thresholds.
    Creates t-SNE, PCA, and MDS plots with nodes colored by true labels
    and convex hulls around cluster assignments.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 300,
        alpha_hull: float = 0.3,
        class_names: Optional[Dict[int, str]] = None,
        shared_cluster_colors: Optional[List] = None,
    ):
        """
        Initialize the blob visualizer.

        Args:
            figsize: Figure size for plots
            dpi: DPI for saved plots
            alpha_hull: Alpha transparency for convex hulls
            class_names: Optional dictionary mapping class indices to names
            shared_cluster_colors: Optional shared color list for consistency across visualizations
        """
        self.figsize = figsize
        self.dpi = dpi
        self.alpha_hull = alpha_hull

        # Class names - use provided or generic defaults
        self.class_names = class_names or {i: f"Class_{i}" for i in range(10)}

        # Use shared colors if provided, otherwise generate new ones
        if shared_cluster_colors is not None:
            self.cluster_colors = shared_cluster_colors
        else:
            self.cluster_colors = self._generate_cluster_colors()

    def _generate_cluster_colors(self, n_colors: int = 20) -> List:
        """Generate distinct colors for cluster hulls using discrete colormaps."""
        colors = []

        # Use multiple discrete colormaps for better variety
        discrete_colormaps = [
            plt.cm.tab10,  # 10 distinct colors
            plt.cm.tab20,  # 20 distinct colors
            plt.cm.tab20b,  # 20 more distinct colors
            plt.cm.tab20c,  # 20 more distinct colors
            plt.cm.Set1,  # 9 distinct colors
            plt.cm.Set2,  # 8 distinct colors
            plt.cm.Set3,  # 12 distinct colors
            plt.cm.Paired,  # 12 paired colors
        ]

        # Fill colors from discrete colormaps
        cmap_idx = 0
        while len(colors) < n_colors and cmap_idx < len(discrete_colormaps):
            cmap = discrete_colormaps[cmap_idx]

            # Get the number of colors in this colormap
            if hasattr(cmap, "N"):
                n_cmap_colors = cmap.N
            else:
                n_cmap_colors = 20  # Default for tab-like maps

            # Sample discrete colors from the colormap
            for i in range(min(n_cmap_colors, n_colors - len(colors))):
                color = cmap(i / max(1, n_cmap_colors - 1))
                colors.append(color)

            cmap_idx += 1

        # If we still need more colors, use HSV generation
        if len(colors) < n_colors:
            golden_ratio = 0.618033988749895
            for i in range(len(colors), n_colors):
                hue = (i * golden_ratio) % 1.0
                saturation = 0.7 + (i % 3) * 0.1
                value = 0.8 + (i % 2) * 0.1

                rgb = mcolors.hsv_to_rgb([hue, saturation, value])
                rgba = (*rgb, 1.0)
                colors.append(rgba)

        return colors[:n_colors]

    def _compute_convex_hull(
        self, points: np.ndarray, padding_factor: float = 0.15
    ) -> Optional[np.ndarray]:
        """
        Compute expanded convex hull for a set of points with padding for better visual coverage.

        Args:
            points: 2D points array
            padding_factor: Factor to expand the hull outward (0.15 = 15% expansion)

        Returns:
            Expanded hull vertices or None if not enough points
        """
        if len(points) < 3:
            return None

        try:
            hull = ConvexHull(points)
            hull_vertices = points[hull.vertices]

            # Calculate centroid of the cluster
            centroid = np.mean(points, axis=0)

            # Expand each vertex outward from the centroid
            expanded_vertices = []
            for vertex in hull_vertices:
                # Vector from centroid to vertex
                direction = vertex - centroid
                # Expand outward by padding_factor
                expanded_vertex = vertex + direction * padding_factor
                expanded_vertices.append(expanded_vertex)

            return np.array(expanded_vertices)

        except Exception as e:
            print(f"    Warning: Could not compute convex hull: {e}")
            return None

    def _compute_smooth_hull(
        self,
        points: np.ndarray,
        padding_factor: float = 0.15,
        smoothing_factor: float = 0.3,
    ) -> Optional[np.ndarray]:
        """
        Compute a smooth, rounded hull around points using interpolation.

        Args:
            points: 2D points array
            padding_factor: Factor to expand the hull outward
            smoothing_factor: How much to round the corners (0.0 = sharp, 1.0 = very round)

        Returns:
            Smooth hull vertices or None if not enough points
        """
        if len(points) < 3:
            return None

        try:
            # First get the convex hull
            hull = ConvexHull(points)
            hull_vertices = points[hull.vertices]
            centroid = np.mean(points, axis=0)

            # Expand vertices outward
            expanded_vertices = []
            for vertex in hull_vertices:
                direction = vertex - centroid
                expanded_vertex = vertex + direction * padding_factor
                expanded_vertices.append(expanded_vertex)

            expanded_vertices = np.array(expanded_vertices)

            # Create smooth boundary by interpolating between vertices with curves
            smooth_points = []
            n_vertices = len(expanded_vertices)
            n_interpolation = 20  # Points between each pair of vertices

            for i in range(n_vertices):
                current = expanded_vertices[i]
                next_vertex = expanded_vertices[(i + 1) % n_vertices]
                prev_vertex = expanded_vertices[(i - 1) % n_vertices]

                # Add the current vertex
                smooth_points.append(current)

                # Create curved transition to next vertex
                # Use bezier-like curve with control points
                control_distance = (
                    np.linalg.norm(next_vertex - current) * smoothing_factor
                )

                # Control point: move from current vertex towards next, but also slightly outward
                direction_to_next = (next_vertex - current) / np.linalg.norm(
                    next_vertex - current
                )
                direction_outward = (current - centroid) / np.linalg.norm(
                    current - centroid
                )
                control_point = (
                    current
                    + direction_to_next * control_distance
                    + direction_outward * control_distance * 0.3
                )

                # Interpolate curve from current to next vertex
                for j in range(1, n_interpolation):
                    t = j / n_interpolation
                    # Quadratic Bezier curve: (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                    curve_point = (
                        (1 - t) ** 2 * current
                        + 2 * (1 - t) * t * control_point
                        + t**2 * next_vertex
                    )
                    smooth_points.append(curve_point)

            return np.array(smooth_points)

        except Exception as e:
            print(f"    Warning: Could not compute smooth hull: {e}")
            # Fallback to regular convex hull
            return self._compute_convex_hull(points, padding_factor)

    def _create_blob_boundary(
        self, points: np.ndarray, method: str = "smooth", padding_factor: float = 0.15
    ) -> Optional[np.ndarray]:
        """
        Create blob boundary around points.

        Args:
            points: 2D points array
            method: 'convex' for convex hull, 'circle' for circular boundary, 'smooth' for rounded convex hull
            padding_factor: Expansion factor for better coverage

        Returns:
            Boundary points or None
        """
        if len(points) < 2:
            return None

        if method == "convex":
            return self._compute_convex_hull(points, padding_factor=padding_factor)
        elif method == "circle":
            # Create circular boundary around points
            center = np.mean(points, axis=0)
            distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
            radius = np.max(distances) * (1.2 + padding_factor)  # Add padding

            # Generate circle points
            theta = np.linspace(0, 2 * np.pi, 50)
            circle_points = np.column_stack(
                [center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)]
            )
            return circle_points
        elif method == "smooth":
            return self._compute_smooth_hull(points, padding_factor=padding_factor)

        return None

    def _plot_dimensionality_reduction(
        self,
        activations: np.ndarray,
        y_true: np.ndarray,
        cluster_labels: np.ndarray,
        method: str,
        threshold: float,
        title_prefix: str,
        save_path: str,
    ) -> plt.Figure:
        """
        Create a dimensionality reduction plot with cluster blobs.

        Args:
            activations: Input activation data
            y_true: True class labels
            cluster_labels: Cluster assignments at threshold
            method: 'pca', 'tsne', or 'mds'
            threshold: Distance threshold value
            title_prefix: Prefix for plot title
            save_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        print(f"      Creating {method.upper()} plot...")

        # Apply dimensionality reduction
        if method.lower() == "pca":
            reducer = PCA(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(activations)
            method_name = "PCA"
            explained_var = reducer.explained_variance_ratio_
            xlabel = f"PC1 ({explained_var[0]:.2%} variance)"
            ylabel = f"PC2 ({explained_var[1]:.2%} variance)"

        else:
            raise ValueError(f"Unknown method: {method}")

        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)

        # First, draw cluster boundaries (behind points)
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points for hull drawing
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_points = reduced_data[cluster_mask]

            if len(cluster_points) >= 3:  # Need at least 3 points for convex hull
                # Get cluster color (discrete colormap)
                color_idx = cluster_id % len(self.cluster_colors)
                cluster_color = self.cluster_colors[color_idx]

                # Create smooth hull
                hull_points = self._create_blob_boundary(
                    cluster_points, method="smooth"
                )
                if hull_points is not None:
                    # Draw filled hull with better border styling
                    hull_polygon = Polygon(
                        hull_points,
                        alpha=self.alpha_hull,
                        facecolor=cluster_color,
                        edgecolor="black",
                        linewidth=2.5,
                        linestyle="-",
                        zorder=1,
                    )
                    ax.add_patch(hull_polygon)

                    # Add cluster label
                    center = np.mean(cluster_points, axis=0)
                    ax.text(
                        center[0],
                        center[1],
                        f"C{cluster_id}",
                        fontsize=11,
                        fontweight="bold",
                        color="black",
                        ha="center",
                        va="center",
                        zorder=3,
                        bbox=dict(
                            boxstyle="round,pad=0.4",
                            facecolor="white",
                            alpha=0.9,
                            edgecolor="black",
                            linewidth=1,
                        ),
                    )

        # Then, draw points colored by true labels (on top)
        # Use discrete colors for true labels
        unique_true_labels = sorted(set(y_true))
        n_true_labels = len(unique_true_labels)

        # Choose appropriate discrete colormap for true labels
        if n_true_labels <= 10:
            true_label_cmap = plt.cm.tab10
        elif n_true_labels <= 20:
            true_label_cmap = plt.cm.tab20
        else:
            true_label_cmap = plt.cm.tab20b

        # Create color array for true labels
        true_label_colors = []
        for label in y_true:
            if label == -1:
                # Gray for noise points
                true_label_colors.append((0.5, 0.5, 0.5, 1.0))
            else:
                label_idx = unique_true_labels.index(label)
                true_label_colors.append(true_label_cmap(label_idx % true_label_cmap.N))

        scatter = ax.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=true_label_colors,
            s=60,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
            zorder=2,
        )

        # Customize plot
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        # ax.set_title(f"{title_prefix} - {method_name} - Death Threshold {threshold:.4f}",
        #             fontsize=16, fontweight='bold', pad=20)
        ax.set_title(
            f"{method_name} - Death Threshold {threshold:.4f}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        # Add colorbar
        true_labels = np.unique(y_true)
        cbar = plt.colorbar(scatter, ax=ax, label="True Class Labels", shrink=0.8)
        cbar.set_ticks(true_labels)
        cbar.set_ticklabels(
            [self.class_names.get(i, f"Class_{i}") for i in true_labels], fontsize=10
        )
        cbar.set_label("True Class Labels", fontsize=12)

        # Clean aesthetics - remove ticks and grids
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_facecolor("white")

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add legend for cluster hulls
        cluster_info = []
        for cluster_id in unique_clusters:
            if cluster_id != -1:
                count = np.sum(cluster_labels == cluster_id)
                cluster_info.append(f"Cluster {cluster_id}: {count} points")

        if cluster_info:
            legend_text = "\n".join(cluster_info[:10])  # Show first 10 clusters
            ax.text(
                0.02,
                0.98,
                legend_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

        plt.tight_layout()

        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        print(f"        Saved: {save_path}")

        return fig

    def analyze_blob_separation(
        self,
        activations: np.ndarray,
        y_true: np.ndarray,
        cluster_evolution: Dict,
        output_dir: str,
        model_name: str,
        condition_name: str,
        layer_name: str,
        distance_metric: str = "Euclidean",
    ) -> Dict:
        """
        Analyze cluster separation at middle and 4th stage thresholds.

        Args:
            activations: Activation data
            y_true: True class labels
            cluster_evolution: Cluster evolution data from ClusterFlowAnalyzer
            output_dir: Output directory for plots
            model_name: Model name
            condition_name: Condition name (e.g., 'gaussian', 'inference')
            layer_name: Layer name
            distance_metric: Distance metric used ('Euclidean', 'Mahalanobis', etc.)

        Returns:
            Dictionary with analysis results
        """
        print(f"    Analyzing blob separation for {layer_name} - {distance_metric}")

        # Extract thresholds and labels from cluster evolution
        components_ = cluster_evolution["components_"]
        labels_ = cluster_evolution["labels_"]

        if distance_metric not in components_ or distance_metric not in labels_:
            print(
                f"      Warning: {distance_metric} not found in cluster evolution data"
            )
            return {}

        # Get all thresholds (sorted)
        thresholds = sorted([float(t) for t in components_[distance_metric].keys()])

        if len(thresholds) < 4:
            print(f"      Warning: Need at least 4 thresholds, got {len(thresholds)}")
            return {}

        # Select middle (3rd) and 4th stage thresholds
        # In 5-stage evolution: Stage 1=True, Stage 2=thresholds[0], Stage 3=thresholds[1],
        # Stage 4=thresholds[2], Stage 5=thresholds[3]
        middle_threshold = thresholds[1]  # 3rd stage (index 1)
        fourth_threshold = thresholds[2]  # 4th stage (index 2)

        print(
            f"      Selected thresholds: Middle={middle_threshold:.4f}, Fourth={fourth_threshold:.4f}"
        )

        # Get cluster labels for both thresholds
        middle_labels = labels_[distance_metric][str(middle_threshold)]
        fourth_labels = labels_[distance_metric][str(fourth_threshold)]

        # Clean names for file paths
        clean_layer_name = layer_name.replace("/", "_").replace(".", "_")

        # Create title prefix
        # if condition_name.lower() in ['inference', 'clean']:
        #     if model_name.lower() == 'original':
        #         title_prefix = f"{distance_metric} - {layer_name}"
        #     else:
        #         title_prefix = f" {model_name.replace('_', ' ').title()} - {distance_metric} - {layer_name}"
        # else:
        #     if model_name.lower() == 'original':
        #         title_prefix = f"{condition_name.replace('_', ' ').title()} - {distance_metric} - {layer_name}"
        #     else:
        #         title_prefix = f" {model_name.replace('_', ' ').title()} - {condition_name.replace('_', ' ').title()} - {distance_metric} - {layer_name}"
        title_prefix = ""
        results = {}

        # Analyze both thresholds
        for stage_name, threshold, cluster_labels in [
            ("middle", middle_threshold, middle_labels),
            ("fourth", fourth_threshold, fourth_labels),
        ]:
            print(f"      Processing {stage_name} stage (threshold={threshold:.4f})...")

            stage_results = {}

            # Create plots for each dimensionality reduction method
            for method in ["pca"]:
                save_path = os.path.join(
                    output_dir,
                    f"{model_name}_{condition_name}_{clean_layer_name}_{distance_metric}_{threshold:.4f}_{method}.png",
                )

                try:
                    fig = self._plot_dimensionality_reduction(
                        activations,
                        y_true,
                        cluster_labels,
                        method,
                        threshold,
                        title_prefix,
                        save_path,
                    )
                    stage_results[method] = {
                        "figure": fig,
                        "save_path": save_path,
                        "threshold": threshold,
                    }
                    plt.close(fig)  # Free memory

                except Exception as e:
                    print(f"        Error creating {method} plot: {e}")
                    stage_results[method] = None

            # Calculate cluster statistics
            unique_clusters = np.unique(cluster_labels)
            n_clusters = len(unique_clusters)
            cluster_sizes = [np.sum(cluster_labels == c) for c in unique_clusters]

            stage_results["statistics"] = {
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "unique_clusters": unique_clusters.tolist(),
                "threshold": threshold,
            }

            results[stage_name] = stage_results

            print(
                f"        {stage_name.title()} stage: {n_clusters} clusters, sizes: {cluster_sizes}"
            )

        return results

    def plot_pca_with_cluster_hulls(
        self,
        points: np.ndarray,
        true_labels: np.ndarray,
        threshold: float,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create PCA plot with points colored by true labels and convex hulls for clusters at threshold.

        Args:
            points: Input data points (n_samples, n_features)
            true_labels: True class labels for coloring points
            threshold: Distance threshold for clustering
            save_path: Optional path to save the plot
            title: Optional title for the plot

        Returns:
            matplotlib Figure object
        """
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.decomposition import PCA

        # Apply PCA for 2D visualization
        pca = PCA(n_components=2, random_state=42)
        points_2d = pca.fit_transform(points)

        # Get cluster assignments at threshold
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold, linkage="single"
        )
        cluster_labels = clustering.fit_predict(points)

        # Create the visualization
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # Define colors for true classes
        class_colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
        ]

        # Define colors for cluster hulls
        hull_colors = [
            "lightblue",
            "lightcoral",
            "lightgreen",
            "lightyellow",
            "lightpink",
            "lightcyan",
            "lightgray",
            "lightsalmon",
            "lightsteelblue",
            "lightgoldenrodyellow",
        ]

        # Plot convex hulls for each cluster (behind points)
        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_points_2d = points_2d[cluster_mask]

            if len(cluster_points_2d) >= 3:  # Need at least 3 points for ConvexHull
                try:
                    hull = ConvexHull(cluster_points_2d)
                    hull_points = cluster_points_2d[hull.vertices]

                    # Add padding to make blobs thicker
                    center = np.mean(hull_points, axis=0)
                    padded_points = center + (hull_points - center) * 1.3

                    # Create polygon for the blob with distinct color per cluster
                    cluster_color = hull_colors[cluster_id % len(hull_colors)]
                    polygon = Polygon(
                        padded_points,
                        alpha=self.alpha_hull,
                        facecolor=cluster_color,
                        edgecolor="darkblue",
                        linewidth=2,
                        linestyle="-",
                    )
                    ax.add_patch(polygon)
                except Exception:
                    # Fallback for degenerate cases
                    pass

        # Plot data points colored by TRUE classes (on top of blobs) - BIGGER for paper
        for class_id in np.unique(true_labels):
            class_mask = true_labels == class_id
            ax.scatter(
                points_2d[class_mask, 0],
                points_2d[class_mask, 1],
                c=class_colors[class_id % len(class_colors)],
                s=120,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.8,
            )  # Bigger points, no label

        # Set labels and title - BIGGER fonts for paper
        ax.set_xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=14
        )
        ax.set_ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=14
        )

        if title is None:
            title = f"HOLE Blob Visualization: PCA + Cluster Hulls (Threshold: {threshold:.3f})"
        ax.set_title(title, fontsize=16, fontweight="bold")

        # Remove ticks but keep axes
        ax.set_xticks([])
        ax.set_yticks([])

        # NO legend and NO grid for paper quality

        plt.tight_layout()

        # Save if path provided
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"Saved blob visualization: {save_path}")

        return fig


def analyze_activation_blobs(
    activation_file: str,
    output_dir: str,
    model_name: str,
    condition_name: str,
    true_labels: Optional[np.ndarray] = None,
    max_points: int = 100,
    distance_metrics: List[str] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Analyze blob separation for activations at middle and 4th stage thresholds.

    Args:
        activation_file: Path to activation .npy file
        output_dir: Output directory for blob visualizations
        model_name: Model name
        condition_name: Condition name
        true_labels: True class labels
        max_points: Maximum points to use for analysis
        distance_metrics: List of distance metrics to analyze
        class_names: Optional dictionary mapping class indices to names

    Returns:
        Dictionary with blob analysis results
    """
    print(f"Analyzing blob separation for {model_name} - {condition_name}")

    if distance_metrics is None:
        distance_metrics = [
            "Euclidean",
            "Mahalanobis",
            "Cosine",
            "Density_Normalized_Euclidean",
            "Density_Normalized_Mahalanobis",
        ]

    # Load activations
    try:
        all_activations = np.load(activation_file, allow_pickle=True).item()
        if not isinstance(all_activations, dict):
            print(f"Warning: Expected dictionary, got {type(all_activations)}")
            return {}
    except Exception as e:
        print(f"Error loading {activation_file}: {e}")
        return {}

    if true_labels is None:
        print("Warning: No true labels provided, skipping blob analysis")
        return {}

    # Import required modules
    from mst_proc import MSTProcessor
    from vis.flow_visualization import ClusterFlowAnalyzer

    from hole.core.distance_metrics import distance_matrix

    # Initialize blob visualizer
    blob_viz = BlobVisualizer(
        figsize=(14, 10), dpi=300, alpha_hull=0.3, class_names=class_names
    )

    # Create output directory
    blob_output_dir = os.path.join(output_dir, "blob_vis")
    os.makedirs(blob_output_dir, exist_ok=True)

    results = {}

    # Process each layer
    for layer_name, activation_data in all_activations.items():
        print(f"  Processing layer: {layer_name}")

        # Handle activation shapes
        if len(activation_data.shape) == 3:
            pc = activation_data[:, 0, :]  # Use class token
        elif len(activation_data.shape) == 2:
            pc = activation_data
        else:
            continue

        # Subsample if needed
        if pc.shape[0] > max_points:
            indices = np.random.choice(pc.shape[0], max_points, replace=False)
            pc = pc[indices]
            layer_labels = true_labels[indices] if true_labels is not None else None
        else:
            layer_labels = true_labels

        if layer_labels is None:
            continue

        # Initialize MST processor for distance calculations
        mst_obj = MSTProcessor()

        try:
            # Compute distance matrices
            X_pca = mst_obj.pca_utils(pc)

            distance_matrices = {}
            if "Euclidean" in distance_metrics:
                distance_matrices["Euclidean"] = distance_matrix(pc)
            if "Mahalanobis" in distance_metrics:
                distance_matrices["Mahalanobis"] = mst_obj.fast_maha(X_pca)
            if "Cosine" in distance_metrics:
                distance_matrices["Cosine"] = mst_obj.cosine_gen(pc)
            if "Density_Normalized_Euclidean" in distance_metrics:
                distance_matrices[
                    "Density_Normalized_Euclidean"
                ] = mst_obj.density_normalizer(
                    X=X_pca,
                    dists=distance_matrices.get("Euclidean", distance_matrix(pc)),
                    k=5,
                )
            if "Density_Normalized_Mahalanobis" in distance_metrics:
                distance_matrices[
                    "Density_Normalized_Mahalanobis"
                ] = mst_obj.density_normalizer(
                    X=X_pca,
                    dists=distance_matrices.get(
                        "Mahalanobis", mst_obj.fast_maha(X_pca)
                    ),
                    k=5,
                )

            layer_results = {}

            # Process each distance metric
            for dist_name, dist_matrix in distance_matrices.items():
                print(f"    Processing {dist_name} distance metric...")

                try:
                    # Compute cluster evolution
                    analyzer = ClusterFlowAnalyzer(dist_matrix, max_thresholds=4)
                    cluster_evolution = analyzer.compute_cluster_evolution(layer_labels)

                    # Analyze blob separation
                    blob_results = blob_viz.analyze_blob_separation(
                        pc,
                        layer_labels,
                        cluster_evolution,
                        blob_output_dir,
                        model_name,
                        condition_name,
                        layer_name,
                        dist_name,
                    )

                    layer_results[dist_name] = blob_results

                except Exception as e:
                    print(f"      Error processing {dist_name}: {e}")
                    continue

            results[layer_name] = layer_results

        except Exception as e:
            print(f"    Error processing layer {layer_name}: {e}")
            continue

    return results


def run_blob_analysis_on_results(
    results_dir: str = "results_compression",
    max_points: int = 100,
    distance_metrics: List[str] = None,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """
    Run blob analysis on all activation files in results directory.

    Args:
        results_dir: Directory containing model results
        max_points: Maximum points per analysis
        distance_metrics: List of distance metrics to analyze
        class_names: Optional dictionary mapping class indices to names
    """
    print(f"Running blob analysis on {results_dir}...")

    if distance_metrics is None:
        distance_metrics = [
            "Euclidean",
            "Mahalanobis",
            "Cosine",
            "Density_Normalized_Euclidean",
            "Density_Normalized_Mahalanobis",
        ]

    # Load true labels
    true_labels = None
    if results_dir == "results":
        labels_file = f"{results_dir}/original/true_labels.npy"
    else:
        # For compression analysis, try multiple locations
        possible_paths = [
            f"{results_dir}/test_labels.npy",  # New location for compression analysis
            f"{results_dir}/../results/original/true_labels.npy",
            "results/original/true_labels.npy",
            f"{results_dir}/original/true_labels.npy",
        ]

        labels_file = None
        for path in possible_paths:
            if os.path.exists(path):
                labels_file = path
                break

    if labels_file and os.path.exists(labels_file):
        true_labels = np.load(labels_file)
        print(f"Loaded true labels from {labels_file}")
    else:
        print(f"Warning: True labels not found in any expected location")
        print(
            f"Searched: {possible_paths if 'possible_paths' 
                in locals() else [labels_file]}"
        )
        return

    # Process each model directory
    for model_name in os.listdir(results_dir):
        model_path = f"{results_dir}/{model_name}"
        if os.path.isdir(model_path) and model_name not in [
            "visualizations",
            "model_stats",
            "tda_analysis",
            "persistence_dendrograms",
            "flow_visualization",
            "blob_vis",
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

                    analyze_activation_blobs(
                        activation_path,
                        results_dir,
                        model_name,
                        condition_name,
                        true_labels,
                        max_points,
                        distance_metrics,
                        class_names,
                    )


if __name__ == "__main__":
    print(
        "Blob Visualization: Cluster separation analysis at persistent homology thresholds"
    )
    print("Usage:")
    print("  from vis.blob_vis import BlobVisualizer, analyze_activation_blobs")
    print("  run_blob_analysis_on_results('results_compression')")
