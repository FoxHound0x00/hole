"""
Comprehensive HOLE Visualization Library Example

This script demonstrates ALL visualization capabilities of the HOLE library:
- Persistence diagrams and barcodes for ALL distance metrics
- PCA, MDS, t-SNE dimensionality reduction
- Heatmaps and dendrograms
- Sankey diagrams for cluster flow
- Stacked bar charts for cluster evolution
- Scatter hull visualizations (blob separation)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from hole import HOLEVisualizer
from hole.core import distance_metrics
from hole.visualization import ClusterFlowAnalyzer
from hole.visualization.scatter_hull import BlobVisualizer

# Create output directory for plots
OUTPUT_DIR = "examples/hole_example_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate synthetic point cloud data
print("Generating synthetic data...")
n_samples = 80
n_features = 5
n_centers = 4

points, true_labels = make_blobs(
    n_samples=n_samples,
    centers=n_centers,
    n_features=n_features,
    cluster_std=2.0,
    random_state=42,
)

print(
    f"Generated {n_samples} points with {n_features} features "
    f"in {n_centers} clusters"
)

# Create distance matrices for all metrics
print("Computing distance matrices...")
distance_matrices = {
    "euclidean": distance_metrics.euclidean_distance(points),
    "cosine": distance_metrics.cosine_distance(points),
    "mahalanobis": distance_metrics.mahalanobis_distance(X=points),
    "dn_euclidean": distance_metrics.density_normalized_distance(
        X=points, dists=distance_metrics.euclidean_distance(points)
    ),
    "dn_cosine": distance_metrics.density_normalized_distance(
        X=points, dists=distance_metrics.cosine_distance(points)
    ),
    "dn_mahalanobis": distance_metrics.density_normalized_distance(
        X=points, dists=distance_metrics.mahalanobis_distance(X=points)
    ),
}

# Handle geodesic distances with infinite value check
try:
    geodesic_dist = distance_metrics.geodesic_distances(points)
    # Replace infinite values with a large finite value
    geodesic_dist[np.isinf(geodesic_dist)] = (
        np.nanmax(geodesic_dist[np.isfinite(geodesic_dist)]) * 10
    )
    distance_matrices["geodesic"] = geodesic_dist
except Exception:
    print("Skipping geodesic distance due to computation issues")
    pass

print(f"Available distance metrics: {list(distance_matrices.keys())}")

# Process each distance metric
for metric_name, dist_matrix in distance_matrices.items():
    print(f"\n=== PROCESSING {metric_name.upper()} DISTANCE ===")

    # Create HOLEVisualizer for this distance matrix
    hole_viz = HOLEVisualizer(
        distance_matrix_input=dist_matrix, distance_metric=metric_name
    )

    # Create sklearn-compatible metric for PCA visualizer
    sklearn_metric_map = {
        "euclidean": "euclidean",
        "cosine": "cosine",
        "mahalanobis": "mahalanobis",
        "dn_euclidean": "euclidean",
        "dn_cosine": "cosine",
        "dn_mahalanobis": "mahalanobis",
        "geodesic": "euclidean",
    }

    # Create separate HOLEVisualizer for PCA (using original point cloud)
    hole_viz_pca = HOLEVisualizer(
        point_cloud=points,
        distance_metric=sklearn_metric_map.get(metric_name, "euclidean"),
    )

    print(f"\n=== PART 1: PERSISTENCE VISUALIZATIONS ({metric_name}) ===")

    # 1. Persistence Diagrams and Barcodes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Persistence Analysis - {metric_name}", fontsize=14)

    # Plot persistence diagram
    hole_viz.plot_persistence_diagram(ax=axes[0], title="Persistence Diagram")

    # Plot persistence barcode
    hole_viz.plot_persistence_barcode(ax=axes[1], title="Persistence Barcode")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/persistence_visualizations_{metric_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\n=== PART 2: DIMENSIONALITY REDUCTION ({metric_name}) ===")

    # 2. Dimensionality Reduction Visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Dimensionality Reduction - {metric_name}", fontsize=14)

    # PCA
    hole_viz_pca.plot_dimensionality_reduction(
        method="pca", ax=axes[0], true_labels=true_labels, title="PCA"
    )

    # MDS
    hole_viz_pca.plot_dimensionality_reduction(
        method="mds", ax=axes[1], true_labels=true_labels, title="MDS"
    )

    # t-SNE
    hole_viz_pca.plot_dimensionality_reduction(
        method="tsne", ax=axes[2], true_labels=true_labels, title="t-SNE"
    )

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/dimensionality_reduction_{metric_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\n=== PART 3: HEATMAPS AND DENDROGRAMS ({metric_name}) ===")

    # 3. Distance Matrix Heatmaps and Dendrograms
    heatmap_dendro_viz = hole_viz.get_heatmap_dendrogram_visualizer(
        distance_matrix=hole_viz.distance_matrix
    )
    heatmap_dendro_viz.compute_persistence()

    # Create heatmap with dendrogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(
        f"Distance Matrix Heatmap with Dendrogram - {metric_name}", fontsize=14
    )

    # Plot heatmap with dendrogram (creates its own figure)
    heatmap_dendro_viz.plot_dendrogram_with_heatmap(
        labels=[f"P{i}" for i in range(n_samples)],
        title=f"{metric_name} Distance Matrix with Dendrogram",
        figsize=(16, 8),
    )
    plt.savefig(
        f"{OUTPUT_DIR}/heatmap_dendrogram_{metric_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\n=== PART 4: CLUSTER FLOW ANALYSIS ({metric_name}) ===")

    # 4. Cluster Flow Analysis (Sankey diagrams and stacked charts)
    analyzer = ClusterFlowAnalyzer(hole_viz.distance_matrix, max_thresholds=6)
    cluster_evolution = analyzer.compute_cluster_evolution(true_labels)

    # Create ComponentEvolutionVisualizer from the cluster evolution data
    from hole.visualization.cluster_flow import ComponentEvolutionVisualizer

    components_ = cluster_evolution.get("components_", {})
    labels_ = cluster_evolution.get("labels_", {})

    if components_ and labels_:
        comp_viz = ComponentEvolutionVisualizer(components_, labels_)

        # Plot Sankey diagram
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        fig.suptitle(f"Cluster Evolution Sankey Diagram - {metric_name}", fontsize=14)

        # Get the first distance metric key
        first_key = list(components_.keys())[0]
        comp_viz.plot_sankey(
            first_key,
            original_labels=true_labels,
            ax=ax,
            title="Cluster Evolution Flow",
            gray_second_layer=True,
        )
        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/sankey_diagram_{metric_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Plot Stacked Bar Chart
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.suptitle(
            f"Cluster Evolution Stacked Bar Chart - {metric_name}", fontsize=14
        )

        comp_viz.plot_stacked_bars(
            first_key,
            original_labels=true_labels,
            ax=ax,
            title="Cluster Evolution Stages",
            gray_second_layer=True,
        )
        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/stacked_bar_chart_{metric_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
    else:
        print(
            f"No cluster evolution data available for {metric_name} flow visualizations"
        )

    print(f"\n=== PART 5: PCA VS BLOB COMPARISON ({metric_name}) ===")

    # 5. PCA vs Blob Comparison: True Labels vs Threshold-based Cluster Hulls
    if components_ and labels_:
        from hole.visualization.scatter_hull import generate_consistent_colors, BlobVisualizer
        
        # Get cluster labels from the best threshold
        first_key = list(components_.keys())[0]
        
        # Find the threshold that gives us reasonable clusters (between 2-10 clusters)
        best_threshold = None
        best_cluster_labels = None
        for threshold_key, n_clusters_value in components_[first_key].items():
            if 2 <= n_clusters_value <= 10:  # Reasonable number of clusters
                best_threshold = threshold_key
                if threshold_key in labels_[first_key]:
                    best_cluster_labels = labels_[first_key][threshold_key]
                break
        
        if best_threshold and best_cluster_labels is not None:
            # Generate consistent colors for true labels and cluster hulls
            n_true_labels = len(np.unique(true_labels))
            n_clusters = len(np.unique(best_cluster_labels))
            
            true_label_colors = generate_consistent_colors(n_true_labels, include_noise=True)
            cluster_colors = generate_consistent_colors(n_clusters, include_noise=True)
            
            # Create color maps
            true_label_color_map = {label: true_label_colors[i] for i, label in enumerate(np.unique(true_labels))}
            cluster_color_map = {label: cluster_colors[i] for i, label in enumerate(np.unique(best_cluster_labels))}
            
            # Create side-by-side PCA plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(f"PCA vs Blob Comparison - {metric_name}", fontsize=16)
            
            # LEFT PLOT: PCA with True Labels
            hole_viz.plot_dimensionality_reduction(
                method="pca", ax=ax1, true_labels=true_labels, 
                title="PCA - True Labels"
            )
            ax1.set_xlabel("PC1")
            ax1.set_ylabel("PC2")
            ax1.grid(True, alpha=0.3)
            
            # RIGHT PLOT: PCA with Cluster Hulls at Threshold
            # First plot the points colored by true labels
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            points_2d = pca.fit_transform(points)
            
            # Plot points colored by true labels
            for label in np.unique(true_labels):
                mask = true_labels == label
                color = true_label_color_map[label]
                ax2.scatter(points_2d[mask, 0], points_2d[mask, 1], 
                           c=[color], s=50, alpha=0.7, label=f"True {label}")
            
            # Draw convex hulls around threshold-based clusters
            blob_viz = BlobVisualizer(figsize=(12, 10), alpha_hull=0.3)
            for cluster_id in np.unique(best_cluster_labels):
                if cluster_id == -1:  # Skip noise
                    continue
                cluster_mask = best_cluster_labels == cluster_id
                if np.sum(cluster_mask) >= 3:  # Need at least 3 points for hull
                    cluster_points = points_2d[cluster_mask]
                    hull_color = cluster_color_map[cluster_id]
                    
                    # Create convex hull boundary
                    hull_boundary = blob_viz._create_blob_boundary(cluster_points, method="convex", padding_factor=0.1)
                    
                    if hull_boundary is not None:
                        # Plot the hull as a filled polygon
                        ax2.fill(hull_boundary[:, 0], hull_boundary[:, 1], 
                                color=hull_color, alpha=0.3, edgecolor=hull_color, linewidth=2)
            
            ax2.set_title(f"PCA - Cluster Hulls (Threshold {best_threshold})")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(
                f"{OUTPUT_DIR}/pca_vs_blob_comparison_{metric_name}.png", 
                dpi=300, bbox_inches="tight"
            )
            plt.show()
            
            print(f"      PCA vs Blob comparison completed for {metric_name} (threshold: {best_threshold})")
        else:
            print(f"No suitable clustering threshold found for {metric_name} blob visualization")
    else:
        print(f"No cluster evolution data available for {metric_name} blob visualization")

print("\n=== COMPARISON VISUALIZATION ===")

# Create a comparison of metrics
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle("Metric Comparison - PCA Visualization", fontsize=16)

axes_flat = axes.flatten()
for i, (metric_name, dist_matrix) in enumerate(distance_matrices.items()):
    if i >= len(axes_flat):
        break

    # Create sklearn-compatible visualizer
    sklearn_metric_map = {
        "euclidean": "euclidean",
        "cosine": "cosine",
        "mahalanobis": "mahalanobis",
        "dn_euclidean": "euclidean",
        "dn_cosine": "cosine",
        "dn_mahalanobis": "mahalanobis",
        "geodesic": "euclidean",
    }

    viz = HOLEVisualizer(
        point_cloud=points,
        distance_metric=sklearn_metric_map.get(metric_name, "euclidean"),
    )

    viz.plot_dimensionality_reduction(
        method="pca",
        ax=axes_flat[i],
        true_labels=true_labels,
        title=f"PCA - {metric_name}",
    )

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metric_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== ALL VISUALIZATIONS COMPLETED ===")
print(f"All plots saved to: {OUTPUT_DIR}/")
print(f"Total distance metrics processed: {len(distance_matrices)}")
