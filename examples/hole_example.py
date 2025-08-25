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

    print(f"\n=== PART 5: SCATTER HULL VISUALIZATION ({metric_name}) ===")

    # 5. Scatter Hull (Blob) Visualization
    blob_viz = hole_viz.get_blob_visualizer()
    blob_viz.compute_cluster_evolution(true_labels)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle(f"Scatter Hull Visualization - {metric_name}", fontsize=14)

    blob_viz.plot_blob_separation(
        ax=ax,
        title=f"Cluster Separation Analysis - {metric_name}",
        show_legend=True,
    )
    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/scatter_hull_{metric_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

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
