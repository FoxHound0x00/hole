"""
Three Blobs HOLE Visualization Example

This script demonstrates ALL visualization capabilities of the HOLE library
on three blob data:
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
from hole.visualization.cluster_flow import (
    ClusterFlowAnalyzer,
    ComponentEvolutionVisualizer,
)
from hole.visualization.scatter_hull import BlobVisualizer

np.random.seed(42)

# Create output directory for plots
OUTPUT_DIR = "examples/three_blobs_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

X, y = make_blobs(n_samples=500, centers=3, n_features=20, random_state=42)

# Create distance matrices for all available metrics
distance_matrices = {
    "euclidean": distance_metrics.euclidean_distance(X),
    "cosine": distance_metrics.cosine_distance(X),
    "mahalanobis": distance_metrics.mahalanobis_distance(X),
    "dn_euclidean": distance_metrics.density_normalized_euclidean(X),
    "dn_cosine": distance_metrics.density_normalized_cosine(X),
    "dn_mahalanobis": distance_metrics.density_normalized_mahalanobis(X),
}

# Try to add geodesic distance if possible
try:
    geodesic_dist = distance_metrics.geodesic_distance(
        X, n_neighbors=min(10, X.shape[0] - 1)
    )
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
for name, dist_matrix in distance_matrices.items():
    print(f"\n=== PROCESSING {name.upper()} DISTANCE ===")

    # Create HOLEVisualizer for persistence analysis (using distance matrix)
    viz = HOLEVisualizer(
        distance_matrix_input=dist_matrix,
        distance_metric=name,
        max_dimension=1,
        max_edge_length=np.inf,
    )

    # Map custom metrics to sklearn-compatible metrics for PCA visualizer
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
    viz_pca = HOLEVisualizer(
        point_cloud=X,
        distance_metric=sklearn_metric_map.get(name, "euclidean"),
        max_dimension=1,
        max_edge_length=np.inf,
    )

    print(f"\n=== PART 1: PERSISTENCE VISUALIZATIONS ({name}) ===")

    # 1. Persistence Diagrams and Barcodes
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Persistence Analysis - {name}", fontsize=14)

    # Plot persistence diagram
    viz.plot_persistence_diagram(ax=axes[0], title="Persistence Diagram")

    # Plot persistence barcode
    viz.plot_persistence_barcode(ax=axes[1], title="Persistence Barcode")

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/persistence_visualizations_{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\n=== PART 2: DIMENSIONALITY REDUCTION ({name}) ===")

    # 2. All Dimensionality Reduction Methods (PCA, t-SNE, MDS)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Dimensionality Reduction - {name}", fontsize=14)

    viz_pca.plot_dimensionality_reduction(
        method="pca", ax=axes[0], true_labels=y, title="PCA"
    )
    viz_pca.plot_dimensionality_reduction(
        method="tsne", ax=axes[1], true_labels=y, title="t-SNE"
    )
    viz_pca.plot_dimensionality_reduction(
        method="mds", ax=axes[2], true_labels=y, title="MDS"
    )

    plt.tight_layout()
    plt.savefig(
        f"{OUTPUT_DIR}/dimensionality_reduction_{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\n=== PART 3: HEATMAP AND DENDROGRAM ({name}) ===")

    # 3. Distance Matrix Heatmap with Dendrogram
    from hole.visualization import plot_heatmap_with_dendrogram

    plot_heatmap_with_dendrogram(
        dist_matrix,
        true_labels=y,
        title=f"Distance Matrix Heatmap - {name}",
        figsize=(16, 8),
    )
    plt.savefig(
        f"{OUTPUT_DIR}/heatmap_dendrograms_{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\n=== PART 4: CLUSTER FLOW ANALYSIS ({name}) ===")

    # 4. Cluster Flow Analysis (Sankey and Stacked Bar Charts)
    flow_analyzer = ClusterFlowAnalyzer(distance_matrix=dist_matrix)
    components_ = flow_analyzer.analyze_cluster_flow()

    if components_:
        # Create component evolution visualizer
        comp_viz = ComponentEvolutionVisualizer(components_)

        # Plot Sankey Diagram
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        fig.suptitle(f"Cluster Flow Evolution - {name}", fontsize=14)

        first_key = list(components_.keys())[0]
        comp_viz.plot_sankey(
            first_key,
            original_labels=y,
            ax=ax,
            title="Cluster Evolution Flow",
            gray_second_layer=True,
        )
        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/sankey_diagram_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Plot Stacked Bar Chart
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.suptitle(f"Cluster Evolution Stages - {name}", fontsize=14)

        comp_viz.plot_stacked_bars(
            first_key,
            original_labels=y,
            ax=ax,
            title="Cluster Evolution Stages",
            gray_second_layer=True,
        )
        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/stacked_bar_chart_{name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    print(f"\n=== PART 5: SCATTER HULL VISUALIZATION ({name}) ===")

    # 5. Scatter Hull Visualization
    blob_viz = BlobVisualizer(distance_matrix=dist_matrix)
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle(f"Scatter Hull Visualization - {name}", fontsize=14)

    blob_viz.visualize_blobs(
        true_labels=y,
        ax=ax,
        title=f"Blob Separation - {name}",
        show_legend=True,
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scatter_hull_{name}.png", dpi=300, bbox_inches="tight")
    plt.show()

print("\n" + "=" * 80)
print("COMPREHENSIVE COMPARISON ACROSS ALL METRICS")
print("=" * 80)

# Create a comprehensive comparison plot
n_metrics = len(distance_matrices)
n_cols = min(4, n_metrics)
n_rows = (n_metrics + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
fig.suptitle("PCA Comparison Across All Distance Metrics", fontsize=16)

if n_rows == 1:
    axes = [axes] if n_metrics == 1 else axes
else:
    axes_flat = axes.flatten()

for i, (metric_name, _) in enumerate(distance_matrices.items()):
    if n_rows == 1:
        ax = axes[i] if n_metrics > 1 else axes
    else:
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
            point_cloud=X,
            distance_metric=sklearn_metric_map.get(metric_name, "euclidean"),
        )

        ax = axes_flat[i]
        viz.plot_dimensionality_reduction(
            method="pca",
            ax=ax,
            true_labels=y,
            title=f"PCA - {metric_name}",
        )

# Hide empty subplots
if n_rows > 1:
    for i in range(len(distance_matrices), len(axes_flat)):
        axes_flat[i].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metric_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print(f"Total distance metrics processed: {len(distance_matrices)}")
