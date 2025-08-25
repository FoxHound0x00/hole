import os

import matplotlib.pyplot as plt
import numpy as np

import hole

np.random.seed(42)

# Create output directory for plots
OUTPUT_DIR = "examples/three_blobs_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=3, n_features=20, random_state=42)

from hole.core import distance_metrics
from hole.visualization.cluster_flow import (
    ClusterFlowAnalyzer,
    ComponentEvolutionVisualizer,
)
from hole.visualizer import HOLEVisualizer

distance_matrices = {
    "euclidean": distance_metrics.euclidean_distance(X),
    "cosine": distance_metrics.cosine_distance(X),
    "mahalanobis": distance_metrics.mahalanobis_distance(X=X),
    "dn_euclidean": distance_metrics.density_normalized_distance(
        X=X, dists=distance_metrics.euclidean_distance(X)
    ),
    "dn_cosine": distance_metrics.density_normalized_distance(
        X=X, dists=distance_metrics.cosine_distance(X)
    ),
    "dn_mahalanobis": distance_metrics.density_normalized_distance(
        X=X, dists=distance_metrics.mahalanobis_distance(X=X)
    ),
}

# Handle geodesic distances with infinite value check
try:
    geodesic_dist = distance_metrics.geodesic_distances(X)
    # Replace infinite values with a large finite value
    geodesic_dist[np.isinf(geodesic_dist)] = (
        np.nanmax(geodesic_dist[np.isfinite(geodesic_dist)]) * 10
    )
    distance_matrices["geodesic"] = geodesic_dist
except:
    print("Skipping geodesic distance due to computation issues")
    pass

for name, dist_matrix in distance_matrices.items():
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
        "dn_euclidean": "euclidean",  # Use base euclidean for PCA
        "dn_cosine": "cosine",  # Use base cosine for PCA
        "dn_mahalanobis": "mahalanobis",  # Use base mahalanobis for PCA
        "geodesic": "euclidean",  # Use euclidean as fallback for geodesic
    }

    # Create separate HOLEVisualizer for PCA (using original point cloud)
    viz_pca = HOLEVisualizer(
        point_cloud=X,
        distance_metric=sklearn_metric_map.get(name, "euclidean"),
        max_dimension=1,
        max_edge_length=np.inf,
    )

    ## plot persistence diagram, barcodes, and dimensionality reduction
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{name} Distance Matrix", fontsize=16)

    ax1 = viz.plot_persistence_diagram(ax=axes[0, 0], title="Persistence Diagram")
    ax2 = viz.plot_persistence_barcode(ax=axes[0, 1], title="Persistence Barcode")
    ax3 = viz_pca.plot_dimensionality_reduction(
        method="pca", ax=axes[1, 0], title="PCA (Original Features)"
    )
    ax4 = viz.plot_dimensionality_reduction(
        method="mds", ax=axes[1, 1], title="MDS (Distance Matrix)"
    )

    plt.tight_layout()
    os.makedirs("plots/", exist_ok=True)
    plt.savefig(
        f"plots/persistence_visualizations_{name}.png", dpi=500, bbox_inches="tight"
    )
    plt.show()
    plt.close()

    ## plot cluster flow evolution - Sankey diagram and stacked charts
    print(f"Computing cluster flow evolution for {name}...")
    analyzer = ClusterFlowAnalyzer(dist_matrix, max_thresholds=6)
    cluster_evolution = analyzer.compute_cluster_evolution(y)

    # Extract components and labels for visualization
    components_ = cluster_evolution.get("components_", {})
    labels_ = cluster_evolution.get("labels_", {})

    if components_ and labels_:
        comp_viz = ComponentEvolutionVisualizer(components_, labels_)

        # Get the first distance metric key
        first_key = list(components_.keys())[0]

        # Plot Sankey diagram
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        fig.suptitle(
            f"{name} Distance Matrix - Cluster Flow Evolution (Sankey)", fontsize=16
        )

        comp_viz.plot_sankey(
            first_key,
            original_labels=y,
            ax=ax,
            title="Cluster Flow Evolution",
            gray_second_layer=True,
        )

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/sankey_diagram_{name}.png", dpi=500, bbox_inches="tight"
        )
        plt.show()
        plt.close()

        # Plot Stacked Bar Chart
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.suptitle(
            f"{name} Distance Matrix - Cluster Flow Evolution (Stacked)", fontsize=16
        )

        comp_viz.plot_stacked_bars(
            first_key,
            original_labels=y,
            ax=ax,
            title="Cluster Evolution",
            gray_second_layer=True,
        )

        plt.tight_layout()
        plt.savefig(
            f"{OUTPUT_DIR}/stacked_bar_chart_{name}.png", dpi=500, bbox_inches="tight"
        )
        plt.show()
        plt.close()
    else:
        print(f"No cluster evolution data available for {name}")

    ## plot heatmap dendrograms
    print(f"Creating heatmap dendrograms for {name}...")
    heatmap_dendro_viz = viz.get_heatmap_dendrogram_visualizer(
        distance_matrix=dist_matrix
    )
    heatmap_dendro_viz.compute_persistence()

    # Create heatmap with dendrogram
    heatmap_dendro_viz.plot_dendrogram_with_heatmap(
        labels=[f"P{i}" for i in range(len(X))],
        title=f"{name} Distance Matrix with Dendrogram",
        figsize=(16, 8),
    )
    plt.savefig(
        f"{OUTPUT_DIR}/heatmap_dendrograms_{name}.png", dpi=500, bbox_inches="tight"
    )
    plt.show()
    plt.close()

    ## plot scatter hull visualization
    print(f"Creating scatter hull visualization for {name}...")
    scatter_hull_viz = viz.get_scatter_hull_visualizer(figsize=(12, 8), alpha_hull=0.3)

    # Perform blob separation analysis
    results = scatter_hull_viz.analyze_blob_separation(
        activations=X,
        y_true=y,
        cluster_evolution=cluster_evolution,
        output_dir="plots/",
        model_name="three_blobs",
        condition_name=name,
        layer_name="input",
        distance_metric=name,
    )

    print(f"Scatter hull analysis completed for {name}")

print("All visualizations completed!")
