"""
Simple HOLE Library Usage Example

This demonstrates the core visualizations that make 
HOLE powerful for topological data analysis.
Shows heatmap dendrograms, blob visualizations, and cluster flow analysis.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

import hole
from hole.core import distance_metrics
from hole.visualization.cluster_flow import ClusterFlowAnalyzer, FlowVisualizer
from hole.visualization.persistence_vis import plot_dimensionality_reduction


def _paper_ax(ax):
    """Post-process axes for paper-quality output."""
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    ax.tick_params(labelsize=14)
    leg = ax.get_legend()
    if leg:
        leg.set_title(leg.get_title().get_text(), prop={"size": 14})
        for t in leg.get_texts():
            t.set_fontsize(14)


def main():
    """Simple example showcasing HOLE's key visualizations."""

    # Create output directory structure
    output_dir = "hole_example_outputs"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/mds", exist_ok=True)
    os.makedirs(f"{output_dir}/heatmaps", exist_ok=True)
    os.makedirs(f"{output_dir}/core", exist_ok=True)

    print("=== HOLE Simple Example - Core Visualizations ===\n")

    # Generate meaningful sample data with WELL-SEPARATED clusters
    print("Generating 4-cluster dataset...")
    # Create centers that are FAR apart
    centers = np.array([[-8, -8, -8], [8, -8, 8], [-8, 8, 8], [8, 8, -8]])
    points, labels = make_blobs(
        n_samples=500,
        centers=centers,
        n_features=3,
        cluster_std=0.8,
        random_state=42,
    )
    CLASS_NAMES = {i: f"Class_{i}" for i in range(4)}
    print(
        f"Created {len(points)} points with {points.shape[1]} features in 4 well-separated clusters\n"
    )

    # Create HOLE visualizer
    print("Creating HOLE visualizer...")
    visualizer = hole.HOLEVisualizer(point_cloud=points, distance_metric="euclidean")
    print(f"Computed persistence with {len(visualizer.persistence)} features\n")

    # 1. HEATMAP DENDROGRAM - Core topological visualization
    print("1. Creating heatmap dendrogram (core HOLE visualization)...")
    heatmap_viz = visualizer.get_persistence_dendrogram_visualizer(
        distance_matrix=visualizer.distance_matrix
    )
    heatmap_viz.compute_persistence()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    heatmap_viz.plot_dendrogram_with_heatmap(figsize=(16, 8), cmap="gray")
    for a in plt.gcf().get_axes():
        _paper_ax(a)
    plt.savefig(
        f"{output_dir}/core/heatmap_dendrogram.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. BLOB VISUALIZATION
    print("2. Creating blob visualization...")

    # Get cluster evolution to find the middle stage threshold
    analyzer = ClusterFlowAnalyzer(visualizer.distance_matrix, max_thresholds=4)
    cluster_evolution = analyzer.compute_cluster_evolution(labels)

    # Get the middle stage threshold (similar to true labels)
    euclidean_labels = cluster_evolution["labels_"]["Euclidean"]
    thresholds = sorted([float(t) for t in euclidean_labels.keys()])
    middle_threshold = thresholds[
        1
    ]  # Second threshold is the "similar to true labels" one

    print(f"   Using middle stage threshold: {middle_threshold:.3f}")

    # Use the library method with new contour and outlier functionality!
    blob_viz = visualizer.get_blob_visualizer(
        figsize=(12, 9),
        outlier_percentage=0.10,
        show_contours=False,
    )
    fig = blob_viz.plot_pca_with_cluster_hulls(
        points,
        labels,
        middle_threshold,
        save_path=None,
        title="",
        class_names=CLASS_NAMES,
    )
    for a in fig.get_axes():
        _paper_ax(a)
    fig.savefig(f"{output_dir}/core/blob_visualization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. CLUSTER FLOW ANALYSIS - Evolution tracking
    print("3. Creating cluster flow analysis (evolution tracking)...")

    # Use the cluster evolution we already computed
    flow_viz = FlowVisualizer(
        figsize=(20, 11), class_names=CLASS_NAMES
    )

    # Create Sankey flow diagram
    sankey_fig = flow_viz.plot_sankey_flow(
        cluster_evolution,
        save_path=None,
        title="",
        show_true_labels_text=False,
        show_filtration_text=False,
    )
    for a in sankey_fig.get_axes():
        _paper_ax(a)
    sankey_fig.savefig(f"{output_dir}/core/sankey_flow.png", dpi=300, bbox_inches="tight")
    plt.close(sankey_fig)

    # Create stacked bar evolution
    bars_fig = flow_viz.plot_stacked_bar_evolution(
        cluster_evolution,
        save_path=None,
        title="",
        show_true_labels_text=False,
        show_filtration_text=False,
    )
    for a in bars_fig.get_axes():
        _paper_ax(a)
    bars_fig.savefig(f"{output_dir}/core/stacked_bars.png", dpi=300, bbox_inches="tight")
    plt.close(bars_fig)

    # 4. PERSISTENCE VISUALIZATIONS - Traditional TDA
    print("4. Creating persistence visualizations (3 separate plots)...")

    # 4a. Persistence diagram
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    visualizer.plot_persistence_diagram(ax=ax1, title="", pts=20)
    _paper_ax(ax1)
    plt.tight_layout()
    fig1.savefig(
        f"{output_dir}/core/persistence_diagram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig1)

    # 4b. Persistence barcode
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    visualizer.plot_persistence_barcode(ax=ax2, title="", pts=20)
    _paper_ax(ax2)
    plt.tight_layout()
    fig2.savefig(
        f"{output_dir}/core/persistence_barcode.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig2)

    # 4c. PCA with true labels
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    visualizer.plot_dimensionality_reduction(
        method="pca", ax=ax3, true_labels=labels, title=""
    )
    ax3.grid(False)
    for collection in ax3.collections:
        if hasattr(collection, "set_sizes"):
            collection.set_sizes([120])
    _paper_ax(ax3)
    plt.tight_layout()
    fig3.savefig(f"{output_dir}/core/pca_analysis.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # 5. DISTANCE METRIC ANALYSIS - MDS and Heatmaps for all metrics
    print("5. Creating visualizations for all distance metrics...")

    # Create essential distance matrices using library functions
    print("   Computing essential distance matrices...")
    distance_matrices = {
        "euclidean": distance_metrics.euclidean_distance(points),
        "dn_euclidean": distance_metrics.density_normalized_distance(
            points, distance_metrics.euclidean_distance(points)
        ),
        "cosine": distance_metrics.cosine_distance(points),
        "dn_cosine": distance_metrics.density_normalized_distance(
            points, distance_metrics.cosine_distance(points)
        ),
        "mahalanobis": distance_metrics.mahalanobis_distance(points),
        "dn_mahalanobis": distance_metrics.density_normalized_distance(
            points, distance_metrics.mahalanobis_distance(points)
        ),
    }

    # Add geodesic distance with error handling
    try:
        geodesic_dist = distance_metrics.geodesic_distances(points)
        # Replace infinite values with large finite value
        geodesic_dist[np.isinf(geodesic_dist)] = (
            np.nanmax(geodesic_dist[np.isfinite(geodesic_dist)]) * 10
        )
        distance_matrices["geodesic"] = geodesic_dist
    except Exception as e:
        print(f"   Skipping geodesic distance: {e}")

    print(f"   Available metrics: {list(distance_matrices.keys())}")

    # Create MDS and heatmap for each metric
    for metric_name, dist_matrix in distance_matrices.items():
        print(f"   Creating visualizations for {metric_name}...")

        # 1. MDS Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plot_dimensionality_reduction(
            dist_matrix,
            method="mds",
            ax=ax,
            labels=labels,
            point_size=120,
            figsize=(10, 8),
            show_legend=False,
            class_names=CLASS_NAMES,
        )
        ax.set_title("")
        ax.grid(False)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.8)
        _paper_ax(ax)
        plt.tight_layout()
        fig.savefig(
            f"{output_dir}/mds/mds_{metric_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        # 2. RCM-reordered Distance Matrix Heatmap
        heatmap_viz = hole.PersistenceDendrogram(distance_matrix=dist_matrix)
        fig, ax = heatmap_viz.plot_rcm_heatmap(
            title="", figsize=(10, 8), cmap="viridis"
        )
        _paper_ax(ax)
        fig.savefig(
            f"{output_dir}/heatmaps/heatmap_{metric_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    print("\n=== HOLE Simple Example Complete ===")
    print(f"All visualizations saved to: {output_dir}/")
    print(" Organized in subfolders:")
    print(f"  📁 {output_dir}/core/ - Core HOLE visualizations")
    print(f"  📁 {output_dir}/mds/ - MDS plots for all distance metrics")
    print(f"  📁 {output_dir}/heatmaps/ - Distance matrix heatmaps")
    print("")
    print("Key HOLE capabilities demonstrated:")
    print("  ✓ Heatmap Dendrograms - Core topological structure")
    print("  ✓ Blob Visualizations - Cluster separation analysis")
    print("  ✓ Sankey Flow Diagrams - Cluster evolution tracking")
    print("  ✓ Persistence Analysis - Traditional TDA")
    print(
        f"  ✓ Distance Metrics - {len(distance_matrices)} metrics with MDS + heatmaps"
    )


if __name__ == "__main__":
    main()
