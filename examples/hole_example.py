"""
Comprehensive HOLE Visualization Library Example

This script demonstrates ALL visualization capabilities of the HOLE library:
- Persistence diagrams and barcodes
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
from sklearn.metrics import pairwise_distances

from hole import HOLEVisualizer
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
    f"Generated {n_samples} points with {n_features} features in {n_centers} clusters"
)

# Create HOLEVisualizer from point cloud
hole_viz = HOLEVisualizer(point_cloud=points, distance_metric="euclidean")

print("\n=== PART 1: PERSISTENCE VISUALIZATIONS ===")

# 1. Persistence Diagrams and Barcodes
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Persistence Homology Visualizations", fontsize=16)

ax1 = hole_viz.plot_persistence_diagram(ax=axes[0], title="Persistence Diagram")
ax2 = hole_viz.plot_persistence_barcode(ax=axes[1], title="Persistence Barcode")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/persistence_visualizations.png", dpi=300, bbox_inches="tight"
)
plt.show()

print("\n=== PART 2: DIMENSIONALITY REDUCTION ===")

# 2. All Dimensionality Reduction Methods (PCA, t-SNE, MDS)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Dimensionality Reduction Visualizations", fontsize=16)

ax1 = hole_viz.plot_dimensionality_reduction(
    method="pca", ax=axes[0], true_labels=true_labels, title="PCA"
)
ax2 = hole_viz.plot_dimensionality_reduction(
    method="tsne", ax=axes[1], true_labels=true_labels, title="t-SNE"
)
ax3 = hole_viz.plot_dimensionality_reduction(
    method="mds", ax=axes[2], true_labels=true_labels, title="MDS"
)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/dimensionality_reduction.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== PART 3: HEATMAPS AND DENDROGRAMS ===")

# 3. Distance Matrix Heatmaps and Dendrograms
heatmap_dendro_viz = hole_viz.get_heatmap_dendrogram_visualizer(
    distance_matrix=hole_viz.distance_matrix
)
heatmap_dendro_viz.compute_persistence()

# Create heatmap with dendrogram
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
fig.suptitle("Distance Matrix Heatmap with Dendrogram", fontsize=16)

# Plot heatmap with dendrogram (creates its own figure)
heatmap_dendro_viz.plot_dendrogram_with_heatmap(
    labels=[f"P{i}" for i in range(n_samples)],
    title="Euclidean Distance Matrix with Dendrogram",
    figsize=(16, 8),
)
plt.savefig(f"{OUTPUT_DIR}/heatmap_dendrogram.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== PART 4: CLUSTER FLOW ANALYSIS ===")

# 4. Cluster Flow Analysis (Sankey diagrams and stacked charts)
# Need to use ClusterFlowAnalyzer for this
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
    fig.suptitle("Cluster Evolution Sankey Diagram", fontsize=16)

    # Get the first distance metric key
    first_key = list(components_.keys())[0]
    comp_viz.plot_sankey(
        first_key, original_labels=true_labels, ax=ax, title="Cluster Evolution Flow"
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sankey_diagram.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Plot Stacked Bar Chart
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.suptitle("Cluster Evolution Stacked Bar Chart", fontsize=16)

    comp_viz.plot_stacked_bars(
        first_key, original_labels=true_labels, ax=ax, title="Cluster Evolution Stages"
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stacked_bar_chart.png", dpi=300, bbox_inches="tight")
    plt.show()
else:
    print("No cluster evolution data available for flow visualizations")

print("\n=== PART 5: SCATTER HULL VISUALIZATIONS ===")

# 5. Scatter Hull Visualizations (Blob separation analysis)
scatter_hull_viz = hole_viz.get_scatter_hull_visualizer()

# For scatter hull viz, we need to analyze at specific thresholds
if cluster_evolution and "labels_" in cluster_evolution:
    labels_dict = cluster_evolution["labels_"]
    first_key = list(labels_dict.keys())[0]

    if first_key in labels_dict:
        # Get labels at middle threshold
        threshold_keys = sorted(labels_dict[first_key].keys())
        if len(threshold_keys) >= 2:
            middle_idx = len(threshold_keys) // 2
            middle_threshold = threshold_keys[middle_idx]
            cluster_labels = labels_dict[first_key][middle_threshold]

            # Create side-by-side PCA and blob visualizations with consistent colors
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
            fig.suptitle("PCA vs Blob Visualization Comparison", fontsize=16)

            # Perform PCA for 2D visualization
            from sklearn.decomposition import PCA

            from hole.visualization.scatter_hull import generate_consistent_colors

            pca = PCA(n_components=2, random_state=42)
            points_2d = pca.fit_transform(points)

            # Generate consistent colors for both visualizations
            # Use cluster_labels (threshold-based) for hulls, true_labels for points
            unique_clusters = np.unique(cluster_labels)
            unique_true_labels = sorted(set(true_labels))
            n_clusters = len(unique_clusters)
            n_true_labels = len(unique_true_labels)

            # Get consistent cluster colors (for threshold-based hulls) and true label colors (for points)
            cluster_colors = generate_consistent_colors(n_clusters, include_noise=True)
            true_label_colors = generate_consistent_colors(
                n_true_labels, include_noise=False
            )

            # Create color mappings
            cluster_color_map = {
                cluster_id: cluster_colors[i]
                for i, cluster_id in enumerate(unique_clusters)
            }
            true_label_color_map = {
                label: true_label_colors[i]
                for i, label in enumerate(unique_true_labels)
            }

            # === LEFT PLOT: PCA with manual hull drawing ===
            # Plot points colored by true labels using discrete colors
            point_colors = [true_label_color_map[label] for label in true_labels]
            scatter1 = ax1.scatter(
                points_2d[:, 0],
                points_2d[:, 1],
                c=point_colors,
                s=80,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
                zorder=2,
            )

            # Draw cluster hulls with consistent colors (using threshold-based clusters)
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise
                    continue

                cluster_mask = cluster_labels == cluster_id
                cluster_points = points_2d[cluster_mask]

                if len(cluster_points) >= 3:
                    from scipy.spatial import ConvexHull

                    try:
                        hull = ConvexHull(cluster_points)
                        hull_color = cluster_color_map[cluster_id]

                        # Draw hull boundary
                        for simplex in hull.simplices:
                            ax1.plot(
                                cluster_points[simplex, 0],
                                cluster_points[simplex, 1],
                                color=hull_color,
                                linewidth=3,
                                alpha=0.7,
                            )

                        # Add cluster label
                        center = np.mean(cluster_points, axis=0)
                        ax1.text(
                            center[0],
                            center[1],
                            f"C{cluster_id}",
                            fontsize=12,
                            fontweight="bold",
                            color="black",
                            ha="center",
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                alpha=0.9,
                                edgecolor="black",
                            ),
                        )
                    except:
                        pass

            ax1.set_xlabel("PC1", fontsize=12)
            ax1.set_ylabel("PC2", fontsize=12)
            ax1.set_title("PCA - True Labels", fontsize=14)
            # Keep axes for PCA vs blob comparison
            ax1.grid(True, alpha=0.3)
            for spine in ax1.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(0.8)

            # === RIGHT PLOT: Blob visualization using BlobVisualizer ===
            from hole.visualization.scatter_hull import BlobVisualizer

            # Create blob visualizer with shared colors
            blob_viz = BlobVisualizer(
                figsize=(12, 10), shared_cluster_colors=cluster_colors
            )

            # Plot points colored by true labels using discrete colors
            point_colors = [true_label_color_map[label] for label in true_labels]
            scatter2 = ax2.scatter(
                points_2d[:, 0],
                points_2d[:, 1],
                c=point_colors,
                s=80,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
                zorder=2,
            )

            # Draw cluster hulls using blob visualizer's method
            for cluster_id in unique_clusters:
                if cluster_id == -1:  # Skip noise
                    continue

                cluster_mask = cluster_labels == cluster_id
                cluster_points = points_2d[cluster_mask]

                if len(cluster_points) >= 3:
                    hull_color = cluster_color_map[cluster_id]

                    # Create smooth hull using blob visualizer's method
                    hull_points = blob_viz._create_blob_boundary(
                        cluster_points, method="smooth"
                    )
                    if hull_points is not None:
                        from matplotlib.patches import Polygon

                        hull_polygon = Polygon(
                            hull_points,
                            alpha=blob_viz.alpha_hull,
                            facecolor=hull_color,
                            edgecolor="black",
                            linewidth=2.5,
                            linestyle="-",
                            zorder=1,
                        )
                        ax2.add_patch(hull_polygon)

                        # Add cluster label
                        center = np.mean(cluster_points, axis=0)
                        ax2.text(
                            center[0],
                            center[1],
                            f"C{cluster_id}",
                            fontsize=12,
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

            ax2.set_xlabel("PC1", fontsize=12)
            ax2.set_ylabel("PC2", fontsize=12)
            ax2.set_title(
                f"PCA - Cluster Hulls (Threshold {middle_threshold})", fontsize=14
            )
            # Keep axes for PCA vs blob comparison
            ax2.grid(True, alpha=0.3)
            for spine in ax2.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(0.8)

            plt.tight_layout()
            plt.savefig(
                f"{OUTPUT_DIR}/pca_vs_blob_comparison.png", dpi=300, bbox_inches="tight"
            )
plt.show()

print("\n=== PART 6: MULTI-METRIC COMPARISON ===")

# 6. Compare different distance metrics
metrics = ["euclidean", "manhattan", "cosine"]
fig, axes = plt.subplots(len(metrics), 3, figsize=(18, 12))
fig.suptitle("Distance Metrics Comparison", fontsize=16)

for i, metric in enumerate(metrics):
    print(f"Processing {metric} metric...")

    # Create HOLEVisualizer with different metric
    hole_metric = HOLEVisualizer(point_cloud=points, distance_metric=metric)

    # Plot persistence barcode
    ax1 = hole_metric.plot_persistence_barcode(
        ax=axes[i, 0], title=f"{metric.title()} Barcode"
    )

    # Plot persistence diagram
    ax2 = hole_metric.plot_persistence_diagram(
        ax=axes[i, 1], title=f"{metric.title()} Persistence"
    )

    # Plot PCA
    ax3 = hole_metric.plot_dimensionality_reduction(
        ax=axes[i, 2],
        true_labels=true_labels,
        title=f"{metric.title()} PCA",
        method="pca",
    )

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/metric_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== ALL VISUALIZATIONS COMPLETED ===")
print("Generated files:")
print(f"- {OUTPUT_DIR}/persistence_visualizations.png")
print(f"- {OUTPUT_DIR}/dimensionality_reduction.png")
print(f"- {OUTPUT_DIR}/heatmap_dendrogram.png")
print(f"- {OUTPUT_DIR}/sankey_diagram.png")
print(f"- {OUTPUT_DIR}/stacked_bar_chart.png")
print(f"- {OUTPUT_DIR}/pca_vs_blob_comparison.png")
print(f"- {OUTPUT_DIR}/metric_comparison.png")
print("\nAll HOLE visualization capabilities demonstrated!")
