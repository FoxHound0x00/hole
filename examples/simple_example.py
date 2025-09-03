"""
HOLE Library - Core Visualizations Demo

This demonstrates HOLE's main visualization capabilities:
- Sankey diagrams showing cluster evolution
- Stacked bar charts for threshold analysis  
- Heatmap dendrograms for distance matrix visualization
- Blob visualizations with convex hulls
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

import hole


def main():
    """Demo of HOLE's core visualization capabilities."""

    # Generate sample data with clear cluster structure
    print("Generating sample data...")
    points, true_labels = make_blobs(
        n_samples=60, centers=4, n_features=3, cluster_std=1.2, random_state=42
    )

    print(f"Created {len(points)} points in {len(np.unique(true_labels))} clusters")

    # Compute distance matrix
    dist_matrix = hole.euclidean_distance(points)

    print("\n=== 1. SANKEY DIAGRAM - Cluster Evolution ===")
    # Create cluster flow analyzer for Sankey diagrams
    analyzer = hole.ClusterFlowAnalyzer(dist_matrix, max_thresholds=6)
    evolution = analyzer.compute_cluster_evolution(true_labels)

    # Create Sankey diagram
    from hole.visualization.cluster_flow import ComponentEvolutionVisualizer
    
    if evolution.get("components_") and evolution.get("labels_"):
        comp_viz = ComponentEvolutionVisualizer(
            evolution["components_"], evolution["labels_"]
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        first_key = list(evolution["components_"].keys())[0]
        comp_viz.plot_sankey(
            first_key,
            original_labels=true_labels,
            ax=ax,
            title="Cluster Evolution Flow (Sankey Diagram)",
            gray_second_layer=True,
        )
        plt.savefig("sankey_diagram.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("\n=== 2. STACKED BAR CHART - Threshold Analysis ===")
        # Create stacked bar chart
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        comp_viz.plot_stacked_bars(
            first_key,
            original_labels=true_labels,
            ax=ax,
            title="Cluster Evolution by Threshold (Stacked Bars)",
            gray_second_layer=True,
        )
        plt.savefig("stacked_bars.png", dpi=300, bbox_inches="tight")
        plt.show()

    print("\n=== 3. HEATMAP DENDROGRAM - Distance Matrix ===")
    # Create heatmap with dendrogram
    viz = hole.HOLEVisualizer(distance_matrix_input=dist_matrix)
    heatmap_viz = viz.get_persistence_dendrogram_visualizer(
        distance_matrix=dist_matrix
    )
    heatmap_viz.compute_persistence()

    # Plot heatmap with dendrogram
    heatmap_viz.plot_dendrogram_with_heatmap(
        labels=[f"P{i}" for i in range(len(points))],
        title="Distance Matrix Heatmap with Hierarchical Clustering",
        figsize=(12, 10),
    )
    plt.savefig("heatmap_dendrogram.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n=== 4. BLOB VISUALIZATION - Convex Hulls ===")
    # Create blob visualization with convex hulls
    blob_viz = hole.BlobVisualizer(figsize=(12, 8))
    
    # Use PCA for 2D visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    points_2d = pca.fit_transform(points)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot points colored by true labels
    scatter = ax.scatter(
        points_2d[:, 0], points_2d[:, 1], 
        c=true_labels, cmap='viridis', s=50, alpha=0.7
    )
    
    # Add convex hulls around each cluster
    for cluster_id in np.unique(true_labels):
        cluster_mask = true_labels == cluster_id
        if np.sum(cluster_mask) >= 3:
            cluster_points = points_2d[cluster_mask]
            hull_boundary = blob_viz._create_blob_boundary(
                cluster_points, method="convex", padding_factor=0.1
            )
            if hull_boundary is not None:
                ax.fill(
                    hull_boundary[:, 0], hull_boundary[:, 1],
                    alpha=0.2, edgecolor='black', linewidth=2
                )

    ax.set_title("Blob Visualization with Convex Hulls")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, label="Cluster")
    plt.savefig("blob_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\n🎉 HOLE Core Visualizations Complete!")
    print("Generated files:")
    print("- sankey_diagram.png - Shows how clusters merge/split")
    print("- stacked_bars.png - Threshold-based cluster analysis")  
    print("- heatmap_dendrogram.png - Hierarchical clustering")
    print("- blob_visualization.png - Convex hull cluster boundaries")


if __name__ == "__main__":
    main()
