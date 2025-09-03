"""
Cluster Analysis Example

Shows how to analyze cluster evolution using HOLE.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

import hole


def main():
    """Example showing cluster flow analysis."""

    # Generate sample data with clear clusters
    np.random.seed(42)
    points, true_labels = make_blobs(
        n_samples=40, centers=3, n_features=2, cluster_std=1.0, random_state=42
    )

    print(f"Generated {len(points)} points in {len(np.unique(true_labels))} clusters")

    # Create distance matrix
    dist_matrix = hole.euclidean_distance(points)

    # Create cluster flow analyzer
    analyzer = hole.ClusterFlowAnalyzer(dist_matrix, max_thresholds=5)

    # Compute cluster evolution
    print("Computing cluster evolution...")
    evolution = analyzer.compute_cluster_evolution(true_labels)

    print("Cluster evolution computed!")
    print(f"Available keys: {list(evolution.keys())}")

    # Create basic visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original data
    scatter = axes[0].scatter(points[:, 0], points[:, 1], c=true_labels, cmap="viridis")
    axes[0].set_title("Original Data with True Labels")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    plt.colorbar(scatter, ax=axes[0])

    # Create HOLE visualizer for persistence diagram
    visualizer = hole.HOLEVisualizer(distance_matrix_input=dist_matrix)
    visualizer.plot_persistence_diagram(ax=axes[1], title="Persistence Diagram")

    plt.tight_layout()
    plt.savefig("cluster_analysis_example.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Cluster analysis example completed!")


if __name__ == "__main__":
    main()
