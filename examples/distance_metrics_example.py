"""
Distance Metrics Example

Shows how to use different distance metrics with the HOLE library.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

import hole


def main():
    """Example showing different distance metrics."""

    # Generate sample data
    np.random.seed(42)
    points, labels = make_blobs(n_samples=30, centers=2, n_features=3, random_state=42)

    print(f"Generated {len(points)} points")

    # Test different distance metrics
    metrics = ["euclidean", "cosine", "manhattan"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

    for i, metric in enumerate(metrics):
        print(f"Computing {metric} distance...")

        # Compute distance matrix
        if metric == "euclidean":
            dist_matrix = hole.euclidean_distance(points)
        elif metric == "cosine":
            dist_matrix = hole.cosine_distance(points)
        elif metric == "manhattan":
            dist_matrix = hole.manhattan_distance(points)

        # Create visualizer with distance matrix
        visualizer = hole.HOLEVisualizer(
            distance_matrix_input=dist_matrix, distance_metric=metric
        )

        # Plot persistence diagram
        visualizer.plot_persistence_diagram(
            ax=axes[i], title=f"{metric.capitalize()} Distance"
        )

    plt.tight_layout()
    plt.savefig("distance_metrics_example.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Distance metrics example completed!")


if __name__ == "__main__":
    main()
