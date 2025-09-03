"""
Simple HOLE Library Usage Example

This demonstrates basic usage of the HOLE library for topological data analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

import hole


def main():
    """Simple example showing basic HOLE functionality."""

    # Generate sample data
    print("Generating sample data...")
    points, labels = make_blobs(
        n_samples=50, centers=3, n_features=2, cluster_std=1.5, random_state=42
    )

    print(f"Created {len(points)} points with {points.shape[1]} features")

    # Create HOLE visualizer
    print("Creating HOLE visualizer...")
    visualizer = hole.HOLEVisualizer(point_cloud=points)

    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Persistence diagram
    visualizer.plot_persistence_diagram(ax=axes[0], title="Persistence Diagram")

    # 2. Persistence barcode
    visualizer.plot_persistence_barcode(ax=axes[1], title="Persistence Barcode")

    # 3. PCA visualization with true labels
    visualizer.plot_dimensionality_reduction(
        method="pca", ax=axes[2], true_labels=labels, title="PCA Visualization"
    )

    plt.tight_layout()
    plt.savefig("simple_hole_example.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Example completed! Check 'simple_hole_example.png' for results.")


if __name__ == "__main__":
    main()
