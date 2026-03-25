"""
Demo script showcasing the correct outlier class detection.

This script demonstrates how outlier classes are detected based on percentage thresholds
within persistent homology clusters, not based on class mismatch.

Outlier classes are classes that make up less than a specified percentage of points
within each persistent homology cluster.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

import hole

def create_realistic_mixed_clusters():
    """Create a dataset that mimics real-world scenarios with mixed classes in PH clusters."""
    
    # Create 2 spatial clusters that will be detected by persistent homology
    centers = np.array([[-3, 0], [3, 0]])
    
    # Cluster 1: Mixed classes with different proportions
    # - 70 points of class 0 (majority)
    # - 20 points of class 1 (above 10% threshold)
    # - 5 points of class 2 (below 10% threshold - outlier class)
    points1, _ = make_blobs(n_samples=95, centers=[centers[0]], cluster_std=1.2, random_state=42)
    labels1 = np.concatenate([
        np.zeros(70, dtype=int),      # Class 0: 70/95 = 73.7%
        np.ones(20, dtype=int),       # Class 1: 20/95 = 21.1%  
        np.full(5, 2, dtype=int)      # Class 2: 5/95 = 5.3% (outlier class)
    ])
    
    # Cluster 2: Different class distribution
    # - 60 points of class 1 (majority)
    # - 25 points of class 2 (above 10% threshold)
    # - 8 points of class 0 (below 10% threshold - outlier class)
    points2, _ = make_blobs(n_samples=93, centers=[centers[1]], cluster_std=1.2, random_state=43)
    labels2 = np.concatenate([
        np.ones(60, dtype=int),       # Class 1: 60/93 = 64.5%
        np.full(25, 2, dtype=int),    # Class 2: 25/93 = 26.9%
        np.zeros(8, dtype=int)        # Class 0: 8/93 = 8.6% (outlier class)
    ])
    
    # Combine and shuffle
    all_points = np.vstack([points1, points2])
    all_labels = np.concatenate([labels1, labels2])
    
    indices = np.random.permutation(len(all_points))
    all_points = all_points[indices]
    all_labels = all_labels[indices]
    
    return all_points, all_labels

def main():
    """Demo blob contour visualization with outlier class detection."""
    
    # Create output directory
    output_dir = "blob_contour_demo_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== HOLE Blob Contour Demo ===\n")
    
    # Generate realistic mixed cluster dataset
    print("Generating dataset with mixed classes in persistent homology clusters...")
    points, labels = create_realistic_mixed_clusters()
    print(f"Created {len(points)} points with 3 classes")
    print(f"Overall class distribution: {np.bincount(labels)}")
    print()
    
    # Create HOLE visualizer
    print("Creating HOLE visualizer...")
    visualizer = hole.HOLEVisualizer(point_cloud=points, distance_metric="euclidean")
    
    # Get cluster evolution to find a good threshold for 2 clusters
    from hole.visualization.cluster_flow import ClusterFlowAnalyzer
    analyzer = ClusterFlowAnalyzer(visualizer.distance_matrix, max_thresholds=6)
    cluster_evolution = analyzer.compute_cluster_evolution(labels)
    
    # Get a threshold that creates 2 clusters (matching our spatial setup)
    euclidean_labels = cluster_evolution["labels_"]["Euclidean"]
    thresholds = sorted([float(t) for t in euclidean_labels.keys()])
    
    # Find threshold that gives us 2 clusters
    best_threshold = None
    for threshold in thresholds:
        cluster_labels = euclidean_labels[str(threshold)]
        n_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
        if n_clusters == 2:
            best_threshold = threshold
            break
    
    if best_threshold is None:
        # Fallback to a threshold that should give us 2-3 clusters
        best_threshold = thresholds[len(thresholds)//3]
    
    print(f"Using threshold: {best_threshold:.3f}")
    
    # Analyze cluster composition
    cluster_labels = euclidean_labels[str(best_threshold)]
    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
    print(f"Found {len(unique_clusters)} persistent homology clusters")
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_true_labels = labels[cluster_mask]
        cluster_size = len(cluster_true_labels)
        unique_classes, class_counts = np.unique(cluster_true_labels, return_counts=True)
        
        print(f"\n  PH Cluster {cluster_id} ({cluster_size} points):")
        for class_id, count in zip(unique_classes, class_counts):
            percentage = (count / cluster_size) * 100
            status = "OUTLIER CLASS" if percentage < 10 else "majority/contour class"
            print(f"    Class {class_id}: {count} points ({percentage:.1f}%) - {status}")
    print()
    
    # Test different outlier thresholds
    outlier_thresholds = [0.05, 0.10, 0.15]  # 5%, 10%, 15%
    
    for threshold in outlier_thresholds:
        print(f"Creating visualization with {threshold*100:.0f}% outlier threshold...")
        
        blob_viz = visualizer.get_blob_visualizer(
            figsize=(12, 10), 
            outlier_percentage=threshold,
            show_contours=True,
            alpha_hull=0.3  # Semi-transparent blobs
        )
        
        fig = blob_viz.plot_pca_with_cluster_hulls(
            points,
            labels,
            best_threshold,
            save_path=f"{output_dir}/outlier_threshold_{threshold*100:.0f}pct.png",
            title=f"Outlier Classes: <{threshold*100:.0f}% of cluster size",
        )
        plt.close(fig)
    
    print(f"\n=== Blob Contour Demo Complete ===")
    print(f"All visualizations saved to: {output_dir}/")
    print("Files created:")
    for threshold in outlier_thresholds:
        print(f"  📊 outlier_threshold_{threshold*100:.0f}pct.png - Classes <{threshold*100:.0f}% are outliers")
    
    print("\nKey concepts demonstrated:")
    print("  ✓ Persistent homology clusters are based on DISTANCE, not class")
    print("  ✓ Outlier classes are classes with <threshold% of points in each PH cluster")
    print("  ✓ Majority classes (≥threshold%) get contour plots")
    print("  ✓ Outlier classes (<threshold%) get scatter plots")
    print("  ✓ Same class can be majority in one cluster, outlier in another")
    print("  ✓ Contours are colored by class, confined within blob boundaries")

if __name__ == "__main__":
    main()
