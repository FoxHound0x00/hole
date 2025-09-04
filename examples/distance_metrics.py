#!/usr/bin/env python3
"""
Comprehensive experiments to understand different distance metrics. 
Distance Metrics:
- Euclidean, Mahalanobis, Cosine
- Density-normalized variants: dn_euclidean, dn_mahalanobis, dn_cosine
- Geodesic

Data Structures:
- Isotropic clusters (dense, sparse, outliers, separable, non-separable)
- Hypersphere (dense, sparse, outliers, separable, non-separable)
- Elliptical clusters (dense, sparse, outliers, separable, non-separable)
- Swiss roll (dense, sparse, outliers, separable, non-separable)
- Tight blobs (outliers, no outliers)

For each combination, generates:
- Persistence diagrams and barcodes
- Sankey diagrams (cluster flow evolution)
- Stacked bar charts (cluster evolution)
- Heatmap dendrograms
- Blob visualizations
"""

import os
import warnings

warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_swiss_roll

from hole.core import distance_metrics
from hole.visualization.cluster_flow import (
    ClusterFlowAnalyzer,
    ComponentEvolutionVisualizer,
)
from hole.visualizer import HOLEVisualizer

# Set random seed for reproducibility
np.random.seed(42)


def create_isotropic_clusters(
    n_samples=500,
    dense=True,
    outliers=True,
    separable=True,
):
    """Create isotropic (spherical) clusters with different characteristics."""
    n_features = 10
    n_centers = 4

    if dense:
        cluster_std = 0.8 if separable else 2.5
    else:
        cluster_std = 1.5 if separable else 4.0

    # Generate base clusters
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=42,
    )

    if outliers:
        # Add 10% outliers
        n_outliers = int(0.1 * n_samples)
        outlier_range = np.max(X) - np.min(X)
        outliers_X = np.random.uniform(
            np.min(X) - outlier_range * 0.5,
            np.max(X) + outlier_range * 0.5,
            (n_outliers, n_features),
        )
        outliers_y = np.full(n_outliers, -1)  # Outlier class

        X = np.vstack([X, outliers_X])
        y = np.hstack([y, outliers_y])

    return X, y


def create_hypersphere(
    n_samples=500,
    dense=True,
    outliers=True,
    separable=True,
):
    """Create hypersphere structure with different characteristics."""
    n_features = 8

    if dense:
        noise = 0.1 if separable else 0.4
    else:
        noise = 0.3 if separable else 0.8

    # Generate concentric hyperspheres
    inner_samples = n_samples // 3
    middle_samples = n_samples // 3
    outer_samples = n_samples - inner_samples - middle_samples

    # Inner sphere
    inner_X = np.random.randn(inner_samples, n_features)
    inner_X = (
        inner_X
        / np.linalg.norm(inner_X, axis=1, keepdims=True)
        * (1.0 + np.random.normal(0, noise, (inner_samples, 1)))
    )
    inner_y = np.zeros(inner_samples)

    # Middle sphere
    middle_X = np.random.randn(middle_samples, n_features)
    middle_radius = 3.0 if separable else 2.0
    middle_X = (
        middle_X
        / np.linalg.norm(middle_X, axis=1, keepdims=True)
        * (middle_radius + np.random.normal(0, noise, (middle_samples, 1)))
    )
    middle_y = np.ones(middle_samples)

    # Outer sphere
    outer_X = np.random.randn(outer_samples, n_features)
    outer_radius = 5.0 if separable else 3.5
    outer_X = (
        outer_X
        / np.linalg.norm(outer_X, axis=1, keepdims=True)
        * (outer_radius + np.random.normal(0, noise, (outer_samples, 1)))
    )
    outer_y = np.ones(outer_samples) * 2

    X = np.vstack([inner_X, middle_X, outer_X])
    y = np.hstack([inner_y, middle_y, outer_y])

    if outliers:
        # Add outliers at random positions
        n_outliers = int(0.1 * n_samples)
        outlier_range = np.max(X) - np.min(X)
        outliers_X = np.random.uniform(
            np.min(X) - outlier_range * 0.3,
            np.max(X) + outlier_range * 0.3,
            (n_outliers, n_features),
        )
        outliers_y = np.full(n_outliers, -1)

        X = np.vstack([X, outliers_X])
        y = np.hstack([y, outliers_y])

    return X, y


def create_elliptical_clusters(
    n_samples=500,
    dense=True,
    outliers=True,
    separable=True,
):
    """Create elliptical clusters with different characteristics."""
    n_features = 6
    n_centers = 3

    # Create base isotropic clusters first
    cluster_std = 0.5 if dense else 1.2
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=cluster_std,
        random_state=42,
    )

    # Transform to elliptical by stretching along different axes
    transformation_matrices = []
    for i in range(n_centers):
        # Create random transformation matrix for each cluster
        A = np.random.randn(n_features, n_features)
        U, s, Vt = np.linalg.svd(A)

        if separable:
            # Strong elliptical distortion with clear separation
            s = np.array([5.0, 2.0, 1.0, 0.8, 0.5, 0.3])
        else:
            # Moderate distortion with overlap
            s = np.array([2.5, 1.5, 1.0, 0.9, 0.8, 0.7])

        transformation_matrices.append(U @ np.diag(s) @ Vt)

    # Apply transformations to each cluster
    for i in range(n_centers):
        mask = y == i
        cluster_points = X[mask]
        center = np.mean(cluster_points, axis=0)
        centered_points = cluster_points - center
        transformed_points = (transformation_matrices[i] @ centered_points.T).T
        X[mask] = transformed_points + center

    if outliers:
        n_outliers = int(0.1 * n_samples)
        outlier_range = np.max(X) - np.min(X)
        outliers_X = np.random.uniform(
            np.min(X) - outlier_range * 0.4,
            np.max(X) + outlier_range * 0.4,
            (n_outliers, n_features),
        )
        outliers_y = np.full(n_outliers, -1)

        X = np.vstack([X, outliers_X])
        y = np.hstack([y, outliers_y])

    return X, y


def create_swiss_roll_structure(
    n_samples=500,
    dense=True,
    outliers=True,
    separable=True,
):
    """Create Swiss roll manifold structure
    with different characteristics."""
    if dense:
        noise = 0.1 if separable else 0.4
    else:
        noise = 0.5 if separable else 1.0

    # Generate Swiss roll
    X, color = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)

    # Create labels based on position along the roll
    if separable:
        # Clear separation into 3 regions
        y = np.digitize(
            color,
            bins=[
                color.min() + (color.max() - color.min()) * 0.33,
                color.min() + (color.max() - color.min()) * 0.66,
            ],
        )
    else:
        # Overlapping regions
        y = np.digitize(
            color,
            bins=[
                color.min() + (color.max() - color.min()) * 0.4,
                color.min() + (color.max() - color.min()) * 0.6,
            ],
        )

    if outliers:
        n_outliers = int(0.1 * n_samples)
        outlier_range = np.max(X) - np.min(X)
        outliers_X = np.random.uniform(
            np.min(X) - outlier_range * 0.5,
            np.max(X) + outlier_range * 0.5,
            (n_outliers, 3),
        )
        outliers_y = np.full(n_outliers, -1)

        X = np.vstack([X, outliers_X])
        y = np.hstack([y, outliers_y])

    return X, y


def create_tight_blobs(
    n_samples=500,
    outliers=True,
):
    """Create very tight, well-separated blobs."""
    n_features = 8
    n_centers = 5

    # Very tight clusters, well separated
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        n_features=n_features,
        cluster_std=0.3,
        center_box=(-15.0, 15.0),
        random_state=42,
    )

    if outliers:
        n_outliers = int(0.15 * n_samples)  # More outliers for tight blobs
        outlier_range = np.max(X) - np.min(X)
        outliers_X = np.random.uniform(
            np.min(X) - outlier_range * 0.3,
            np.max(X) + outlier_range * 0.3,
            (n_outliers, n_features),
        )
        outliers_y = np.full(n_outliers, -1)

        X = np.vstack([X, outliers_X])
        y = np.hstack([y, outliers_y])

    return X, y


def compute_all_distance_matrices(X):
    """Compute all distance matrices for the given data."""
    print("    Computing distance matrices...")

    distance_matrices = {
        "euclidean": distance_metrics.distance_matrix(X, metric="euclidean"),
        "cosine": distance_metrics.distance_matrix(X, metric="cosine"),
        "mahalanobis": distance_metrics.distance_matrix(X, metric="mahalanobis"),
    }

    # Compute density normalized versions
    distance_matrices.update(
        {
            "dn_euclidean": distance_metrics.density_normalized_distance(
                X, distance_matrices["euclidean"]
            ),
            "dn_cosine": distance_metrics.density_normalized_distance(
                X, distance_matrices["cosine"]
            ),
            "dn_mahalanobis": distance_metrics.density_normalized_distance(
                X, distance_matrices["mahalanobis"]
            ),
        }
    )

    # Handle geodesic distances with error checking
    try:
        geodesic_dist = distance_metrics.geodesic_distances(X)
        # Replace infinite values with large finite values
        if np.any(np.isinf(geodesic_dist)):
            geodesic_dist[np.isinf(geodesic_dist)] = (
                np.nanmax(geodesic_dist[np.isfinite(geodesic_dist)]) * 10
            )
        distance_matrices["geodesic"] = geodesic_dist
        print("      ✓ All distance metrics computed successfully")
    except Exception as e:
        print(f"      ! Geodesic distance failed: {e}")
        print("      ✓ Other distance metrics computed successfully")

    return distance_matrices


def generate_all_visualizations(
    X,
    y,
    distance_matrices,
    structure_name,
    variant_name,
    base_dir,
):
    """Generate all visualizations for each distance metric."""
    variant_dir = os.path.join(base_dir, structure_name, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    print(
        f"    Generating visualizations for {len(distance_matrices)} distance metrics..."
    )

    for metric_name, dist_matrix in distance_matrices.items():
        print(f"      Processing {metric_name}...")

        try:
            # Create HOLEVisualizer for persistence analysis (using distance matrix)
            viz = HOLEVisualizer(
                distance_matrix_input=dist_matrix,
                max_dimension=1,
                max_edge_length=np.inf,
            )

            # Create separate HOLEVisualizer for PCA (using original point cloud)
            viz_pca = HOLEVisualizer(
                point_cloud=X,
                distance_metric="euclidean",
                max_dimension=1,
                max_edge_length=np.inf,
            )

            # 1. Persistence visualizations (diagrams, barcodes, PCA, MDS)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(
                f"{structure_name} - {variant_name} - {metric_name.upper()}",
                fontsize=14,
            )

            viz.plot_persistence_diagram(ax=axes[0, 0], title="Persistence Diagram")
            viz.plot_persistence_barcode(ax=axes[0, 1], title="Persistence Barcode")
            viz_pca.plot_dimensionality_reduction(
                method="pca",
                ax=axes[1, 0],
                true_labels=y,
                title="PCA (Original Features)",
            )
            viz.plot_dimensionality_reduction(
                method="mds",
                ax=axes[1, 1],
                true_labels=y,
                title="MDS (Distance Matrix)",
            )

            plt.tight_layout()
            plt.savefig(
                f"{variant_dir}/persistence_viz_{metric_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # 2. Cluster flow evolution
            analyzer = ClusterFlowAnalyzer(dist_matrix, max_thresholds=6)
            cluster_evolution = analyzer.compute_cluster_evolution(y)

            components_ = cluster_evolution.get("components_", {})
            labels_ = cluster_evolution.get("labels_", {})

            if components_ and labels_:
                comp_viz = ComponentEvolutionVisualizer(components_, labels_)
                first_key = list(components_.keys())[0]

                # Sankey diagram
                fig, ax = plt.subplots(1, 1, figsize=(20, 12))
                fig.suptitle(
                    f"{structure_name} - {variant_name} - {metric_name.upper()} - Cluster Evolution",
                    fontsize=14,
                )
                comp_viz.plot_sankey(
                    first_key,
                    original_labels=y,
                    ax=ax,
                    # title="Cluster Flow Evolution",
                    gray_second_layer=True,
                    show_true_labels_text=False,
                    show_filtration_text=False,
                )
                plt.tight_layout()
                plt.savefig(
                    f"{variant_dir}/sankey_{metric_name}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

                # Stacked bar chart
                fig, ax = plt.subplots(1, 1, figsize=(16, 10))
                fig.suptitle(
                    f"{structure_name} - {variant_name} - {metric_name.upper()} - Evolution Stages",
                    fontsize=14,
                )
                comp_viz.plot_stacked_bars(
                    first_key,
                    original_labels=y,
                    ax=ax,
                    # title="Cluster Evolution Stages",
                    gray_second_layer=True,
                    show_true_labels_text=False,
                    show_filtration_text=False,
                )
                plt.tight_layout()
                plt.savefig(
                    f"{variant_dir}/stacked_bars_{metric_name}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            # 3. Heatmap dendrograms
            heatmap_dendro_viz = viz.get_persistence_dendrogram_visualizer(
                distance_matrix=dist_matrix
            )
            heatmap_dendro_viz.compute_persistence()

            heatmap_dendro_viz.plot_dendrogram_with_heatmap(
                labels=[f"P{i}" for i in range(len(X))],
                # title=f"{structure_name} - {variant_name} - {metric_name.upper()} Distance Matrix",
                figsize=(16, 8),
            )
            plt.savefig(
                f"{variant_dir}/heatmap_dendrogram_{metric_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # 4. Blob visualization
            blob_viz = viz.get_blob_visualizer(figsize=(12, 8))

            try:
                # Find a good threshold from cluster evolution
                if cluster_evolution and "labels_" in cluster_evolution:
                    labels_dict = cluster_evolution["labels_"]
                    if labels_dict:
                        first_key = list(labels_dict.keys())[0]
                        thresholds = sorted(
                            [float(t) for t in labels_dict[first_key].keys()]
                        )
                        if len(thresholds) >= 2:
                            # Use middle threshold
                            middle_threshold = thresholds[len(thresholds) // 2]

                            fig = blob_viz.plot_pca_with_cluster_hulls(
                                X,
                                y,
                                middle_threshold,
                                save_path=f"{variant_dir}/blob_viz_{metric_name}.png",
                                # title=f"{structure_name} - {variant_name} - {metric_name.upper()} (Threshold: {middle_threshold:.3f})"
                            )
                            plt.close(fig)
                        else:
                            print(
                                f"        ! Not enough thresholds for blob visualization: {len(thresholds)}"
                            )
                    else:
                        print("        ! No labels found in cluster evolution")
                else:
                    print("        ! No cluster evolution data for blob visualization")
            except Exception as e:
                print(f"        ! Blob visualization failed for {metric_name}: {e}")

            print(f"        ✓ {metric_name} completed")

        except Exception as e:
            print(f"        ✗ {metric_name} failed: {e}")
            continue


def run_comprehensive_analysis():
    """Run the comprehensive analysis across all structures and metrics."""
    print("=" * 80)
    print("COMPREHENSIVE METRIC-STRUCTURE ANALYSIS")
    print("=" * 80)

    base_dir = "comprehensive_analysis_results"
    os.makedirs(base_dir, exist_ok=True)

    # Define all data structure variants
    structure_generators = {
        "isotropic_clusters": [
            (
                "dense_separable_outliers",
                lambda: create_isotropic_clusters(
                    dense=True, separable=True, outliers=True
                ),
            ),
            (
                "dense_separable_no_outliers",
                lambda: create_isotropic_clusters(
                    dense=True, separable=True, outliers=False
                ),
            ),
            (
                "dense_nonseparable_outliers",
                lambda: create_isotropic_clusters(
                    dense=True, separable=False, outliers=True
                ),
            ),
            (
                "dense_nonseparable_no_outliers",
                lambda: create_isotropic_clusters(
                    dense=True, separable=False, outliers=False
                ),
            ),
            (
                "sparse_separable_outliers",
                lambda: create_isotropic_clusters(
                    dense=False, separable=True, outliers=True
                ),
            ),
            (
                "sparse_separable_no_outliers",
                lambda: create_isotropic_clusters(
                    dense=False, separable=True, outliers=False
                ),
            ),
            (
                "sparse_nonseparable_outliers",
                lambda: create_isotropic_clusters(
                    dense=False, separable=False, outliers=True
                ),
            ),
            (
                "sparse_nonseparable_no_outliers",
                lambda: create_isotropic_clusters(
                    dense=False, separable=False, outliers=False
                ),
            ),
        ],
        "hypersphere": [
            (
                "dense_separable_outliers",
                lambda: create_hypersphere(dense=True, separable=True, outliers=True),
            ),
            (
                "dense_separable_no_outliers",
                lambda: create_hypersphere(dense=True, separable=True, outliers=False),
            ),
            (
                "dense_nonseparable_outliers",
                lambda: create_hypersphere(dense=True, separable=False, outliers=True),
            ),
            (
                "dense_nonseparable_no_outliers",
                lambda: create_hypersphere(dense=True, separable=False, outliers=False),
            ),
            (
                "sparse_separable_outliers",
                lambda: create_hypersphere(dense=False, separable=True, outliers=True),
            ),
            (
                "sparse_separable_no_outliers",
                lambda: create_hypersphere(dense=False, separable=True, outliers=False),
            ),
            (
                "sparse_nonseparable_outliers",
                lambda: create_hypersphere(dense=False, separable=False, outliers=True),
            ),
            (
                "sparse_nonseparable_no_outliers",
                lambda: create_hypersphere(
                    dense=False, separable=False, outliers=False
                ),
            ),
        ],
        "elliptical_clusters": [
            (
                "dense_separable_outliers",
                lambda: create_elliptical_clusters(
                    dense=True, separable=True, outliers=True
                ),
            ),
            (
                "dense_separable_no_outliers",
                lambda: create_elliptical_clusters(
                    dense=True, separable=True, outliers=False
                ),
            ),
            (
                "dense_nonseparable_outliers",
                lambda: create_elliptical_clusters(
                    dense=True, separable=False, outliers=True
                ),
            ),
            (
                "dense_nonseparable_no_outliers",
                lambda: create_elliptical_clusters(
                    dense=True, separable=False, outliers=False
                ),
            ),
            (
                "sparse_separable_outliers",
                lambda: create_elliptical_clusters(
                    dense=False, separable=True, outliers=True
                ),
            ),
            (
                "sparse_separable_no_outliers",
                lambda: create_elliptical_clusters(
                    dense=False, separable=True, outliers=False
                ),
            ),
            (
                "sparse_nonseparable_outliers",
                lambda: create_elliptical_clusters(
                    dense=False, separable=False, outliers=True
                ),
            ),
            (
                "sparse_nonseparable_no_outliers",
                lambda: create_elliptical_clusters(
                    dense=False, separable=False, outliers=False
                ),
            ),
        ],
        "swiss_roll": [
            (
                "dense_separable_outliers",
                lambda: create_swiss_roll_structure(
                    dense=True, separable=True, outliers=True
                ),
            ),
            (
                "dense_separable_no_outliers",
                lambda: create_swiss_roll_structure(
                    dense=True, separable=True, outliers=False
                ),
            ),
            (
                "dense_nonseparable_outliers",
                lambda: create_swiss_roll_structure(
                    dense=True, separable=False, outliers=True
                ),
            ),
            (
                "dense_nonseparable_no_outliers",
                lambda: create_swiss_roll_structure(
                    dense=True, separable=False, outliers=False
                ),
            ),
            (
                "sparse_separable_outliers",
                lambda: create_swiss_roll_structure(
                    dense=False, separable=True, outliers=True
                ),
            ),
            (
                "sparse_separable_no_outliers",
                lambda: create_swiss_roll_structure(
                    dense=False, separable=True, outliers=False
                ),
            ),
            (
                "sparse_nonseparable_outliers",
                lambda: create_swiss_roll_structure(
                    dense=False, separable=False, outliers=True
                ),
            ),
            (
                "sparse_nonseparable_no_outliers",
                lambda: create_swiss_roll_structure(
                    dense=False, separable=False, outliers=False
                ),
            ),
        ],
        "tight_blobs": [
            ("with_outliers", lambda: create_tight_blobs(outliers=True)),
            ("no_outliers", lambda: create_tight_blobs(outliers=False)),
        ],
    }

    total_experiments = sum(len(variants) for variants in structure_generators.values())
    experiment_count = 0

    # Run analysis for each structure and variant
    for structure_name, variants in structure_generators.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING: {structure_name.upper()}")
        print(f"{'='*60}")

        for variant_name, generator_func in variants:
            experiment_count += 1
            print(
                f"\n[{experiment_count}/{total_experiments}] Processing {structure_name} - {variant_name}"
            )

            # Generate data
            X, y = generator_func()
            print(f"  Data shape: {X.shape}, Labels: {len(np.unique(y))} classes")

            # Compute distance matrices
            distance_matrices = compute_all_distance_matrices(X)

            # Generate all visualizations
            generate_all_visualizations(
                X, y, distance_matrices, structure_name, variant_name, base_dir
            )

            print(f"  ✓ Completed {structure_name} - {variant_name}")

    print(f"\n{'='*80}")
    print("COMPREHENSIVE ANALYSIS COMPLETED!")
    print(f"Total experiments: {experiment_count}")
    print(f"Results saved in: {base_dir}/")
    print("=" * 80)

    # Generate summary report
    generate_summary_report(base_dir, total_experiments)


def generate_summary_report(base_dir, total_experiments):
    """Generate a summary report of the analysis."""
    report_path = os.path.join(base_dir, "analysis_summary.txt")

    with open(report_path, "w") as f:
        f.write("COMPREHENSIVE METRIC-STRUCTURE ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total experiments conducted: {total_experiments}\n")
        f.write(
            "Distance metrics tested: 7 (euclidean, cosine, mahalanobis, dn_euclidean, dn_cosine, dn_mahalanobis, geodesic)\n\n"
        )

        f.write("Data structures analyzed:\n")
        f.write("- Isotropic clusters (8 variants)\n")
        f.write("- Hypersphere (8 variants)\n")
        f.write("- Elliptical clusters (8 variants)\n")
        f.write("- Swiss roll (8 variants)\n")
        f.write("- Tight blobs (2 variants)\n\n")

        f.write("Visualizations generated for each experiment:\n")
        f.write("- Persistence diagrams and barcodes\n")
        f.write("- PCA and MDS dimensionality reduction plots\n")
        f.write("- Sankey diagrams (cluster flow evolution)\n")
        f.write("- Stacked bar charts (cluster evolution stages)\n")
        f.write("- Heatmap dendrograms\n")
        f.write("- Scatter hull visualizations\n\n")

        f.write("This systematic analysis provides insights into:\n")
        f.write("1. How different distance metrics capture structural properties\n")
        f.write("2. Robustness of topological features across data characteristics\n")
        f.write("3. Optimal metric selection for specific data structures\n")
        f.write(
            "4. Impact of outliers, density, and separability on topological analysis\n"
        )

    print(f"Summary report saved: {report_path}")


if __name__ == "__main__":
    run_comprehensive_analysis()
