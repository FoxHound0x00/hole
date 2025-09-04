"""
Integration tests to ensure all components work together correctly.
"""

import os
import tempfile
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

import hole
from hole.visualization.cluster_flow import ClusterFlowAnalyzer, FlowVisualizer


class TestFullWorkflowIntegration:
    """Test complete workflow integration."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        # Create well-separated clusters for reliable testing
        from sklearn.datasets import make_blobs

        self.points, self.labels = make_blobs(
            n_samples=50, centers=3, cluster_std=1.0, random_state=42
        )
        self.distance_matrix = hole.euclidean_distance(self.points)

    def test_complete_visualization_pipeline(self):
        """Test the complete visualization pipeline."""
        # Create HOLEVisualizer
        viz = hole.HOLEVisualizer(point_cloud=self.points)

        # Test all major visualizations
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Persistence visualizations
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            viz.plot_persistence_diagram(ax=axes[0, 0])
            viz.plot_persistence_barcode(ax=axes[0, 1])
            viz.plot_dimensionality_reduction(
                method="pca", ax=axes[1, 0], true_labels=self.labels
            )
            viz.plot_dimensionality_reduction(
                method="mds", ax=axes[1, 1], true_labels=self.labels
            )

            plt.tight_layout()
            plt.savefig(os.path.join(temp_dir, "persistence_test.png"))
            plt.close()

            # 2. Blob visualization
            blob_viz = viz.get_blob_visualizer()
            fig = blob_viz.plot_pca_with_cluster_hulls(
                self.points,
                self.labels,
                threshold=1.0,
                save_path=os.path.join(temp_dir, "blob_test.png"),
            )
            plt.close(fig)

            # 3. Heatmap dendrogram
            dendro_viz = viz.get_persistence_dendrogram_visualizer(
                distance_matrix=viz.distance_matrix
            )
            dendro_viz.compute_persistence()
            fig, ax = dendro_viz.plot_dendrogram_with_heatmap()
            plt.savefig(os.path.join(temp_dir, "heatmap_test.png"))
            plt.close()

            # 4. Cluster flow analysis
            analyzer = ClusterFlowAnalyzer(viz.distance_matrix, max_thresholds=3)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                evolution = analyzer.compute_cluster_evolution(self.labels)

            if evolution and "components_" in evolution and "labels_" in evolution:
                flow_viz = FlowVisualizer()
                fig = flow_viz.plot_sankey_flow(
                    evolution, save_path=os.path.join(temp_dir, "sankey_test.png")
                )
                plt.close(fig)

            # Verify files were created
            expected_files = [
                "persistence_test.png",
                "blob_test.png",
                "heatmap_test.png",
            ]
            for filename in expected_files:
                filepath = os.path.join(temp_dir, filename)
                assert os.path.exists(filepath), f"File {filename} was not created"
                assert os.path.getsize(filepath) > 0, f"File {filename} is empty"

    def test_multiple_distance_metrics_integration(self):
        """Test integration with multiple distance metrics."""
        metrics = ["euclidean", "cosine", "manhattan"]

        for metric in metrics:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                viz = hole.HOLEVisualizer(
                    point_cloud=self.points, distance_metric=metric
                )

                # Test that all components work with different metrics
                assert viz.persistence is not None
                assert viz.distance_matrix.shape == (50, 50)

                # Test visualization creation
                fig, ax = plt.subplots()
                viz.plot_persistence_diagram(ax=ax)
                plt.close(fig)

                # Test cluster flow analysis
                analyzer = ClusterFlowAnalyzer(viz.distance_matrix, max_thresholds=2)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    evolution = analyzer.compute_cluster_evolution(self.labels)
                    assert isinstance(evolution, dict)

    def test_error_recovery_and_robustness(self):
        """Test error recovery and robustness."""
        # Test with challenging data

        # 1. Very small dataset
        small_points = np.random.rand(3, 2)
        viz_small = hole.HOLEVisualizer(point_cloud=small_points)
        assert viz_small.persistence is not None

        # 2. High-dimensional data
        high_dim_points = np.random.rand(20, 50)
        viz_high_dim = hole.HOLEVisualizer(point_cloud=high_dim_points)
        assert viz_high_dim.persistence is not None

        # 3. Data with outliers
        outlier_points = np.vstack(
            [
                np.random.normal(0, 1, (15, 3)),  # Normal cluster
                np.random.normal(10, 0.1, (5, 3)),  # Outlier cluster
            ]
        )
        viz_outliers = hole.HOLEVisualizer(point_cloud=outlier_points)
        assert viz_outliers.persistence is not None

    def test_memory_efficiency(self):
        """Test memory efficiency with larger datasets."""
        # Test with moderately large dataset
        large_points = np.random.rand(200, 4)

        # This should complete without memory errors
        viz = hole.HOLEVisualizer(point_cloud=large_points)
        assert viz.persistence is not None
        assert viz.distance_matrix.shape == (200, 200)

        # Test that visualizations can be created
        fig, ax = plt.subplots()
        viz.plot_persistence_diagram(ax=ax)
        plt.close(fig)

    def test_reproducibility(self):
        """Test that results are reproducible."""
        # Create two identical visualizers
        viz1 = hole.HOLEVisualizer(point_cloud=self.points, distance_metric="euclidean")
        viz2 = hole.HOLEVisualizer(point_cloud=self.points, distance_metric="euclidean")

        # Distance matrices should be identical
        assert np.allclose(viz1.distance_matrix, viz2.distance_matrix)

        # Persistence should be identical (order might differ)
        assert len(viz1.persistence) == len(viz2.persistence)

    def test_api_consistency(self):
        """Test API consistency across different usage patterns."""
        # Test with point cloud
        viz1 = hole.HOLEVisualizer(point_cloud=self.points)

        # Test with distance matrix
        viz2 = hole.HOLEVisualizer(distance_matrix_input=self.distance_matrix)

        # Both should have same interface
        assert hasattr(viz1, "plot_persistence_diagram")
        assert hasattr(viz2, "plot_persistence_diagram")
        assert hasattr(viz1, "get_blob_visualizer")
        assert hasattr(viz2, "get_blob_visualizer")

        # Both should produce valid visualizations
        fig1, ax1 = plt.subplots()
        viz1.plot_persistence_diagram(ax=ax1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        viz2.plot_persistence_diagram(ax=ax2)
        plt.close(fig2)


class TestExampleCompatibility:
    """Test compatibility with documented examples."""

    def test_readme_example_compatibility(self):
        """Test that README examples work correctly."""
        from sklearn.datasets import make_blobs

        # This is the example from the README
        points, labels = make_blobs(n_samples=50, centers=3, random_state=42)

        # Create HOLE visualizer
        viz = hole.HOLEVisualizer(point_cloud=points)

        # Generate visualizations (from README)
        fig1, ax1 = plt.subplots()
        viz.plot_persistence_diagram(ax=ax1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        viz.plot_dimensionality_reduction(method="pca", ax=ax2, true_labels=labels)
        plt.close(fig2)

        # Distance metrics example from README
        euclidean_dist = hole.euclidean_distance(points)
        cosine_dist = hole.cosine_distance(points)
        manhattan_dist = hole.manhattan_distance(points)

        # Use with visualizer
        viz2 = hole.HOLEVisualizer(distance_matrix_input=euclidean_dist)
        assert viz2.n_points == 50

    def test_library_imports_work(self):
        """Test that all documented imports work correctly."""
        # Test main library import
        import hole

        # Test that all documented classes are available
        assert hasattr(hole, "HOLEVisualizer")
        assert hasattr(hole, "MSTProcessor")
        assert hasattr(hole, "ClusterFlowAnalyzer")
        assert hasattr(hole, "BlobVisualizer")
        assert hasattr(hole, "PersistenceDendrogram")

        # Test distance functions
        assert hasattr(hole, "euclidean_distance")
        assert hasattr(hole, "cosine_distance")
        assert hasattr(hole, "manhattan_distance")
        assert hasattr(hole, "mahalanobis_distance")
        assert hasattr(hole, "geodesic_distances")

        # Test version info
        assert hasattr(hole, "__version__")
        assert hole.__version__ == "0.1.0"

    def test_workflow_matches_documentation(self):
        """Test that actual workflow matches documented workflow."""
        # Generate sample data
        points = np.random.rand(30, 3)
        labels = np.array([0] * 15 + [1] * 15)

        # Follow the documented workflow
        # 1. Create visualizer
        viz = hole.HOLEVisualizer(point_cloud=points)

        # 2. Create components
        blob_viz = viz.get_blob_visualizer()
        dendro_viz = viz.get_persistence_dendrogram_visualizer(
            distance_matrix=viz.distance_matrix
        )
        analyzer = ClusterFlowAnalyzer(viz.distance_matrix, max_thresholds=3)

        # 3. Generate visualizations
        with tempfile.TemporaryDirectory() as temp_dir:
            # Persistence diagram
            fig, ax = plt.subplots()
            viz.plot_persistence_diagram(ax=ax)
            plt.savefig(os.path.join(temp_dir, "persistence.png"))
            plt.close()

            # Blob visualization
            fig = blob_viz.plot_pca_with_cluster_hulls(
                points,
                labels,
                threshold=1.0,
                save_path=os.path.join(temp_dir, "blob.png"),
            )
            plt.close(fig)

            # Heatmap dendrogram
            dendro_viz.compute_persistence()
            fig, ax = dendro_viz.plot_dendrogram_with_heatmap()
            plt.savefig(os.path.join(temp_dir, "heatmap.png"))
            plt.close()

            # Verify all files were created
            for filename in ["persistence.png", "blob.png", "heatmap.png"]:
                filepath = os.path.join(temp_dir, filename)
                assert os.path.exists(filepath)
                assert os.path.getsize(filepath) > 0
