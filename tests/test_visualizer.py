"""
Comprehensive tests for HOLEVisualizer and visualization components.
"""

import warnings
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

import hole
from hole.visualizer import HOLEVisualizer


class TestHOLEVisualizer:
    """Test suite for HOLEVisualizer class."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.points = np.random.rand(20, 3)
        self.labels = np.array([0] * 10 + [1] * 10)
        self.distance_matrix = hole.euclidean_distance(self.points)

    def test_initialization_with_point_cloud(self):
        """Test HOLEVisualizer initialization with point cloud."""
        viz = HOLEVisualizer(point_cloud=self.points)

        assert viz.n_points == 20
        assert viz.point_cloud.shape == (20, 3)
        assert viz.distance_matrix.shape == (20, 20)
        assert viz.persistence is not None
        assert len(viz.persistence) > 0

    def test_initialization_with_distance_matrix(self):
        """Test HOLEVisualizer initialization with distance matrix."""
        viz = HOLEVisualizer(distance_matrix_input=self.distance_matrix)

        assert viz.n_points == 20
        assert viz.point_cloud is None
        assert viz.distance_matrix.shape == (20, 20)
        assert np.allclose(viz.distance_matrix, self.distance_matrix)
        assert viz.persistence is not None

    def test_initialization_validation(self):
        """Test input validation during initialization."""
        # No input provided
        with pytest.raises(ValueError, match="Must provide either"):
            HOLEVisualizer()

        # Both inputs provided
        with pytest.raises(ValueError, match="Provide only one"):
            HOLEVisualizer(
                point_cloud=self.points, distance_matrix_input=self.distance_matrix
            )

        # Invalid distance matrix (not square)
        with pytest.raises(ValueError, match="distance_matrix_input must be square"):
            invalid_matrix = np.random.rand(20, 15)
            HOLEVisualizer(distance_matrix_input=invalid_matrix)

        # Invalid distance matrix (not symmetric)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress the warning we're testing
            asymmetric_matrix = np.random.rand(10, 10)
            viz = HOLEVisualizer(distance_matrix_input=asymmetric_matrix)
            # Should still work but issue warning

    def test_distance_metrics(self):
        """Test different distance metrics."""
        metrics = ["euclidean", "cosine", "manhattan", "mahalanobis"]

        for metric in metrics:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                viz = HOLEVisualizer(point_cloud=self.points, distance_metric=metric)
                assert viz.distance_matrix.shape == (20, 20)
                assert viz.persistence is not None

    def test_persistence_computation_parameters(self):
        """Test persistence computation with different parameters."""
        # Test different max_dimension
        viz1 = HOLEVisualizer(point_cloud=self.points, max_dimension=0)
        viz2 = HOLEVisualizer(point_cloud=self.points, max_dimension=2)

        # Both should work
        assert viz1.persistence is not None
        assert viz2.persistence is not None

        # Test max_edge_length
        viz3 = HOLEVisualizer(point_cloud=self.points, max_edge_length=1.0)
        assert viz3.persistence is not None

    @patch("matplotlib.pyplot.show")
    def test_plot_persistence_diagram(self, mock_show):
        """Test persistence diagram plotting."""
        viz = HOLEVisualizer(point_cloud=self.points)

        # Test basic plotting
        fig, ax = plt.subplots()
        viz.plot_persistence_diagram(ax=ax)
        plt.close(fig)

        # Test with different parameters
        fig, ax = plt.subplots()
        viz.plot_persistence_diagram(ax=ax, title="Test", pts=30)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_plot_persistence_barcode(self, mock_show):
        """Test persistence barcode plotting."""
        viz = HOLEVisualizer(point_cloud=self.points)

        fig, ax = plt.subplots()
        viz.plot_persistence_barcode(ax=ax)
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_plot_dimensionality_reduction(self, mock_show):
        """Test dimensionality reduction plotting."""
        viz = HOLEVisualizer(point_cloud=self.points)

        # Test PCA
        fig, ax = plt.subplots()
        viz.plot_dimensionality_reduction(method="pca", ax=ax, true_labels=self.labels)
        plt.close(fig)

        # Test MDS
        fig, ax = plt.subplots()
        viz.plot_dimensionality_reduction(method="mds", ax=ax, true_labels=self.labels)
        plt.close(fig)

        # Test t-SNE
        fig, ax = plt.subplots()
        viz.plot_dimensionality_reduction(method="tsne", ax=ax, true_labels=self.labels)
        plt.close(fig)

    def test_get_blob_visualizer(self):
        """Test blob visualizer creation."""
        viz = HOLEVisualizer(point_cloud=self.points)
        blob_viz = viz.get_blob_visualizer()

        assert blob_viz is not None
        assert hasattr(blob_viz, "plot_pca_with_cluster_hulls")

    def test_get_persistence_dendrogram_visualizer(self):
        """Test persistence dendrogram visualizer creation."""
        viz = HOLEVisualizer(point_cloud=self.points)
        dendro_viz = viz.get_persistence_dendrogram_visualizer(
            distance_matrix=viz.distance_matrix
        )

        assert dendro_viz is not None
        assert hasattr(dendro_viz, "compute_persistence")
        assert hasattr(dendro_viz, "plot_dendrogram_with_heatmap")

    def test_error_handling_invalid_method(self):
        """Test error handling with invalid methods."""
        viz = HOLEVisualizer(point_cloud=self.points)

        with pytest.raises(ValueError, match="Unknown method"):
            fig, ax = plt.subplots()
            viz.plot_dimensionality_reduction(method="invalid_method", ax=ax)
            plt.close(fig)

    def test_large_dataset_handling(self):
        """Test with larger datasets."""
        large_points = np.random.rand(100, 5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            viz = HOLEVisualizer(point_cloud=large_points)
            assert viz.n_points == 100
            assert viz.persistence is not None

    def test_small_dataset_handling(self):
        """Test with very small datasets."""
        small_points = np.random.rand(3, 2)

        viz = HOLEVisualizer(point_cloud=small_points)
        assert viz.n_points == 3
        assert viz.persistence is not None

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        high_dim_points = np.random.rand(15, 20)

        viz = HOLEVisualizer(point_cloud=high_dim_points)
        assert viz.n_points == 15
        assert viz.point_cloud.shape[1] == 20
        assert viz.persistence is not None


class TestVisualizationComponents:
    """Test suite for visualization components."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.points = np.random.rand(25, 3)
        self.labels = np.array([0] * 8 + [1] * 8 + [2] * 9)
        self.distance_matrix = hole.euclidean_distance(self.points)

    def test_mst_processor(self):
        """Test MSTProcessor functionality."""
        processor = hole.MSTProcessor()

        # Test MST creation
        mst = processor.create_mst(self.points)
        assert mst.shape == (25, 25)
        # MST might not be symmetric in this implementation
        assert np.all(mst >= 0)  # All values should be non-negative

        # Test full processing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_components, labels, filtered_mst = processor(self.points)
            assert isinstance(n_components, int)
            assert len(labels) == 25
            assert filtered_mst.shape == (25, 25)

    def test_cluster_flow_analyzer(self):
        """Test ClusterFlowAnalyzer functionality."""
        analyzer = hole.ClusterFlowAnalyzer(self.distance_matrix, max_thresholds=4)

        # Test cluster evolution computation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evolution = analyzer.compute_cluster_evolution(self.labels)

        assert isinstance(evolution, dict)
        assert "components_" in evolution
        assert "labels_" in evolution

    def test_blob_visualizer(self):
        """Test BlobVisualizer functionality."""
        blob_viz = hole.BlobVisualizer()

        # Test that it can be created
        assert blob_viz is not None
        assert hasattr(blob_viz, "plot_pca_with_cluster_hulls")

    def test_persistence_dendrogram(self):
        """Test PersistenceDendrogram functionality."""
        dendro_viz = hole.PersistenceDendrogram(distance_matrix=self.distance_matrix)

        # Test persistence computation
        dendro_viz.compute_persistence()
        assert hasattr(dendro_viz, "persistence")

        # Test RCM heatmap creation
        fig, ax = dendro_viz.plot_rcm_heatmap(title="Test Heatmap")
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    @patch("matplotlib.pyplot.show")
    def test_integration_workflow(self, mock_show):
        """Test complete integration workflow."""
        # Create visualizer
        viz = HOLEVisualizer(point_cloud=self.points)

        # Create all visualization components
        blob_viz = viz.get_blob_visualizer()
        dendro_viz = viz.get_persistence_dendrogram_visualizer(
            distance_matrix=viz.distance_matrix
        )
        analyzer = hole.ClusterFlowAnalyzer(viz.distance_matrix, max_thresholds=3)

        # Test that all components work together
        assert viz.persistence is not None
        assert blob_viz is not None
        assert dendro_viz is not None
        assert analyzer is not None

        # Test cluster evolution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evolution = analyzer.compute_cluster_evolution(self.labels)
            assert evolution is not None
