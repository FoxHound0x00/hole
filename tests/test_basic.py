"""
Basic tests for HOLE library.
"""

import warnings

import numpy as np
import pytest

import hole


def test_package_imports():
    """Test that the package imports successfully."""
    assert hasattr(hole, "__version__")
    assert hole.__version__ == "0.1.0"


def test_core_components():
    """Test that core components are available."""
    assert hasattr(hole, "MSTProcessor")
    assert hasattr(hole, "ClusterFlowAnalyzer")
    assert hasattr(hole, "BlobVisualizer")
    assert hasattr(hole, "PersistenceDendrogram")
    assert hasattr(hole, "HOLEVisualizer")


def test_distance_metrics():
    """Test distance metric functions."""
    # Create sample data
    np.random.seed(42)
    data = np.random.rand(10, 3)

    # Test euclidean distance
    dist = hole.euclidean_distance(data)
    assert dist.shape == (10, 10)
    assert np.allclose(np.diag(dist), 0)  # diagonal should be zero
    assert np.allclose(dist, dist.T)  # should be symmetric

    # Test cosine distance
    dist = hole.cosine_distance(data)
    assert dist.shape == (10, 10)
    assert np.allclose(np.diag(dist), 0)  # diagonal should be zero
    assert np.allclose(dist, dist.T)  # should be symmetric

    # Test manhattan distance
    dist = hole.manhattan_distance(data)
    assert dist.shape == (10, 10)
    assert np.allclose(np.diag(dist), 0)  # diagonal should be zero

    # Test mahalanobis distance (may fall back to euclidean for small datasets)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist = hole.mahalanobis_distance(data)
        assert dist.shape == (10, 10)
        assert np.allclose(np.diag(dist), 0)


def test_hole_visualizer_basic():
    """Test HOLEVisualizer basic functionality."""
    # Create sample data
    np.random.seed(42)
    data = np.random.rand(20, 3)

    # Test with point cloud
    viz = hole.HOLEVisualizer(point_cloud=data)
    assert viz.n_points == 20
    assert viz.distance_matrix.shape == (20, 20)
    assert viz.persistence is not None

    # Test with distance matrix
    dist_matrix = hole.euclidean_distance(data)
    viz2 = hole.HOLEVisualizer(distance_matrix_input=dist_matrix)
    assert viz2.n_points == 20
    assert np.allclose(viz2.distance_matrix, dist_matrix)


def test_mst_processor():
    """Test MST processor basic functionality."""
    # Create sample data
    np.random.seed(42)
    data = np.random.rand(15, 3)

    # Initialize processor
    processor = hole.MSTProcessor()

    # Test MST creation
    mst = processor.create_mst(data)
    assert mst.shape == (15, 15)

    # MST should be sparse (mostly zeros)
    assert np.sum(mst > 0) < mst.size  # Should have fewer edges than complete graph

    # Test complete pipeline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_components, labels, filtered_mst = processor(data)
        assert isinstance(n_components, int)
        assert len(labels) == 15
        assert filtered_mst.shape == (15, 15)


def test_cluster_flow_analyzer():
    """Test cluster flow analyzer basic functionality."""
    # Create sample distance matrix
    np.random.seed(42)
    data = np.random.rand(12, 2)

    # Compute distance matrix
    dist_matrix = hole.euclidean_distance(data)

    # Initialize analyzer
    analyzer = hole.ClusterFlowAnalyzer(dist_matrix, max_thresholds=3)

    # Test cluster evolution computation
    labels = np.array([0] * 6 + [1] * 6)  # true labels

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evolution = analyzer.compute_cluster_evolution(labels)

    assert "components_" in evolution
    assert "labels_" in evolution
    assert isinstance(evolution["components_"], dict)
    assert isinstance(evolution["labels_"], dict)


def test_input_validation():
    """Test input validation and error handling."""
    # Test HOLEVisualizer with invalid inputs
    with pytest.raises(ValueError, match="Must provide either"):
        hole.HOLEVisualizer()

    with pytest.raises(ValueError, match="Provide only one"):
        data = np.random.rand(10, 3)
        dist_matrix = hole.euclidean_distance(data)
        hole.HOLEVisualizer(point_cloud=data, distance_matrix_input=dist_matrix)


if __name__ == "__main__":
    pytest.main([__file__])
