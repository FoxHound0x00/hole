"""
Basic tests for HOLE library.
"""

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
    assert hasattr(hole, "FlowVisualizer")
    assert hasattr(hole, "BlobVisualizer")
    assert hasattr(hole, "HeatmapDendrogram")


def test_distance_metrics():
    """Test distance metric functions."""
    # Create sample data
    data = np.random.rand(10, 3)

    # Test euclidean distance
    dist = hole.euclidean_distance(data)
    assert dist.shape == (10, 10)
    assert np.allclose(np.diag(dist), 0)  # diagonal should be zero

    # Test cosine distance
    dist = hole.cosine_distance(data)
    assert dist.shape == (10, 10)
    assert np.allclose(np.diag(dist), 0)  # diagonal should be zero


def test_mst_processor():
    """Test MST processor basic functionality."""
    # Create sample data
    data = np.random.rand(20, 3)

    # Initialize processor
    processor = hole.MSTProcessor()

    # Test MST creation
    mst = processor.create_mst(data)
    assert mst.shape == (20, 20)

    # Test complete pipeline
    n_components, labels, filtered_mst = processor(data)
    assert isinstance(n_components, int)
    assert len(labels) == 20
    assert filtered_mst.shape == (20, 20)


def test_cluster_flow_analyzer():
    """Test cluster flow analyzer basic functionality."""
    # Create sample distance matrix
    np.random.seed(42)
    data = np.random.rand(15, 2)

    # Compute distance matrix using scipy
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(data))

    # Initialize analyzer
    analyzer = hole.ClusterFlowAnalyzer(dist_matrix, max_thresholds=3)

    # Test cluster evolution computation
    labels = np.array([0] * 7 + [1] * 8)  # true labels
    evolution = analyzer.compute_cluster_evolution(labels)

    assert "components_" in evolution
    assert "labels_" in evolution
    assert "distance_metric" in evolution["components_"]


if __name__ == "__main__":
    pytest.main([__file__])
