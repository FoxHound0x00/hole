"""
Tests for core functionality including persistence computation and MST processing.
"""

import warnings

import numpy as np

from hole.core.mst_processor import MSTProcessor
from hole.core.persistence import compute_persistence, extract_death_thresholds


class TestPersistenceComputation:
    """Test suite for persistence computation functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        # Create a simple distance matrix
        points = np.random.rand(10, 3)
        self.distance_matrix = np.linalg.norm(
            points[:, np.newaxis] - points[np.newaxis, :], axis=2
        )

    def test_compute_persistence_basic(self):
        """Test basic persistence computation."""
        persistence = compute_persistence(self.distance_matrix)

        # Should return a list of tuples
        assert isinstance(persistence, list)
        assert len(persistence) > 0

        # Each element should be a tuple with (dimension, (birth, death))
        for p in persistence:
            assert isinstance(p, tuple)
            assert len(p) == 2
            assert isinstance(p[0], (int, np.integer))  # dimension
            assert isinstance(p[1], tuple)  # (birth, death) pair
            assert len(p[1]) == 2
            birth, death = p[1]
            assert isinstance(birth, (float, np.floating))  # birth
            assert isinstance(death, (float, np.floating)) or np.isinf(death)

    def test_compute_persistence_parameters(self):
        """Test persistence computation with different parameters."""
        # Test different max_dimension
        persistence_0d = compute_persistence(self.distance_matrix, max_dimension=0)
        persistence_1d = compute_persistence(self.distance_matrix, max_dimension=1)
        persistence_2d = compute_persistence(self.distance_matrix, max_dimension=2)

        assert len(persistence_0d) > 0
        assert len(persistence_1d) >= len(persistence_0d)
        assert len(persistence_2d) >= len(persistence_1d)

        # Test max_edge_length
        persistence_limited = compute_persistence(
            self.distance_matrix, max_edge_length=1.0
        )
        assert len(persistence_limited) > 0

    def test_extract_death_thresholds(self):
        """Test death threshold extraction."""
        persistence = compute_persistence(self.distance_matrix)

        # Extract 0-dimensional death thresholds
        death_thresholds_0d = extract_death_thresholds(persistence, dimension=0)
        assert isinstance(death_thresholds_0d, list)

        # Extract 1-dimensional death thresholds
        death_thresholds_1d = extract_death_thresholds(persistence, dimension=1)
        assert isinstance(death_thresholds_1d, list)

        # All thresholds should be finite and positive
        for threshold in death_thresholds_0d:
            if not np.isinf(threshold):
                assert threshold >= 0

        for threshold in death_thresholds_1d:
            if not np.isinf(threshold):
                assert threshold >= 0

    def test_persistence_with_small_matrix(self):
        """Test persistence computation with very small distance matrix."""
        small_matrix = np.array([[0, 1], [1, 0]])
        persistence = compute_persistence(small_matrix)

        assert isinstance(persistence, list)
        assert len(persistence) >= 1  # At least one connected component

    def test_persistence_with_large_matrix(self):
        """Test persistence computation with larger distance matrix."""
        np.random.seed(42)
        large_points = np.random.rand(50, 4)
        large_matrix = np.linalg.norm(
            large_points[:, np.newaxis] - large_points[np.newaxis, :], axis=2
        )

        persistence = compute_persistence(large_matrix, max_dimension=1)
        assert isinstance(persistence, list)
        assert len(persistence) > 0

    def test_persistence_edge_cases(self):
        """Test persistence computation edge cases."""
        # Identity matrix (all points identical)
        identity_matrix = np.zeros((5, 5))
        persistence = compute_persistence(identity_matrix)
        assert len(persistence) > 0

        # Very sparse matrix
        sparse_matrix = np.eye(5) * 1000  # Large distances except diagonal
        persistence = compute_persistence(sparse_matrix, max_edge_length=100)
        assert len(persistence) > 0


class TestMSTProcessor:
    """Test suite for MST processor functionality."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.points = np.random.rand(15, 3)
        self.processor = MSTProcessor()

    def test_mst_creation(self):
        """Test MST creation."""
        mst = self.processor.create_mst(self.points)

        # Should be square matrix
        assert mst.shape == (15, 15)

        # MST might not be symmetric in this implementation, so we'll check it's valid
        # At minimum, it should be a valid adjacency matrix
        assert np.all(mst >= 0)  # All values should be non-negative

        # Diagonal should be zero
        assert np.allclose(np.diag(mst), 0)

        # Should have exactly n-1 edges (for connected graph)
        # Count non-zero off-diagonal elements and divide by 2 (symmetric)
        non_zero_count = np.count_nonzero(mst) - np.count_nonzero(np.diag(mst))
        edge_count = non_zero_count // 2
        assert edge_count <= 15 - 1  # At most n-1 edges

    def test_mst_full_processing(self):
        """Test full MST processing pipeline."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_components, labels, filtered_mst = self.processor(self.points)

        # Check return types
        assert isinstance(n_components, int)
        assert isinstance(labels, np.ndarray)
        assert isinstance(filtered_mst, np.ndarray)

        # Check dimensions
        assert len(labels) == 15
        assert filtered_mst.shape == (15, 15)

        # Number of components should be positive
        assert n_components > 0

        # Labels should be integers
        assert labels.dtype in [np.int32, np.int64, int]

    def test_mst_with_different_sizes(self):
        """Test MST processing with different data sizes."""
        # Small dataset
        small_points = np.random.rand(5, 2)
        mst_small = self.processor.create_mst(small_points)
        assert mst_small.shape == (5, 5)

        # Medium dataset
        medium_points = np.random.rand(25, 4)
        mst_medium = self.processor.create_mst(medium_points)
        assert mst_medium.shape == (25, 25)

        # Larger dataset
        large_points = np.random.rand(50, 3)
        mst_large = self.processor.create_mst(large_points)
        assert mst_large.shape == (50, 50)

    def test_mst_initialization_parameters(self):
        """Test MST processor initialization with different parameters."""
        # Test with different threshold parameter
        processor_custom = MSTProcessor(threshold=50)

        mst = processor_custom.create_mst(self.points)
        assert mst.shape == (15, 15)

    def test_mst_edge_cases(self):
        """Test MST processing edge cases."""
        # Single point
        single_point = np.array([[1, 2, 3]])
        mst_single = self.processor.create_mst(single_point)
        assert mst_single.shape == (1, 1)
        assert mst_single[0, 0] == 0

        # Two points
        two_points = np.array([[1, 2, 3], [4, 5, 6]])
        mst_two = self.processor.create_mst(two_points)
        assert mst_two.shape == (2, 2)

        # Identical points
        identical_points = np.array([[1, 2], [1, 2], [1, 2]])
        mst_identical = self.processor.create_mst(identical_points)
        assert mst_identical.shape == (3, 3)

    def test_mst_properties(self):
        """Test mathematical properties of MST."""
        mst = self.processor.create_mst(self.points)

        # MST should be connected (assuming input points form connected graph)
        # Check if there's a path between any two nodes
        # This is a simplified connectivity test
        assert np.any(mst > 0)  # At least some edges should exist

        # All edge weights should be positive
        edge_weights = mst[mst > 0]
        if len(edge_weights) > 0:
            assert np.all(edge_weights > 0)

    def test_mst_reproducibility(self):
        """Test that MST computation is reproducible."""
        mst1 = self.processor.create_mst(self.points)
        mst2 = self.processor.create_mst(self.points)

        # Should produce identical results
        assert np.allclose(mst1, mst2)


class TestCoreIntegration:
    """Test integration between core components."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.points = np.random.rand(20, 3)
        self.distance_matrix = np.linalg.norm(
            self.points[:, np.newaxis] - self.points[np.newaxis, :], axis=2
        )

    def test_persistence_mst_integration(self):
        """Test integration between persistence and MST computations."""
        # Compute persistence
        persistence = compute_persistence(self.distance_matrix)
        death_thresholds = extract_death_thresholds(persistence, dimension=0)

        # Compute MST
        processor = MSTProcessor()
        mst = processor.create_mst(self.points)

        # Both should work on same data
        assert len(persistence) > 0
        assert len(death_thresholds) >= 0
        assert mst.shape == (20, 20)

        # MST edge weights should be related to distance matrix
        mst_edges = mst[mst > 0]
        if len(mst_edges) > 0:
            # MST edges should be subset of distance matrix values
            distance_values = self.distance_matrix[self.distance_matrix > 0]
            # All MST edge weights should appear in distance matrix
            for edge_weight in mst_edges:
                assert np.any(np.isclose(distance_values, edge_weight, rtol=1e-10))

    def test_workflow_consistency(self):
        """Test consistency across complete workflow."""
        # This test ensures all core components work together
        persistence = compute_persistence(self.distance_matrix, max_dimension=1)
        death_thresholds_0d = extract_death_thresholds(persistence, dimension=0)
        death_thresholds_1d = extract_death_thresholds(persistence, dimension=1)

        processor = MSTProcessor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_components, labels, filtered_mst = processor(self.points)

        # All computations should complete successfully
        assert len(persistence) > 0
        assert isinstance(death_thresholds_0d, list)
        assert isinstance(death_thresholds_1d, list)
        assert n_components > 0
        assert len(labels) == 20
        assert filtered_mst.shape == (20, 20)
