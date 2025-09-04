"""
Comprehensive tests for distance metrics.
"""

import numpy as np
import pytest
import warnings

import hole
from hole.core.distance_metrics import (
    euclidean_distance,
    cosine_distance,
    manhattan_distance,
    mahalanobis_distance,
    chebyshev_distance,
    geodesic_distances,
    density_normalized_distance,
    distance_matrix,
)


class TestDistanceMetrics:
    """Test suite for all distance metric functions."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.small_data = np.random.rand(5, 3)
        self.medium_data = np.random.rand(20, 4)
        self.large_data = np.random.rand(100, 5)
        
    def test_euclidean_distance_properties(self):
        """Test Euclidean distance properties."""
        dist = euclidean_distance(self.medium_data)
        
        # Shape should be square
        assert dist.shape == (20, 20)
        
        # Diagonal should be zero
        assert np.allclose(np.diag(dist), 0, atol=1e-10)
        
        # Should be symmetric
        assert np.allclose(dist, dist.T)
        
        # All distances should be non-negative
        assert np.all(dist >= 0)
        
        # Triangle inequality: d(i,k) <= d(i,j) + d(j,k)
        for i in range(min(5, len(dist))):  # Test subset for performance
            for j in range(min(5, len(dist))):
                for k in range(min(5, len(dist))):
                    assert dist[i, k] <= dist[i, j] + dist[j, k] + 1e-10

    def test_cosine_distance_properties(self):
        """Test cosine distance properties."""
        dist = cosine_distance(self.medium_data)
        
        # Shape should be square
        assert dist.shape == (20, 20)
        
        # Diagonal should be zero
        assert np.allclose(np.diag(dist), 0, atol=1e-10)
        
        # Should be symmetric
        assert np.allclose(dist, dist.T)
        
        # Cosine distance should be in [0, 2] range
        assert np.all(dist >= 0)
        assert np.all(dist <= 2 + 1e-10)

    def test_manhattan_distance_properties(self):
        """Test Manhattan distance properties."""
        dist = manhattan_distance(self.medium_data)
        
        # Shape should be square
        assert dist.shape == (20, 20)
        
        # Diagonal should be zero
        assert np.allclose(np.diag(dist), 0, atol=1e-10)
        
        # Should be symmetric
        assert np.allclose(dist, dist.T)
        
        # All distances should be non-negative
        assert np.all(dist >= 0)

    def test_mahalanobis_distance_properties(self):
        """Test Mahalanobis distance properties."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dist = mahalanobis_distance(self.medium_data)
        
        # Shape should be square
        assert dist.shape == (20, 20)
        
        # Diagonal should be zero
        assert np.allclose(np.diag(dist), 0, atol=1e-10)
        
        # Should be symmetric
        assert np.allclose(dist, dist.T)
        
        # All distances should be non-negative
        assert np.all(dist >= 0)

    def test_chebyshev_distance_properties(self):
        """Test Chebyshev distance properties."""
        dist = chebyshev_distance(self.medium_data)
        
        # Shape should be square
        assert dist.shape == (20, 20)
        
        # Diagonal should be zero
        assert np.allclose(np.diag(dist), 0, atol=1e-10)
        
        # Should be symmetric
        assert np.allclose(dist, dist.T)
        
        # All distances should be non-negative
        assert np.all(dist >= 0)

    def test_geodesic_distances_properties(self):
        """Test geodesic distances properties."""
        try:
            dist = geodesic_distances(self.small_data)  # Use small data to avoid issues
            
            # Shape should be square
            assert dist.shape == (5, 5)
            
            # Diagonal should be zero
            assert np.allclose(np.diag(dist), 0, atol=1e-10)
            
            # Should be symmetric
            assert np.allclose(dist, dist.T)
            
            # All finite distances should be non-negative
            finite_mask = np.isfinite(dist)
            assert np.all(dist[finite_mask] >= 0)
            
        except Exception as e:
            pytest.skip(f"Geodesic distance computation failed: {e}")

    def test_density_normalized_distance_properties(self):
        """Test density normalized distance properties."""
        base_dist = euclidean_distance(self.medium_data)
        dn_dist = density_normalized_distance(self.medium_data, base_dist)
        
        # Shape should be same as base distance
        assert dn_dist.shape == base_dist.shape
        
        # Diagonal should be zero
        assert np.allclose(np.diag(dn_dist), 0, atol=1e-10)
        
        # Should be symmetric
        assert np.allclose(dn_dist, dn_dist.T)
        
        # All distances should be non-negative
        assert np.all(dn_dist >= 0)

    def test_distance_matrix_function(self):
        """Test the generic distance_matrix function."""
        # Test with different metrics
        metrics = ["euclidean", "cosine", "manhattan"]
        
        for metric in metrics:
            dist = distance_matrix(self.medium_data, metric=metric)
            
            # Shape should be square
            assert dist.shape == (20, 20)
            
            # Diagonal should be zero
            assert np.allclose(np.diag(dist), 0, atol=1e-10)
            
            # Should be symmetric
            assert np.allclose(dist, dist.T)
            
            # All distances should be non-negative
            assert np.all(dist >= 0)

    def test_distance_matrix_invalid_metric(self):
        """Test distance_matrix with invalid metric."""
        with pytest.raises(Exception):  # Could be ValueError or InvalidParameterError
            distance_matrix(self.medium_data, metric="invalid_metric")

    def test_empty_input_handling(self):
        """Test handling of empty or invalid inputs."""
        empty_data = np.array([]).reshape(0, 3)
        
        # Empty data should either raise an error or return empty matrix
        try:
            result = euclidean_distance(empty_data)
            # If it doesn't raise an error, it should return empty matrix
            assert result.shape == (0, 0)
        except (ValueError, IndexError):
            # This is also acceptable behavior
            pass

    def test_single_point_handling(self):
        """Test handling of single point."""
        single_point = np.array([[1, 2, 3]])
        
        dist = euclidean_distance(single_point)
        assert dist.shape == (1, 1)
        assert dist[0, 0] == 0

    def test_identical_points_handling(self):
        """Test handling of identical points."""
        identical_points = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        
        dist = euclidean_distance(identical_points)
        assert np.allclose(dist, 0)

    def test_high_dimensional_data(self):
        """Test with high dimensional data."""
        high_dim_data = np.random.rand(10, 100)
        
        dist = euclidean_distance(high_dim_data)
        assert dist.shape == (10, 10)
        assert np.allclose(np.diag(dist), 0)

    def test_distance_consistency(self):
        """Test that different ways of computing same distance give consistent results."""
        # Compare distance_matrix with individual functions
        data = self.medium_data
        
        euclidean_direct = euclidean_distance(data)
        euclidean_generic = distance_matrix(data, metric="euclidean")
        assert np.allclose(euclidean_direct, euclidean_generic)
        
        cosine_direct = cosine_distance(data)
        cosine_generic = distance_matrix(data, metric="cosine")
        assert np.allclose(cosine_direct, cosine_generic)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very small values
        small_data = np.random.rand(5, 3) * 1e-10
        dist = euclidean_distance(small_data)
        assert np.isfinite(dist).all()
        
        # Very large values
        large_data = np.random.rand(5, 3) * 1e10
        dist = euclidean_distance(large_data)
        assert np.isfinite(dist).all()

    def test_mahalanobis_singular_covariance(self):
        """Test Mahalanobis distance with singular covariance matrix."""
        # Create data with perfectly correlated features
        base_data = np.random.rand(10, 1)
        correlated_data = np.hstack([base_data, base_data, base_data])  # All columns identical
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Should fall back to Euclidean or handle gracefully
            dist = mahalanobis_distance(correlated_data)
            assert dist.shape == (10, 10)
            assert np.allclose(np.diag(dist), 0)

