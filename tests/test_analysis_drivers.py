"""
End-to-end smoke tests for the high-level analysis drivers.

These exercise `analyze_activation_flows` and `analyze_activation_persistence`
across all five distance metrics — the code paths that previously called
non-existent MSTProcessor methods (fast_maha / cosine_gen / density_normalizer).
A regression on those phantom calls would fail this test.
"""

import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import hole

ALL_METRICS = [
    "Euclidean",
    "Mahalanobis",
    "Cosine",
    "Density_Normalized_Euclidean",
    "Density_Normalized_Mahalanobis",
]


@pytest.fixture
def synthetic_activations(tmp_path):
    """Two-layer synthetic activation dict saved as a .npy file."""
    rng = np.random.default_rng(0)
    n_per_class, n_classes, n_features = 16, 3, 32
    n_samples = n_per_class * n_classes
    centers = rng.normal(scale=4.0, size=(n_classes, n_features))

    labels = np.repeat(np.arange(n_classes), n_per_class)
    layer_a = np.vstack([centers[c] + rng.normal(scale=0.5, size=n_features)
                         for c in labels])
    layer_b = np.vstack([centers[c] + rng.normal(scale=1.5, size=n_features)
                         for c in labels])

    activations = {"layer_a": layer_a, "layer_b": layer_b}
    path = tmp_path / "activations.npy"
    np.save(path, activations, allow_pickle=True)
    return str(path), labels


def test_analyze_activation_flows_all_metrics(synthetic_activations, tmp_path):
    """All 5 distance metrics complete without phantom-method errors."""
    activation_file, labels = synthetic_activations
    output_dir = tmp_path / "flow_out"

    results = hole.analyze_activation_flows(
        activation_file=activation_file,
        output_dir=str(output_dir),
        model_name="test_model",
        condition_name="clean",
        true_labels=labels,
        max_points=48,
        max_thresholds=4,
        distance_metrics=ALL_METRICS,
    )

    assert isinstance(results, dict)
    assert "layer_a" in results and "layer_b" in results
    for layer, layer_results in results.items():
        for metric in ALL_METRICS:
            assert metric in layer_results, f"missing {metric} in {layer}"


def test_analyze_activation_persistence_all_metrics(synthetic_activations, tmp_path):
    """All 5 distance metrics complete in the dendrogram driver too."""
    activation_file, _ = synthetic_activations
    output_dir = tmp_path / "pers_out"

    hole.analyze_activation_persistence(
        activation_file=activation_file,
        output_dir=str(output_dir),
        model_name="test_model",
        condition_name="clean",
        max_points=48,
        distance_metrics=ALL_METRICS,
    )

    # At least one metric-specific output directory should exist per layer.
    produced = list(output_dir.glob("test_model_clean_*"))
    assert produced, "expected metric output dirs to be created"


def test_public_api_imports():
    """Every symbol in hole.__all__ must be importable from the top level."""
    for name in hole.__all__:
        assert hasattr(hole, name), f"hole.__all__ promises {name} but it's missing"
