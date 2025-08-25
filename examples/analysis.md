# Comprehensive Metric-Structure Analysis for HOLE Library

This document describes the systematic analysis of distance metrics across various data structures using topological data analysis (TDA) and persistent homology.

## Overview

The comprehensive analysis tests **7 distance metrics** across **34 data structure variants** for a total of **238 experiments**. Each experiment generates 5 types of visualizations to understand how different metrics capture topological features under various data characteristics.

## Distance Metrics Tested

| Metric | Description | Properties |
|--------|-------------|------------|
| `euclidean` | Standard L2 distance | Classic geometric distance |
| `cosine` | Cosine similarity distance | Captures directional relationships |
| `mahalanobis` | Scaled L2 with covariance | Accounts for data correlations |
| `dn_euclidean` | Density-normalized Euclidean | Euclidean adjusted for local density |
| `dn_cosine` | Density-normalized Cosine | Cosine adjusted for local density |
| `dn_mahalanobis` | Density-normalized Mahalanobis | Mahalanobis adjusted for local density |
| `geodesic` | Graph-based geodesic distance | Captures manifold structure |

## Input Data Structures

### 1. Isotropic Clusters
**Purpose**: Test baseline clustering performance across metrics
**Generation**: `sklearn.datasets.make_blobs`
**Parameters**:
- Sample size: 500 points
- Dimensions: 10 features
- Clusters: 4 centers
- Cluster std: 0.8 (dense) / 1.5 (sparse) for separable; 2.5 (dense) / 4.0 (sparse) for non-separable
- Outliers: 10% random uniform outliers when included

### 2. Hypersphere Structure  
**Purpose**: Test manifold detection capabilities
**Generation**: Custom concentric sphere generation
**Parameters**:
- Sample size: 500 points (divided into 3 concentric spheres)
- Dimensions: 8 features
- Radii: Inner=1.0, Middle=3.0/2.0, Outer=5.0/3.5 (separable/non-separable)
- Noise: 0.1-0.8 depending on density/separability
- Outliers: 10% random uniform outliers when included

### 3. Elliptical Clusters
**Purpose**: Test anisotropic cluster handling
**Generation**: `make_blobs` + SVD transformation to elliptical shapes  
**Parameters**:
- Sample size: 500 points
- Dimensions: 6 features  
- Clusters: 3 centers
- Base cluster std: 0.5 (dense) / 1.2 (sparse)
- Elliptical distortion: SVD with singular values [5.0, 2.0, 1.0, 0.8, 0.5, 0.3] (separable) / [2.5, 1.5, 1.0, 0.9, 0.8, 0.7] (non-separable)
- Outliers: 10% random uniform outliers when included

### 4. Swiss Roll Manifold
**Purpose**: Test non-linear manifold analysis capabilities
**Generation**: `sklearn.datasets.make_swiss_roll`
**Parameters**:
- Sample size: 500 points
- Dimensions: 3 features (embedded 2D manifold)
- Noise: 0.1-1.0 depending on density/separability  
- Labels: Based on position along roll (3 regions for separable, 2 overlapping for non-separable)
- Outliers: 10% random uniform outliers when included

### 5. Tight Blobs
**Purpose**: Test clean separation scenarios
**Generation**: `make_blobs` with very tight clusters
**Parameters**:
- Sample size: 500 points
- Dimensions: 8 features
- Clusters: 5 centers
- Cluster std: 0.3 (very tight)
- Center box: (-15.0, 15.0) for maximum separation
- Outliers: 15% random uniform outliers when included (higher percentage to test robustness)

## Complete Experimental Design Table

| # | Structure Type | Variant | Density | Separability | Outliers | Sample Size | Dimensions | Clusters | Characteristics | Comments |
|---|---------------|---------|---------|--------------|----------|-------------|------------|----------|-----------------|----------|
| 1 | Isotropic Clusters | dense_separable_outliers | Dense | Separable | Yes | 500 | 10 | 4 | σ=0.8, well-separated + 10% outliers | |
| 2 | Isotropic Clusters | dense_separable_no_outliers | Dense | Separable | No | 500 | 10 | 4 | σ=0.8, well-separated | |
| 3 | Isotropic Clusters | dense_nonseparable_outliers | Dense | Non-separable | Yes | 500 | 10 | 4 | σ=2.5, overlapping + 10% outliers | |
| 4 | Isotropic Clusters | dense_nonseparable_no_outliers | Dense | Non-separable | No | 500 | 10 | 4 | σ=2.5, overlapping | |
| 5 | Isotropic Clusters | sparse_separable_outliers | Sparse | Separable | Yes | 500 | 10 | 4 | σ=1.5, well-separated + 10% outliers | |
| 6 | Isotropic Clusters | sparse_separable_no_outliers | Sparse | Separable | No | 500 | 10 | 4 | σ=1.5, well-separated | |
| 7 | Isotropic Clusters | sparse_nonseparable_outliers | Sparse | Non-separable | Yes | 500 | 10 | 4 | σ=4.0, heavily overlapping + 10% outliers | |
| 8 | Isotropic Clusters | sparse_nonseparable_no_outliers | Sparse | Non-separable | No | 500 | 10 | 4 | σ=4.0, heavily overlapping | |
| 9 | Hypersphere | dense_separable_outliers | Dense | Separable | Yes | 500 | 8 | 3 | Radii [1.0,3.0,5.0], noise=0.1 + outliers | |
| 10 | Hypersphere | dense_separable_no_outliers | Dense | Separable | No | 500 | 8 | 3 | Radii [1.0,3.0,5.0], noise=0.1 | |
| 11 | Hypersphere | dense_nonseparable_outliers | Dense | Non-separable | Yes | 500 | 8 | 3 | Radii [1.0,2.0,3.5], noise=0.4 + outliers | |
| 12 | Hypersphere | dense_nonseparable_no_outliers | Dense | Non-separable | No | 500 | 8 | 3 | Radii [1.0,2.0,3.5], noise=0.4 | |
| 13 | Hypersphere | sparse_separable_outliers | Sparse | Separable | Yes | 500 | 8 | 3 | Radii [1.0,3.0,5.0], noise=0.3 + outliers | |
| 14 | Hypersphere | sparse_separable_no_outliers | Sparse | Separable | No | 500 | 8 | 3 | Radii [1.0,3.0,5.0], noise=0.3 | |
| 15 | Hypersphere | sparse_nonseparable_outliers | Sparse | Non-separable | Yes | 500 | 8 | 3 | Radii [1.0,2.0,3.5], noise=0.8 + outliers | |
| 16 | Hypersphere | sparse_nonseparable_no_outliers | Sparse | Non-separable | No | 500 | 8 | 3 | Radii [1.0,2.0,3.5], noise=0.8 | |
| 17 | Elliptical Clusters | dense_separable_outliers | Dense | Separable | Yes | 500 | 6 | 3 | Strong elliptical distortion + outliers | |
| 18 | Elliptical Clusters | dense_separable_no_outliers | Dense | Separable | No | 500 | 6 | 3 | Strong elliptical distortion | |
| 19 | Elliptical Clusters | dense_nonseparable_outliers | Dense | Non-separable | Yes | 500 | 6 | 3 | Moderate elliptical distortion + outliers | |
| 20 | Elliptical Clusters | dense_nonseparable_no_outliers | Dense | Non-separable | No | 500 | 6 | 3 | Moderate elliptical distortion | |
| 21 | Elliptical Clusters | sparse_separable_outliers | Sparse | Separable | Yes | 500 | 6 | 3 | Strong elliptical + base noise + outliers | |
| 22 | Elliptical Clusters | sparse_separable_no_outliers | Sparse | Separable | No | 500 | 6 | 3 | Strong elliptical + base noise | |
| 23 | Elliptical Clusters | sparse_nonseparable_outliers | Sparse | Non-separable | Yes | 500 | 6 | 3 | Moderate elliptical + base noise + outliers | |
| 24 | Elliptical Clusters | sparse_nonseparable_no_outliers | Sparse | Non-separable | No | 500 | 6 | 3 | Moderate elliptical + base noise | |
| 25 | Swiss Roll | dense_separable_outliers | Dense | Separable | Yes | 500 | 3 | 3 | Low noise (0.1), clear regions + outliers | |
| 26 | Swiss Roll | dense_separable_no_outliers | Dense | Separable | No | 500 | 3 | 3 | Low noise (0.1), clear regions | |
| 27 | Swiss Roll | dense_nonseparable_outliers | Dense | Non-separable | Yes | 500 | 3 | 2 | Low noise (0.4), overlapping regions + outliers | |
| 28 | Swiss Roll | dense_nonseparable_no_outliers | Dense | Non-separable | No | 500 | 3 | 2 | Low noise (0.4), overlapping regions | |
| 29 | Swiss Roll | sparse_separable_outliers | Sparse | Separable | Yes | 500 | 3 | 3 | High noise (0.5), clear regions + outliers | |
| 30 | Swiss Roll | sparse_separable_no_outliers | Sparse | Separable | No | 500 | 3 | 3 | High noise (0.5), clear regions | |
| 31 | Swiss Roll | sparse_nonseparable_outliers | Sparse | Non-separable | Yes | 500 | 3 | 2 | Very high noise (1.0), overlapping + outliers | |
| 32 | Swiss Roll | sparse_nonseparable_no_outliers | Sparse | Non-separable | No | 500 | 3 | 2 | Very high noise (1.0), overlapping | |
| 33 | Tight Blobs | with_outliers | Dense | Separable | Yes | 500 | 8 | 5 | σ=0.3, max separation + 15% outliers | |
| 34 | Tight Blobs | no_outliers | Dense | Separable | No | 500 | 8 | 5 | σ=0.3, max separation | |

## Experimental Parameters

- **Random Seed**: 42 (for reproducibility)
- **Distance Matrix Computation**: All 7 metrics computed for each dataset
- **Persistence Computation**: Max dimension = 1, Max edge length = ∞
- **Cluster Flow Analysis**: Max 6 thresholds for evolution tracking
- **Visualization Export**: 300 DPI, tight bounding box

## Output Structure

Each experiment generates approximately 5-7 files:
```
comprehensive_analysis_results/
├── {structure_type}/
│   ├── {variant_name}/
│   │   ├── persistence_viz_{metric}.png    # 4-panel: diagram + barcode + PCA + MDS
│   │   ├── sankey_{metric}.png              # Cluster flow Sankey diagram  
│   │   ├── stacked_bars_{metric}.png        # Evolution stage composition
│   │   ├── heatmap_dendrogram_{metric}.png  # Distance matrix + clustering tree
│   │   └── [scatter hull files]             # Geometric separation analysis
│   └── ...
└── analysis_summary.txt                     # Final summary report
```

**Total Expected Files**: ~1,190 visualization files
**Total Experiments**: 238 (34 variants × 7 metrics)
**Estimated Runtime**: 2-4 hours 