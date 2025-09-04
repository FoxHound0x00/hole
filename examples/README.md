# HOLE Examples

This directory contains examples demonstrating the HOLE library functionality.

## Available Examples

### 1. `hole_example.py`
**Recommended starting point** - Shows HOLE's core visualization capabilities.
- Sankey diagrams for cluster evolution
- Stacked bar charts for threshold analysis  
- Heatmap dendrograms for distance matrices
- Blob visualizations with convex hulls
- Persistence diagrams and barcodes
- MDS plots for all distance metrics
- Demonstrates what makes HOLE unique

### 2. `distance_metrics.py` 
**Advanced comprehensive analysis** - Systematic comparison across data structures and metrics.
- ~800 lines of code
- 5 different data structures (isotropic clusters, hypersphere, elliptical, Swiss roll, tight blobs)
- 7 distance metrics (euclidean, cosine, mahalanobis, density-normalized variants, geodesic)
- Generates complete visualization suites for each combination
- Produces extensive analysis results

## Running Examples

```bash
cd examples/

# Start with the basic example (recommended)
python hole_example.py

# For comprehensive analysis (takes much longer, generates extensive results)
python distance_metrics.py
```

## Output

Examples will generate PNG files showing the visualizations. These are automatically saved to the current directory.

## Requirements

All examples require the HOLE library to be installed with its dependencies:
- numpy
- matplotlib  
- scikit-learn
- gudhi
- scipy
- seaborn
