# HOLE Examples

This directory contains examples demonstrating the HOLE library functionality.

## Available Examples

### 1. `hole_example.py`
**Recommended starting point** - Shows HOLE's core visualization capabilities.
- Sankey diagrams for cluster evolution
- Stacked bar charts for threshold analysis  
- Heatmap dendrograms for distance matrices
- **Blob visualizations with class-colored contours and outlier detection**
- Persistence diagrams and barcodes
- MDS plots for all distance metrics
- Demonstrates what makes HOLE unique

### 2. `blob_contour_demo.py`
**Blob contour visualization demo** - Demonstrates the new contour and outlier class features.
- Shows how persistent homology clusters contain mixed classes
- Demonstrates percentage-based outlier class detection
- Class-colored contours for majority classes (≥threshold%)
- Scatter plots for outlier classes (<threshold%)
- Multiple threshold examples (5%, 10%, 15%)

### 3. `distance_metrics.py` 
**Advanced comprehensive analysis** - Systematic comparison across data structures and metrics.
- ~800 lines of code
- 5 different data structures (isotropic clusters, hypersphere, elliptical, Swiss roll, tight blobs)
- 7 distance metrics (euclidean, cosine, mahalanobis, density-normalized variants, geodesic)
- Generates complete visualization suites for each combination
- Produces extensive analysis results

### 4. `vit_inference_random.py`
**ViT model inference with HOLE** - Random subset analysis.
- Loads trained ViT model from `vit.zip`
- Runs inference on 500 random CIFAR-10 test images
- Extracts embeddings from ViT pooler layer
- Performs complete HOLE topological analysis
- Generates PCA, blob, dendrogram, and flow visualizations
- Output: `vit_random_hole_outputs/`

### 5. `vit_inference_balanced.py`
**ViT model inference with HOLE** - Balanced subset analysis.
- Loads trained ViT model from `vit.zip`
- Runs inference on balanced CIFAR-10 subset (15 images per class = 150 total)
- Extracts embeddings from ViT pooler layer
- Performs complete HOLE topological analysis
- Generates PCA, blob, dendrogram, and flow visualizations
- Output: `vit_balanced_hole_outputs/`

## Running Examples

```bash
cd examples/

# Start with the basic example (recommended)
python hole_example.py

# Demo the new blob contour features
python blob_contour_demo.py

# For comprehensive analysis (takes much longer, generates extensive results)
python distance_metrics.py

# ViT model inference with HOLE analysis (requires vit.zip in repo root)
python vit_inference_random.py      # Random 500 samples
python vit_inference_balanced.py    # Balanced 15 per class
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

**For ViT inference examples** (`vit_inference_*.py`):
- torch
- transformers
- datasets
- torchvision
- Requires `vit.zip` in repo root directory
