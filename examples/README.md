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
- Loads trained ViT model from `vit_cifar10_finetuned/`
- Runs inference on 1500 random CIFAR-10 test images
- Extracts embeddings from all ViT encoder layers separately
- Performs complete HOLE topological analysis per layer
- Cluster filtering and color matching in flow diagrams
- Output: `vit_random_hole_outputs/layer_0/` through `layer_11/`

### 5. `vit_inference_balanced.py`
**ViT model inference with HOLE** - Balanced subset analysis.
- Loads trained ViT model from `vit_cifar10_finetuned/`
- Runs inference on balanced CIFAR-10 subset (100 images per class = 1000 total)
- Extracts embeddings from all ViT encoder layers separately
- Performs complete HOLE topological analysis per layer
- Cluster filtering and color matching in flow diagrams
- Output: `vit_balanced_hole_outputs/layer_0/` through `layer_11/`

### 6. `vit_inference_noisy.py`
**ViT model inference with HOLE** - Noisy data robustness analysis.
- Loads trained ViT model from `vit_cifar10_finetuned/`
- Adds Gaussian noise (std=0.1) to input images
- Runs inference on balanced CIFAR-10 subset (100 images per class)
- Extracts embeddings from all ViT encoder layers
- Analyzes how noise affects topological structure per layer
- Output: `vit_noisy_hole_outputs/layer_0/` through `layer_11/`

### 7. `vit_inference_quantized.py`
**ViT model inference with HOLE** - INT8 quantized model analysis.
- Loads trained ViT model from `vit_cifar10_finetuned/`
- Applies dynamic INT8 quantization to model (4x smaller, faster inference)
- Runs inference on balanced CIFAR-10 subset (100 images per class)
- Extracts embeddings from all quantized encoder layers
- Analyzes how quantization affects topological structure per layer
- Output: `vit_quantized_hole_outputs/layer_0/` through `layer_11/`

## Running Examples

```bash
cd examples/

# Start with the basic example (recommended)
python hole_example.py

# Demo the new blob contour features
python blob_contour_demo.py

# For comprehensive analysis (takes much longer, generates extensive results)
python distance_metrics.py

# ViT model inference with HOLE analysis (requires vit_cifar10_finetuned/)
python vit_inference_random.py      # Random 1500 samples, all layers
python vit_inference_balanced.py    # Balanced 100 per class, all layers
python vit_inference_noisy.py       # Noisy data (Gaussian std=0.1), all layers
python vit_inference_quantized.py   # INT8 quantized model, all layers
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
- Requires `vit_cifar10_finetuned/` directory in repo root
