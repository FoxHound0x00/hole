# HOLE Model Training and Inference Pipeline

A unified, configurable pipeline for training and analyzing deep learning models using Hydra configuration management and HOLE (Homological Observation of Latent Embeddings) analysis.

## Features

- **Hydra Configuration**: Fully configurable training and inference pipelines
- **Multiple Models**: Support for ResNet, MobileNet, ViT, ConvNeXt, and MLP-Mixer architectures
- **HOLE Analysis**: Topological data analysis of learned representations
- **t-SNE with PCA**: Advanced dimensionality reduction visualization
- **Multi-run Support**: Train/analyze multiple models in parallel

## Installation

```bash
# Install dependencies
pip install hydra-core omegaconf timm

# Or using poetry
poetry install --extras hooks
```

## Quick Start

### Training

```bash
# Train with default config (ResNet50)
python train.py

# Train specific model
python train.py model=convnext_base

# Override parameters
python train.py model=mixer_b16 training.epochs=20 training.batch_size=32

# Train multiple models
python train.py -m model=resnet18,resnet34,resnet50
```

### Inference

```bash
# Run inference with default config
python infer.py

# Run on specific model
python infer.py model=convnext_base

# Override parameters
python infer.py model=mixer_b16 inference.samples_per_class=100

# Run on multiple models
python infer.py -m model=resnet18,resnet34,resnet50
```

## Configuration Structure

```
configs/
├── config.yaml              # Main config with defaults
├── model/                   # Model-specific configs
│   ├── resnet18.yaml
│   ├── resnet34.yaml
│   ├── resnet50.yaml
│   ├── mobilenet_v3_small.yaml
│   ├── mobilenet_v3_large.yaml
│   ├── vit_b_32.yaml
│   ├── vit_l_32.yaml
│   ├── convnext_base.yaml
│   ├── convnext_large.yaml
│   ├── mixer_b16.yaml
│   └── mixer_l16.yaml
├── dataset/                 # Dataset configs
│   └── cifar10.yaml
├── training/                # Training configs
│   └── default.yaml
└── inference/               # Inference configs
    └── default.yaml
```

## Available Models

### ResNet Family
- `resnet18`, `resnet34`, `resnet50`
- Batch size: 128, LR: 0.001

### MobileNet Family
- `mobilenet_v3_small`, `mobilenet_v3_large`
- Batch size: 128, LR: 0.001

### Vision Transformer (ViT)
- `vit_b_32` (Base/32), `vit_l_32` (Large/32)
- Batch size: 64/32, LR: 0.0001

### ConvNeXt
- `convnext_base`, `convnext_large`
- Batch size: 16/8, LR: 0.0001
- Uses gradient accumulation (4 steps)

### MLP-Mixer (requires timm)
- `mixer_b16` (Base/16), `mixer_l16` (Large/16)
- Batch size: 16/8, LR: 0.0001
- Uses gradient accumulation (4 steps)

## Configuration Examples

### Custom Training Config

```yaml
# configs/training/custom.yaml
training:
  epochs: 20
  batch_size: 64
  learning_rate: 0.0005
  gradient_accumulation_steps: 2
  log_interval: 50
```

Usage:
```bash
python train.py training=custom model=resnet50
```

### Custom Inference Config

```yaml
# configs/inference/detailed.yaml
inference:
  samples_per_class: 100
  max_thresholds: 6
  
  tsne:
    pca_components: 100
    perplexity: 50
```

Usage:
```bash
python infer.py inference=detailed model=resnet50
```

## Output Structure

### Training Outputs
```
finetune/
├── resnet18_cifar10.pth
├── resnet34_cifar10.pth
├── resnet50_cifar10.pth
└── ...
```

### Inference Outputs
```
inference/hole_outputs/
├── resnet18/
│   ├── core/
│   │   ├── heatmap_dendrogram.png
│   │   ├── blob_visualization.png
│   │   ├── sankey_flow.png
│   │   ├── stacked_bars.png
│   │   ├── persistence_diagram.png
│   │   ├── persistence_barcode.png
│   │   ├── pca_analysis.png
│   │   └── tsne_pca.png
│   ├── mds/
│   │   ├── mds_euclidean.png
│   │   └── mds_cosine.png
│   └── heatmaps/
│       ├── heatmap_euclidean.png
│       └── heatmap_cosine.png
└── ...
```

## Advanced Usage

### Sweeps (Multi-run)

Train multiple models with different hyperparameters:

```bash
# Train all ResNet variants
python train.py -m model=resnet18,resnet34,resnet50

# Hyperparameter sweep
python train.py -m training.learning_rate=0.001,0.0001,0.00001 model=resnet50

# Grid sweep
python train.py -m model=resnet18,resnet34 training.batch_size=64,128
```

### Override Nested Configs

```bash
# Override specific model settings
python train.py model=resnet50 model.pretrained=false

# Override dataset settings
python train.py dataset.image_size=384

# Override multiple settings
python train.py \
  model=convnext_base \
  training.epochs=15 \
  training.learning_rate=0.0002 \
  inference.samples_per_class=75
```

### Custom Device and Workers

```bash
# Use CPU
python train.py device=cpu

# Change number of workers
python train.py num_workers=8
```

## Visualization Outputs

All inference runs generate the following visualizations:

1. **Heatmap Dendrogram**: Distance matrix with hierarchical clustering
2. **Blob Visualization**: PCA projection with cluster convex hulls
3. **Sankey Flow**: Cluster evolution across filtration thresholds
4. **Stacked Bars**: Cluster composition over thresholds
5. **Persistence Diagram**: Topological features birth-death diagram
6. **Persistence Barcode**: Barcode representation of persistence
7. **PCA Analysis**: 2D PCA projection colored by true labels
8. **t-SNE with PCA**: t-SNE visualization with PCA preprocessing
9. **MDS Plots**: Multidimensional scaling for different distance metrics
10. **Distance Heatmaps**: Heatmaps for Euclidean and Cosine distances

## Pipeline Workflow

### Complete Pipeline for New Model

```bash
# 1. Train the model
python train.py model=convnext_base

# 2. Run inference and HOLE analysis
python infer.py model=convnext_base

# 3. Results are saved to:
#    - Model: finetune/convnext_base_cifar10.pth
#    - Visualizations: inference/hole_outputs/convnext_base/
```

### Batch Processing All Models

```bash
# Train all models
python train.py -m model=resnet18,resnet34,resnet50,mobilenet_v3_small,mobilenet_v3_large,vit_b_32,vit_l_32,convnext_base,convnext_large,mixer_b16,mixer_l16

# Run inference on all models
python infer.py -m model=resnet18,resnet34,resnet50,mobilenet_v3_small,mobilenet_v3_large,vit_b_32,vit_l_32,convnext_base,convnext_large,mixer_b16,mixer_l16
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or increase gradient accumulation:

```bash
python train.py model=convnext_large training.batch_size=4 training.gradient_accumulation_steps=8
```

### Missing timm Module

Install timm for MLP-Mixer models:

```bash
pip install timm
```

### Configuration Not Found

Ensure you're running from the `examples/modelling/` directory:

```bash
cd examples/modelling/
python train.py
```

## Legacy Scripts

The original individual training and inference scripts are still available in:
- `finetune/*.py` - Individual training scripts
- `inference/*_infer.py` - Individual inference scripts

These are kept for backward compatibility but the new Hydra-based pipeline is recommended.

## References

- **Hydra**: https://hydra.cc/
- **HOLE**: Higher Order Laplacian Eigenmaps for topological data analysis
- **Models**: PyTorch torchvision and timm libraries





