# HOLE Examples

This directory contains examples demonstrating the HOLE library functionality.

## Directory Structure

```
examples/
  core/          # Core HOLE demos with synthetic data
  nlp/           # NLP examples (BERT + HOLE)
  modeling/
    finetune/    # Vision model fine-tuning scripts
    inference/   # Model inference + HOLE analysis
```

## Core Examples (`core/`)

### `hole_example.py`
**Recommended starting point** - Shows HOLE's core visualization capabilities.
- Sankey diagrams for cluster evolution
- Stacked bar charts for threshold analysis
- Heatmap dendrograms for distance matrices
- Blob visualizations with class-colored contours and outlier detection
- Persistence diagrams and barcodes
- MDS plots for all distance metrics

### `blob_contour_demo.py`
**Blob contour visualization demo** - Demonstrates contour and outlier class features.
- Shows how persistent homology clusters contain mixed classes
- Percentage-based outlier class detection
- Class-colored contours for majority classes
- Multiple threshold examples (5%, 10%, 15%)

### `distance_metrics.py`
**Advanced comprehensive analysis** - Systematic comparison across data structures and metrics.
- 5 data structures (isotropic clusters, hypersphere, elliptical, Swiss roll, tight blobs)
- 7 distance metrics (euclidean, cosine, mahalanobis, density-normalized variants, geodesic)
- Generates complete visualization suites for each combination

## NLP Examples (`nlp/`)

### `bert_sentiment_classification.py`
**BERT sentiment analysis + HOLE** - Topological analysis of BERT hidden states.
- Model: `textattack/bert-base-uncased-SST-2`
- Extracts hidden-state embeddings from each BERT layer (0-11)
- Runs full HOLE pipeline with filter and no-filter variants

### `bert_ner_classification.py`
**BERT Named Entity Recognition + HOLE** - NER with topological analysis.
- Model: `dslim/bert-base-NER`
- Dataset: CoNLL-2003 with collapsed BIO tags (O, PER, ORG, LOC, MISC)
- Per-entity embeddings analyzed through HOLE pipeline

## Modeling (`modeling/`)

### `finetune/`
Fine-tuning scripts for vision models on CIFAR-10:
- ResNet-18, ResNet-50
- ViT-Base, ViT-Large
- ConvNeXt-Base, ConvNeXt-Large
- EfficientNet-B0
- MobileNetV2 (small, large)

See [`finetune/README.md`](modeling/finetune/README.md) for details.

### `inference/`
Unified inference scripts with HOLE topological analysis:
- `resnet50_inference_unified.py` - ResNet-50 inference with 7 noise conditions
- `vit_inference_unified.py` - ViT inference with 7 noise conditions

Experiments: balanced, gaussian, salt & pepper, speckle, poisson, uniform, quantized (INT8).

## Running Examples

```bash
# Core examples
cd examples/core
python hole_example.py           # Recommended starting point
python blob_contour_demo.py      # Blob contour features
python distance_metrics.py       # Comprehensive analysis (longer runtime)

# NLP examples (requires torch, transformers, datasets)
cd examples/nlp
python bert_sentiment_classification.py
python bert_ner_classification.py

# Inference (requires fine-tuned model checkpoints)
cd examples/modeling/inference
python resnet50_inference_unified.py
python vit_inference_unified.py
```

## Requirements

All examples require the HOLE library installed with its dependencies:
- numpy, matplotlib, scikit-learn, gudhi, scipy, seaborn

**NLP examples** additionally require: torch, transformers, datasets

**Modeling examples** additionally require: torch, transformers, datasets, torchvision, timm
