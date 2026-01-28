#!/bin/bash
# Complete pipeline for training and inference on all models

set -e  # Exit on error

echo "========================================="
echo "Evaluation Training & Inference Pipeline"
echo "========================================="
echo ""

# Parse arguments
MODE=${1:-all}  # train, infer, or all
MODELS=${2:-all}  # comma-separated model names or 'all'

# Define all models
ALL_MODELS="resnet18,resnet34,resnet50,mobilenet_v3_small,mobilenet_v3_large,vit_b_32,vit_l_32,convnext_base,convnext_large,mixer_b16,mixer_l16"

# Set models to process
if [ "$MODELS" = "all" ]; then
    MODEL_LIST=$ALL_MODELS
else
    MODEL_LIST=$MODELS
fi

echo "Mode: $MODE"
echo "Models: $MODEL_LIST"
echo ""

# Training
if [ "$MODE" = "train" ] || [ "$MODE" = "all" ]; then
    echo "========================================="
    echo "Starting Training..."
    echo "========================================="
    python3 train.py -m model=$MODEL_LIST
    echo ""
    echo "Training complete!"
    echo ""
fi

# Inference
if [ "$MODE" = "infer" ] || [ "$MODE" = "all" ]; then
    echo "========================================="
    echo "Starting Inference..."
    echo "========================================="
    python3 infer.py -m model=$MODEL_LIST
    echo ""
    echo "Inference complete!"
    echo ""
fi

echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Results:"
echo "  - Models: finetune/"
echo "  - Visualizations: inference/hole_outputs/"




