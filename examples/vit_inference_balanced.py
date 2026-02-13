"""
ViT CIFAR-10 Inference with HOLE Analysis - Balanced Subset

This script loads a trained ViT model, runs inference on a balanced subset
of CIFAR-10 (15 images per class), extracts embeddings, and performs HOLE analysis.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor
import matplotlib.pyplot as plt

import hole
from hole.visualization.cluster_flow import ClusterFlowAnalyzer, FlowVisualizer
from hole.visualization.persistence_vis import plot_dimensionality_reduction


def extract_vit_embeddings(model, dataloader, device):
    """Extract embeddings from all ViT encoder layers separately."""
    model.eval()
    layer_embeddings_list = None  # Will store list of arrays, one per layer
    labels_list = []
    
    # Register hooks on all encoder layers
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # Encoder layers return tuple (hidden_states, ...)
            if isinstance(output, tuple):
                activation[name] = output[0].detach()
            else:
                activation[name] = output.detach()
        return hook
    
    # Hook into all encoder layers
    hooks = []
    n_layers = len(model.vit.encoder.layer)
    for i, layer in enumerate(model.vit.encoder.layer):
        hook = layer.register_forward_hook(get_activation(f'layer_{i}'))
        hooks.append(hook)
    
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch['pixel_values'].to(device)
            batch_labels = batch['labels'].cpu().numpy()
            
            # Forward pass
            _ = model(pixel_values)
            
            # Extract CLS token embeddings from all layers
            if layer_embeddings_list is None:
                # Initialize list of lists for each layer
                layer_embeddings_list = [[] for _ in range(n_layers)]
            
            for i in range(n_layers):
                cls_token = activation[f'layer_{i}'][:, 0, :].cpu().numpy()
                layer_embeddings_list[i].append(cls_token)
            
            labels_list.append(batch_labels)
    
    # Remove all hooks
    for hook in hooks:
        hook.remove()
    
    # Stack embeddings for each layer
    layer_embeddings = {}
    for i in range(n_layers):
        layer_embeddings[f'layer_{i}'] = np.vstack(layer_embeddings_list[i])
    
    labels = np.concatenate(labels_list)
    
    return layer_embeddings, labels


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def select_balanced_subset(dataset, n_per_class=15):
    """Select exactly n_per_class samples from each class."""
    # Get all labels
    all_labels = np.array([dataset[i]['label'] for i in range(len(dataset))])
    
    selected_indices = []
    np.random.seed(42)
    
    for class_id in range(10):  # CIFAR-10 has 10 classes
        class_indices = np.where(all_labels == class_id)[0]
        # Randomly select n_per_class samples from this class
        selected = np.random.choice(class_indices, n_per_class, replace=False)
        selected_indices.extend(selected)
    
    return np.array(selected_indices)


def main():
    """Run ViT inference on balanced CIFAR-10 subset with HOLE analysis."""
    
    # Configuration
    N_PER_CLASS = 20  # 100 images per class (1000 total)
    BATCH_SIZE = 32
    MODEL_PATH = "../vit_cifar10_finetuned"
    OUTPUT_DIR = "vit_balanced_hole_outputs"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/core", exist_ok=True)
    
    print("=== ViT CIFAR-10 Inference with HOLE Analysis (Balanced Subset) ===\n")
    
    # Load model
    print(f"Loading ViT model from {MODEL_PATH}...")
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}\n")
    
    # Load CIFAR-10
    print("Loading CIFAR-10 dataset...")
    test_ds = load_dataset('cifar10', split='test')
    
    # Setup transforms
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    
    normalize = Normalize(mean=image_mean, std=image_std)
    transforms = Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])
    
    def transform_fn(examples):
        examples['pixel_values'] = [transforms(image.convert("RGB")) for image in examples['img']]
        return examples
    
    test_ds.set_transform(transform_fn)
    
    # Select balanced subset
    print(f"Selecting {N_PER_CLASS} samples per class...")
    balanced_indices = select_balanced_subset(test_ds, n_per_class=N_PER_CLASS)
    subset_ds = Subset(test_ds, balanced_indices)
    
    print(f"Selected {len(balanced_indices)} samples ({N_PER_CLASS} per class)\n")
    
    # Create dataloader
    dataloader = DataLoader(subset_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Extract embeddings
    print("Extracting ViT embeddings from all layers...")
    layer_embeddings, labels = extract_vit_embeddings(model, dataloader, DEVICE)
    n_layers = len(layer_embeddings)
    print(f"Extracted embeddings from {n_layers} layers")
    print(f"Labels shape: {labels.shape}\n")
    
    # Process each layer separately
    for layer_idx in range(n_layers):
        layer_name = f'layer_{layer_idx}'
        embeddings = layer_embeddings[layer_name]
        
        print(f"\n{'='*60}")
        print(f"Processing Layer {layer_idx} (shape: {embeddings.shape})")
        print(f"{'='*60}\n")
        
        # Create layer-specific output directory
        layer_output_dir = f"{OUTPUT_DIR}/{layer_name}"
        os.makedirs(layer_output_dir, exist_ok=True)
        os.makedirs(f"{layer_output_dir}/core", exist_ok=True)
        
        # CIFAR-10 class names
        cifar10_classes = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
        }
        
        # Create HOLE visualizer for this layer
        print(f"Layer {layer_idx}: Creating HOLE visualizer...")
        visualizer = hole.HOLEVisualizer(point_cloud=embeddings, distance_metric="euclidean")
        print(f"Layer {layer_idx}: Computed persistence with {len(visualizer.persistence)} features\n")
        
        # 1. PCA Visualization
        print(f"Layer {layer_idx}: 1. Creating PCA visualization...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plot_dimensionality_reduction(
            embeddings,
            method="pca",
            labels=labels,
            ax=ax,
            title=f"Layer {layer_idx} - PCA",
            point_size=50,
            alpha=0.7
        )
        plt.savefig(f"{layer_output_dir}/pca_visualization.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # 2. Heatmap Dendrogram
        print(f"Layer {layer_idx}: 2. Creating heatmap dendrogram...")
        heatmap_viz = visualizer.get_persistence_dendrogram_visualizer(
            distance_matrix=visualizer.distance_matrix
        )
        heatmap_viz.compute_persistence()
        heatmap_viz.plot_dendrogram_with_heatmap(figsize=(16, 8), cmap="gray")
        plt.savefig(f"{layer_output_dir}/heatmap_dendrogram.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # 3. Blob Visualization
        print(f"Layer {layer_idx}: 3. Creating blob visualization...")
        analyzer = ClusterFlowAnalyzer(visualizer.distance_matrix, max_thresholds=4)
        cluster_evolution = analyzer.compute_cluster_evolution(
            labels,
            filter_small_clusters=True,
            min_cluster_size=2
        )
        
        euclidean_labels = cluster_evolution["labels_"]["Euclidean"]
        thresholds = sorted([float(t) for t in euclidean_labels.keys()])
        middle_threshold = thresholds[1] if len(thresholds) > 1 else thresholds[0]
        
        print(f"Layer {layer_idx}:    Using threshold: {middle_threshold:.3f}")
        
        blob_viz = visualizer.get_blob_visualizer(
            figsize=(10, 8),
            outlier_percentage=0.0,
            show_contours=False
        )
        fig = blob_viz.plot_pca_with_cluster_hulls(
            embeddings,
            labels,
            middle_threshold,
            save_path=f"{layer_output_dir}/blob_visualization.png",
            title=f"Layer {layer_idx} - Blob Visualization (Threshold: {middle_threshold:.3f})",
        )
        plt.close(fig)
        
        # 4. Cluster Flow Analysis
        print(f"Layer {layer_idx}: 4. Creating cluster flow analysis...")
        flow_viz = FlowVisualizer(figsize=(18, 10), class_names=cifar10_classes)
        
        sankey_fig = flow_viz.plot_sankey_flow(
            cluster_evolution,
            save_path=f"{layer_output_dir}/sankey_flow.png",
            show_true_labels_text=False,
            show_filtration_text=False,
        )
        plt.close(sankey_fig)
        
        bars_fig = flow_viz.plot_stacked_bar_evolution(
            cluster_evolution,
            save_path=f"{layer_output_dir}/stacked_bars.png",
            show_true_labels_text=False,
            show_filtration_text=False,
        )
        plt.close(bars_fig)
        
        # 5. Persistence Visualizations
        print(f"Layer {layer_idx}: 5. Creating persistence visualizations...")
        
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        visualizer.plot_persistence_diagram(ax=ax1, title=f"Layer {layer_idx} - Persistence Diagram", pts=20)
        plt.tight_layout()
        fig1.savefig(f"{layer_output_dir}/persistence_diagram.png", dpi=300, bbox_inches="tight")
        plt.close(fig1)
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
        visualizer.plot_persistence_barcode(ax=ax2, title=f"Layer {layer_idx} - Persistence Barcode", pts=20)
        plt.tight_layout()
        fig2.savefig(f"{layer_output_dir}/persistence_barcode.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)
        
        print(f"Layer {layer_idx}: ✓ All visualizations saved to {layer_output_dir}/\n")
    
    print(f"\n{'='*60}")
    print(f"✓ All layers processed successfully!")
    print(f"✓ Results saved to {OUTPUT_DIR}/")
    print("=== Analysis Complete ===")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
