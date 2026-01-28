"""
Unified inference script using Hydra configuration.

Usage:
    # Run inference with default config (ResNet50)
    python infer.py
    
    # Run inference on specific model
    python infer.py model=convnext_base
    
    # Override parameters
    python infer.py model=mixer_b16 inference.samples_per_class=100
    
    # Run inference on multiple models
    python infer.py -m model=resnet18,resnet34,resnet50
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import (
    resnet18, resnet34, resnet50,
    mobilenet_v3_small, mobilenet_v3_large,
    vit_b_32, vit_l_32,
    convnext_base, convnext_large,
)
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import hole
from hole.core import distance_metrics
from hole.visualization.cluster_flow import ClusterFlowAnalyzer, FlowVisualizer


def get_model(cfg: DictConfig, num_classes: int, device):
    """Load trained model based on configuration."""
    model_name = cfg.model.name
    architecture = cfg.model.architecture
    
    # Handle timm models
    if cfg.model.get('requires_timm', False):
        import timm
        model = timm.create_model(cfg.model.weights, pretrained=False, num_classes=num_classes)
    else:
        # Map model names to constructors
        model_map = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'mobilenet_v3_small': mobilenet_v3_small,
            'mobilenet_v3_large': mobilenet_v3_large,
            'vit_b_32': vit_b_32,
            'vit_l_32': vit_l_32,
            'convnext_base': convnext_base,
            'convnext_large': convnext_large,
        }
        
        model_fn = model_map[model_name]
        model = model_fn(weights=None)
        
        # Modify final layer for num_classes
        if architecture == 'resnet':
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif architecture == 'mobilenet_v3':
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        elif architecture == 'vit':
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        elif architecture == 'convnext':
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    
    # Load checkpoint
    checkpoint_path = Path(cfg.inference.checkpoint_dir) / f"{model_name}_cifar10.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def get_hook_layer(model, cfg: DictConfig):
    """Get the layer to hook for activation extraction."""
    hook_layer_name = cfg.model.hook_layer
    
    # Navigate to the hook layer
    parts = hook_layer_name.split('.')
    layer = model
    for part in parts:
        layer = getattr(layer, part)
    
    return layer


def extract_activations(model, loader, device, cfg: DictConfig):
    """Extract activations from the model."""
    activations = []
    labels_list = []
    
    def hook_fn(module, input, output):
        # Handle different layer types
        if cfg.model.architecture == 'vit' or cfg.model.get('requires_timm', False):
            activations.append(input[0].detach().cpu().numpy())
        else:
            activations.append(output.detach().cpu().numpy())
    
    # Register hook
    hook_layer = get_hook_layer(model, cfg)
    hook = hook_layer.register_forward_hook(hook_fn)
    
    # Extract
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _ = model(images)
            labels_list.extend(labels.numpy())
    
    hook.remove()
    
    # Combine and flatten
    activations = np.concatenate(activations, axis=0)
    activations = activations.reshape(activations.shape[0], -1)
    labels_array = np.array(labels_list)
    
    return activations, labels_array


@hydra.main(version_base=None, config_path="configs", config_name="config")
def infer(cfg: DictConfig):
    """Main inference function."""
    print("=" * 80)
    print(f"Running inference for {cfg.model.name} on {cfg.dataset.name}")
    print("=" * 80)
    
    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.Resize(cfg.dataset.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            cfg.dataset.test_transforms.normalize.mean,
            cfg.dataset.test_transforms.normalize.std
        )
    ])
    
    test_dataset = datasets.CIFAR10(
        root=cfg.dataset.data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Sample images per class
    print(f"Sampling {cfg.inference.samples_per_class} images per class...")
    indices = []
    class_counts = {i: 0 for i in range(cfg.dataset.num_classes)}
    for idx, (_, label) in enumerate(test_dataset):
        if class_counts[label] < cfg.inference.samples_per_class:
            indices.append(idx)
            class_counts[label] += 1
        if all(count >= cfg.inference.samples_per_class for count in class_counts.values()):
            break
    
    sampled_dataset = Subset(test_dataset, indices)
    loader = DataLoader(sampled_dataset, batch_size=32, shuffle=False)
    print(f"Total samples: {len(sampled_dataset)}")
    
    # Load model
    print(f"Loading model from {cfg.inference.checkpoint_dir}...")
    model = get_model(cfg, cfg.dataset.num_classes, device)
    
    # Extract activations
    print("Extracting activations...")
    activations, labels_array = extract_activations(model, loader, device, cfg)
    print(f"Activations shape: {activations.shape}")
    print(f"Labels shape: {labels_array.shape}")
    
    # Create output directory
    output_dir = Path(cfg.inference.output_base_dir) / cfg.model.name
    for subdir in ['core', 'mds', 'heatmaps']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("\n=== Running HOLE Analysis ===\n")
    
    # Create HOLE visualizer
    print("Creating HOLE visualizer...")
    visualizer = hole.HOLEVisualizer(
        point_cloud=activations, 
        distance_metric=cfg.inference.distance_metric
    )
    print(f"Computed persistence with {len(visualizer.persistence)} features\n")
    
    viz_cfg = cfg.inference.visualizations
    
    # 1. HEATMAP DENDROGRAM
    if viz_cfg.heatmap_dendrogram:
        print("1. Creating heatmap dendrogram...")
        heatmap_viz = visualizer.get_persistence_dendrogram_visualizer(
            distance_matrix=visualizer.distance_matrix
        )
        heatmap_viz.compute_persistence()
        heatmap_viz.plot_dendrogram_with_heatmap(figsize=(16, 8), cmap="gray")
        plt.savefig(output_dir / "core/heatmap_dendrogram.png", dpi=cfg.inference.viz_settings.dpi, bbox_inches="tight")
        plt.close()
    
    # 2. BLOB VISUALIZATION
    if viz_cfg.blob_visualization:
        print("2. Creating blob visualization...")
        analyzer = ClusterFlowAnalyzer(visualizer.distance_matrix, max_thresholds=cfg.inference.max_thresholds)
        cluster_evolution = analyzer.compute_cluster_evolution(labels_array)
        
        euclidean_labels = cluster_evolution["labels_"]["Euclidean"]
        thresholds = sorted([float(t) for t in euclidean_labels.keys()])
        middle_threshold = thresholds[min(1, len(thresholds)-1)]
        
        blob_viz = visualizer.get_blob_visualizer(figsize=tuple(cfg.inference.viz_settings.figsize_large))
        fig = blob_viz.plot_pca_with_cluster_hulls(
            activations,
            labels_array,
            middle_threshold,
            save_path=str(output_dir / "core/blob_visualization.png"),
            title=f"{cfg.model.name} - Threshold: {middle_threshold:.3f}",
        )
        plt.close(fig)
    
    # 3. CLUSTER FLOW ANALYSIS
    if viz_cfg.sankey_flow or viz_cfg.stacked_bars:
        print("3. Creating cluster flow analysis...")
        class_names_dict = {i: name for i, name in enumerate(cfg.dataset.class_names)}
        flow_viz = FlowVisualizer(
            figsize=tuple(cfg.inference.viz_settings.figsize_flow),
            class_names=class_names_dict
        )
        
        if viz_cfg.sankey_flow:
            sankey_fig = flow_viz.plot_sankey_flow(
                cluster_evolution,
                save_path=str(output_dir / "core/sankey_flow.png"),
                show_true_labels_text=False,
                show_filtration_text=False,
            )
            plt.close(sankey_fig)
        
        if viz_cfg.stacked_bars:
            bars_fig = flow_viz.plot_stacked_bar_evolution(
                cluster_evolution,
                save_path=str(output_dir / "core/stacked_bars.png"),
                show_true_labels_text=False,
                show_filtration_text=False,
            )
            plt.close(bars_fig)
    
    # 4. PERSISTENCE VISUALIZATIONS
    print("4. Creating persistence visualizations...")
    
    if viz_cfg.persistence_diagram:
        fig1, ax1 = plt.subplots(1, 1, figsize=tuple(cfg.inference.viz_settings.figsize_core))
        visualizer.plot_persistence_diagram(ax=ax1, title="Persistence Diagram", pts=20)
        plt.tight_layout()
        fig1.savefig(output_dir / "core/persistence_diagram.png", dpi=cfg.inference.viz_settings.dpi, bbox_inches="tight")
        plt.close(fig1)
    
    if viz_cfg.persistence_barcode:
        fig2, ax2 = plt.subplots(1, 1, figsize=tuple(cfg.inference.viz_settings.figsize_core))
        visualizer.plot_persistence_barcode(ax=ax2, title="Persistence Barcode", pts=20)
        plt.tight_layout()
        fig2.savefig(output_dir / "core/persistence_barcode.png", dpi=cfg.inference.viz_settings.dpi, bbox_inches="tight")
        plt.close(fig2)
    
    if viz_cfg.pca_analysis:
        fig3, ax3 = plt.subplots(1, 1, figsize=tuple(cfg.inference.viz_settings.figsize_core))
        visualizer.plot_dimensionality_reduction(
            method="pca", ax=ax3, true_labels=labels_array, title="PCA"
        )
        ax3.grid(False)
        for collection in ax3.collections:
            if hasattr(collection, "set_sizes"):
                collection.set_sizes([cfg.inference.viz_settings.point_size])
        plt.tight_layout()
        fig3.savefig(output_dir / "core/pca_analysis.png", dpi=cfg.inference.viz_settings.dpi, bbox_inches="tight")
        plt.close(fig3)
    
    # t-SNE with PCA preprocessing
    if viz_cfg.tsne_pca:
        print("   Creating t-SNE with PCA preprocessing...")
        pca_components = min(cfg.inference.tsne.pca_components, activations.shape[0]-1, activations.shape[1])
        pca = PCA(n_components=pca_components)
        activations_pca = pca.fit_transform(activations)
        
        perplexity = min(cfg.inference.tsne.perplexity, (activations_pca.shape[0] - 1) // 3)
        perplexity = max(5, perplexity)
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity, 
            random_state=cfg.inference.tsne.random_state, 
            n_iter=cfg.inference.tsne.n_iter
        )
        tsne_results = tsne.fit_transform(activations_pca)
        
        fig4, ax4 = plt.subplots(1, 1, figsize=tuple(cfg.inference.viz_settings.figsize_core))
        scatter = ax4.scatter(
            tsne_results[:, 0], tsne_results[:, 1],
            c=labels_array, cmap='tab10', 
            s=cfg.inference.viz_settings.point_size, alpha=0.7
        )
        ax4.set_title("t-SNE (with PCA preprocessing)")
        ax4.grid(False)
        plt.colorbar(scatter, ax=ax4)
        plt.tight_layout()
        fig4.savefig(output_dir / "core/tsne_pca.png", dpi=cfg.inference.viz_settings.dpi, bbox_inches="tight")
        plt.close(fig4)
    
    # 5. DISTANCE METRIC ANALYSIS
    if viz_cfg.mds_plots or viz_cfg.distance_heatmaps:
        print("5. Creating visualizations for distance metrics...")
        
        distance_matrices = {}
        for metric_name in cfg.inference.distance_metrics:
            if metric_name == "euclidean":
                distance_matrices[metric_name] = distance_metrics.euclidean_distance(activations)
            elif metric_name == "cosine":
                distance_matrices[metric_name] = distance_metrics.cosine_distance(activations)
        
        for metric_name, dist_matrix in distance_matrices.items():
            print(f"   Processing {metric_name}...")
            
            if viz_cfg.mds_plots:
                from hole.visualization.persistence_vis import plot_dimensionality_reduction
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                plot_dimensionality_reduction(
                    dist_matrix,
                    method="mds",
                    ax=ax,
                    labels=labels_array,
                    figsize=(10, 8),
                    show_legend=False,
                )
                ax.set_title("")
                ax.grid(False)
                for collection in ax.collections:
                    if hasattr(collection, "set_sizes"):
                        collection.set_sizes([cfg.inference.viz_settings.point_size])
                plt.tight_layout()
                fig.savefig(output_dir / f"mds/mds_{metric_name}.png", dpi=cfg.inference.viz_settings.dpi, bbox_inches="tight")
                plt.close(fig)
            
            if viz_cfg.distance_heatmaps:
                heatmap_viz = hole.PersistenceDendrogram(distance_matrix=dist_matrix)
                fig, ax = heatmap_viz.plot_rcm_heatmap(
                    title=f"{metric_name} Distance Matrix", figsize=(10, 8), cmap="viridis"
                )
                fig.savefig(output_dir / f"heatmaps/heatmap_{metric_name}.png", dpi=cfg.inference.viz_settings.dpi, bbox_inches="tight")
                plt.close(fig)
    
    print("\n=== HOLE Analysis Complete ===")
    print(f"All visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    infer()





