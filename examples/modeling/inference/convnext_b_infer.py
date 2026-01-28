# Model: ConvNeXt_Base_Weights.IMAGENET1K_V1
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

import hole
from hole.core import distance_metrics
from hole.visualization.cluster_flow import ClusterFlowAnalyzer, FlowVisualizer

# Config
SAMPLES_PER_CLASS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../finetune/convnext_b_cifar10.pth"
OUTPUT_DIR = "hole_outputs/convnext_b"

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Data transforms
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 test set
print("Loading CIFAR-10 test set...")
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Sample 50 images per class
print(f"Sampling {SAMPLES_PER_CLASS} images per class...")
indices = []
class_counts = {i: 0 for i in range(10)}
for idx, (_, label) in enumerate(test_dataset):
    if class_counts[label] < SAMPLES_PER_CLASS:
        indices.append(idx)
        class_counts[label] += 1
    if all(count >= SAMPLES_PER_CLASS for count in class_counts.values()):
        break

sampled_dataset = Subset(test_dataset, indices)
loader = DataLoader(sampled_dataset, batch_size=32, shuffle=False)

print(f"Total samples: {len(sampled_dataset)}")

# Load model
print("Loading finetuned ConvNeXt-Base...")
model = convnext_base(weights=None)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, 10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Hook to extract activations from the layer before final FC
activations = []
labels_list = []

def hook_fn(module, input, output):
    activations.append(output.detach().cpu().numpy())

# Register hook on the avgpool layer (before classifier)
hook = model.avgpool.register_forward_hook(hook_fn)

# Extract activations
print("Extracting activations...")
with torch.no_grad():
    for images, labels in loader:
        images = images.to(DEVICE)
        _ = model(images)
        labels_list.extend(labels.numpy())

hook.remove()

# Combine activations
activations = np.concatenate(activations, axis=0)
activations = activations.reshape(activations.shape[0], -1)  # Flatten
labels_array = np.array(labels_list)

print(f"Activations shape: {activations.shape}")
print(f"Labels shape: {labels_array.shape}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/core", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/mds", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/heatmaps", exist_ok=True)

print("\n=== Running HOLE Analysis ===\n")

# Create HOLE visualizer
print("Creating HOLE visualizer...")
visualizer = hole.HOLEVisualizer(point_cloud=activations, distance_metric="euclidean")
print(f"Computed persistence with {len(visualizer.persistence)} features\n")

# 1. HEATMAP DENDROGRAM
print("1. Creating heatmap dendrogram...")
heatmap_viz = visualizer.get_persistence_dendrogram_visualizer(
    distance_matrix=visualizer.distance_matrix
)
heatmap_viz.compute_persistence()
heatmap_viz.plot_dendrogram_with_heatmap(figsize=(16, 8), cmap="gray")
plt.savefig(f"{OUTPUT_DIR}/core/heatmap_dendrogram.png", dpi=300, bbox_inches="tight")
plt.close()

# 2. BLOB VISUALIZATION
print("2. Creating blob visualization...")
analyzer = ClusterFlowAnalyzer(visualizer.distance_matrix, max_thresholds=4)
cluster_evolution = analyzer.compute_cluster_evolution(labels_array)

euclidean_labels = cluster_evolution["labels_"]["Euclidean"]
thresholds = sorted([float(t) for t in euclidean_labels.keys()])
middle_threshold = thresholds[min(1, len(thresholds)-1)]

blob_viz = visualizer.get_blob_visualizer(figsize=(12, 9))
fig = blob_viz.plot_pca_with_cluster_hulls(
    activations,
    labels_array,
    middle_threshold,
    save_path=f"{OUTPUT_DIR}/core/blob_visualization.png",
    title=f"ConvNeXt-Base - Threshold: {middle_threshold:.3f}",
)
plt.close(fig)

# 3. CLUSTER FLOW ANALYSIS
print("3. Creating cluster flow analysis...")
flow_viz = FlowVisualizer(
    figsize=(18, 10), 
    class_names={i: CIFAR10_CLASSES[i] for i in range(10)}
)

sankey_fig = flow_viz.plot_sankey_flow(
    cluster_evolution,
    save_path=f"{OUTPUT_DIR}/core/sankey_flow.png",
    show_true_labels_text=False,
    show_filtration_text=False,
)
plt.close(sankey_fig)

bars_fig = flow_viz.plot_stacked_bar_evolution(
    cluster_evolution,
    save_path=f"{OUTPUT_DIR}/core/stacked_bars.png",
    show_true_labels_text=False,
    show_filtration_text=False,
)
plt.close(bars_fig)

# 4. PERSISTENCE VISUALIZATIONS
print("4. Creating persistence visualizations...")

fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
visualizer.plot_persistence_diagram(ax=ax1, title="Persistence Diagram", pts=20)
plt.tight_layout()
fig1.savefig(f"{OUTPUT_DIR}/core/persistence_diagram.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
visualizer.plot_persistence_barcode(ax=ax2, title="Persistence Barcode", pts=20)
plt.tight_layout()
fig2.savefig(f"{OUTPUT_DIR}/core/persistence_barcode.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
visualizer.plot_dimensionality_reduction(
    method="pca", ax=ax3, true_labels=labels_array, title="PCA"
)
ax3.grid(False)
for collection in ax3.collections:
    if hasattr(collection, "set_sizes"):
        collection.set_sizes([120])
plt.tight_layout()
fig3.savefig(f"{OUTPUT_DIR}/core/pca_analysis.png", dpi=300, bbox_inches="tight")
plt.close(fig3)

# t-SNE with PCA preprocessing
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE

print("   Creating t-SNE with PCA preprocessing...")
# Apply PCA to reduce to 50 dimensions first
pca_50 = SklearnPCA(n_components=min(50, activations.shape[0]-1, activations.shape[1]))
activations_pca = pca_50.fit_transform(activations)

# Apply t-SNE on PCA-reduced features
perplexity = min(30, (activations_pca.shape[0] - 1) // 3)
perplexity = max(5, perplexity)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
tsne_results = tsne.fit_transform(activations_pca)

# Plot t-SNE results
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 6))
scatter = ax4.scatter(tsne_results[:, 0], tsne_results[:, 1], 
                     c=labels_array, cmap='tab10', s=120, alpha=0.7)
ax4.set_title("t-SNE (with PCA preprocessing)")
ax4.grid(False)
plt.colorbar(scatter, ax=ax4)
plt.tight_layout()
fig4.savefig(f"{OUTPUT_DIR}/core/tsne_pca.png", dpi=300, bbox_inches="tight")
plt.close(fig4)

# 5. DISTANCE METRIC ANALYSIS
print("5. Creating visualizations for distance metrics...")

distance_matrices = {
    "euclidean": distance_metrics.euclidean_distance(activations),
    "cosine": distance_metrics.cosine_distance(activations),
}

for metric_name, dist_matrix in distance_matrices.items():
    print(f"   Processing {metric_name}...")
    
    # MDS Plot
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
            collection.set_sizes([120])
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/mds/mds_{metric_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    # Heatmap
    heatmap_viz = hole.PersistenceDendrogram(distance_matrix=dist_matrix)
    fig, ax = heatmap_viz.plot_rcm_heatmap(
        title=f"{metric_name} Distance Matrix", figsize=(10, 8), cmap="viridis"
    )
    fig.savefig(f"{OUTPUT_DIR}/heatmaps/heatmap_{metric_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

print("\n=== HOLE Analysis Complete ===")
print(f"All visualizations saved to: {OUTPUT_DIR}/")

