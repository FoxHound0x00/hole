"""
Unified ViT CIFAR-10 Inference + HOLE Analysis
===============================================

Experiments
-----------
  balanced          – clean images, full-precision model
  gaussian          – AWGN (σ=0.1)
  salt_and_pepper   – Impulse noise (p=0.10)
  speckle           – Multiplicative noise (σ=0.1)
  poisson           – Shot noise (λ=100)
  uniform           – Additive uniform noise (a=0.1)
  quantized         – clean images, INT8 dynamically-quantized model

Every experiment produces BOTH filter/ and no_filter/ sub-trees.

Output layout
-------------
  vit_hole_outputs/
    balanced/
      metrics.json
      filter/   layer_0/ … layer_11/
      no_filter/ layer_0/ … layer_11/
    gaussian/
      metrics.json
      filter/   layer_0/ … layer_11/
      no_filter/ layer_0/ … layer_11/
    … (one folder per experiment)
    quantized/
      metrics.json
      filter/   …
      no_filter/ …

Reproducibility
---------------
  np.random.seed(42)   – balanced data-subset selection
  torch.manual_seed(42) – noise generation (reset before every noisy experiment)
"""

import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report,
)
from datasets import load_dataset
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor
import matplotlib.pyplot as plt

import hole
from hole.visualization.cluster_flow import ClusterFlowAnalyzer, FlowVisualizer
from hole.visualization.persistence_vis import plot_dimensionality_reduction


# ── configuration ──────────────────────────────────────────────────────────────

MODEL_PATH   = "../../../vit_cifar10_finetuned"
OUTPUT_ROOT  = "vit_hole_outputs_cosine"
N_PER_CLASS  = 20
BATCH_SIZE   = 32

CIFAR10_CLASSES = {
    0: "airplane", 1: "automobile", 2: "bird",  3: "cat",  4: "deer",
    5: "dog",      6: "frog",       7: "horse", 8: "ship", 9: "truck",
}


# ── noise functions ────────────────────────────────────────────────────────────

def apply_gaussian(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """AWGN: x' = x + N(0, σ²)"""
    return x + torch.randn_like(x) * sigma


def apply_salt_and_pepper(x: torch.Tensor, p: float = 0.10) -> torch.Tensor:
    """Impulse noise: each pixel → min or max with probability p."""
    noisy = x.clone()
    mask  = torch.rand_like(x)
    noisy[mask < p / 2]                 = x.max()
    noisy[(mask >= p / 2) & (mask < p)] = x.min()
    return noisy


def apply_speckle(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Multiplicative noise: x' = x + x·N(0, σ²)"""
    return x + x * torch.randn_like(x) * sigma


def apply_poisson(x: torch.Tensor) -> torch.Tensor:
    """Shot noise scaled so SNR is comparable to the other corruption types."""
    lam = 100.0
    mn, mx = x.min(), x.max()
    x_n = (x - mn) / (mx - mn + 1e-8)
    return torch.poisson(x_n * lam) / lam * (mx - mn) + mn


def apply_uniform(x: torch.Tensor, a: float = 0.1) -> torch.Tensor:
    """Additive uniform noise: x' = x + U(-a, a)"""
    return x + (torch.rand_like(x) * 2 * a - a)


NOISE_CONFIGS = {
    "gaussian":        {"fn": apply_gaussian,        "label": "Gaussian (σ=0.1)"},
    "salt_and_pepper": {"fn": apply_salt_and_pepper,  "label": "Salt & Pepper (p=0.10)"},
    "speckle":         {"fn": apply_speckle,          "label": "Speckle (σ=0.1)"},
    "poisson":         {"fn": apply_poisson,          "label": "Poisson (λ=100)"},
    "uniform":         {"fn": apply_uniform,          "label": "Uniform (a=0.1)"},
}


# ── data helpers ───────────────────────────────────────────────────────────────

def collate_fn(examples):
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
        "labels":       torch.tensor([e["label"]       for e in examples]),
    }


def select_balanced_subset(dataset, n_per_class: int = 20, seed: int = 42) -> np.ndarray:
    """Pick exactly n_per_class random samples from each CIFAR-10 class."""
    all_labels = np.array([dataset[i]["label"] for i in range(len(dataset))])
    np.random.seed(seed)
    indices = []
    for c in range(10):
        cls_idx = np.where(all_labels == c)[0]
        indices.extend(np.random.choice(cls_idx, n_per_class, replace=False))
    return np.array(indices)


# ── embedding + metric extraction ──────────────────────────────────────────────

def extract_embeddings_and_metrics(model, dataloader, device, noise_fn=None):
    """
    Run a forward pass (optionally with noise) and collect:
      - CLS-token embeddings for every encoder layer
      - Classification metrics (accuracy, F1, precision, recall, per-class report)

    Parameters
    ----------
    model      : ViTForImageClassification (full-precision or quantized)
    dataloader : DataLoader over the balanced subset
    device     : torch.device
    noise_fn   : callable(tensor) → tensor | None (clean inference if None)

    Returns
    -------
    layer_embeddings : dict  "layer_i" → np.ndarray (N, D)
    labels           : np.ndarray  (N,)
    metrics          : dict
    """
    model.eval()

    # ── register hooks on every encoder layer ─────────────────────────────────
    activation = {}

    def make_hook(name):
        def hook(_, __, output):
            t = output[0] if isinstance(output, tuple) else output
            activation[name] = t.detach()
        return hook

    hooks    = []
    n_layers = len(model.vit.encoder.layer)
    for i, layer in enumerate(model.vit.encoder.layer):
        hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))

    layer_buf  = None
    labels_buf = []
    preds_buf  = []

    # Seed torch RNG before the loop so noise is identical each run
    if noise_fn is not None:
        torch.manual_seed(42)

    with torch.no_grad():
        for batch in dataloader:
            pv  = batch["pixel_values"].to(device)
            lbl = batch["labels"].cpu().numpy()

            inp = noise_fn(pv) if noise_fn is not None else pv
            out = model(inp)

            preds_buf.append(out.logits.argmax(dim=-1).cpu().numpy())
            labels_buf.append(lbl)

            if layer_buf is None:
                layer_buf = [[] for _ in range(n_layers)]
            for i in range(n_layers):
                layer_buf[i].append(activation[f"layer_{i}"][:, 0, :].cpu().numpy())

    for h in hooks:
        h.remove()

    labels = np.concatenate(labels_buf)
    preds  = np.concatenate(preds_buf)

    layer_embeddings = {
        f"layer_{i}": np.vstack(layer_buf[i]) for i in range(n_layers)
    }

    # ── classification metrics ─────────────────────────────────────────────────
    class_names = [CIFAR10_CLASSES[i] for i in range(10)]
    report      = classification_report(labels, preds, target_names=class_names, output_dict=True)

    metrics = {
        "accuracy":        float(accuracy_score(labels, preds)),
        "f1_macro":        float(f1_score(labels, preds, average="macro")),
        "f1_weighted":     float(f1_score(labels, preds, average="weighted")),
        "f1_micro":        float(f1_score(labels, preds, average="micro")),
        "precision_macro": float(precision_score(labels, preds, average="macro")),
        "recall_macro":    float(recall_score(labels, preds, average="macro")),
        "per_class": {
            cls: {k: round(float(v), 4) for k, v in vals.items() if k != "support"}
            for cls, vals in report.items()
            if cls in class_names
        },
    }

    return layer_embeddings, labels, metrics


# ── per-layer HOLE visualizations ─────────────────────────────────────────────

def _paper_ax(ax):
    """Post-process axes for paper-quality output."""
    ax.set_xlabel(ax.get_xlabel(), fontsize=18)
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    ax.tick_params(labelsize=14)
    leg = ax.get_legend()
    if leg:
        leg.set_title(leg.get_title().get_text(), prop={"size": 14})
        for t in leg.get_texts():
            t.set_fontsize(14)


def process_layer(
    layer_idx: int,
    embeddings: np.ndarray,
    labels: np.ndarray,
    exp_label: str,
    layer_dir: str,
    filter_clusters: bool,
):
    os.makedirs(layer_dir, exist_ok=True)
    os.makedirs(f"{layer_dir}/core", exist_ok=True)

    visualizer = hole.HOLEVisualizer(point_cloud=embeddings, distance_metric="cosine")

    # 1. PCA ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_dimensionality_reduction(
        embeddings, method="pca", labels=labels, ax=ax,
        title="",
        point_size=90, alpha=0.7,
        class_names=CIFAR10_CLASSES,
    )
    _paper_ax(ax)
    plt.savefig(f"{layer_dir}/pca_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Heatmap dendrogram ────────────────────────────────────────────────────
    hv = visualizer.get_persistence_dendrogram_visualizer(
        distance_matrix=visualizer.distance_matrix
    )
    hv.compute_persistence()
    hv.plot_dendrogram_with_heatmap(figsize=(16, 8), cmap="gray")
    for a in plt.gcf().get_axes():
        _paper_ax(a)
    plt.savefig(f"{layer_dir}/heatmap_dendrogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Cluster evolution ─────────────────────────────────────────────────────
    analyzer  = ClusterFlowAnalyzer(visualizer.distance_matrix, max_thresholds=4)
    clust_evo = analyzer.compute_cluster_evolution(
        labels,
        filter_small_clusters=filter_clusters,
        min_cluster_size=10,
        metric_name="Cosine",
    )

    thresholds = sorted([float(t) for t in clust_evo["labels_"]["Cosine"].keys()])
    mid_t      = thresholds[1] if len(thresholds) > 1 else thresholds[0]

    # 4. Blob ──────────────────────────────────────────────────────────────────
    bv  = visualizer.get_blob_visualizer(figsize=(12, 9), outlier_percentage=0.0, show_contours=False)
    fig = bv.plot_pca_with_cluster_hulls(
        embeddings, labels, mid_t,
        save_path=None,
        title="",
        metric="cosine",
        class_names=CIFAR10_CLASSES,
    )
    for a in fig.get_axes():
        _paper_ax(a)
    fig.savefig(f"{layer_dir}/blob_visualization.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5. Sankey + stacked bars ─────────────────────────────────────────────────
    fv = FlowVisualizer(figsize=(20, 11), class_names=CIFAR10_CLASSES)

    fig_sankey = fv.plot_sankey_flow(
        clust_evo,
        save_path=None,
        title="",
        show_true_labels_text=False,
        show_filtration_text=False,
    )
    for a in fig_sankey.get_axes():
        _paper_ax(a)
    fig_sankey.savefig(f"{layer_dir}/sankey_flow.png", dpi=300, bbox_inches="tight")
    plt.close(fig_sankey)

    fig_bars = fv.plot_stacked_bar_evolution(
        clust_evo,
        save_path=None,
        title="",
        show_true_labels_text=False,
        show_filtration_text=False,
    )
    for a in fig_bars.get_axes():
        _paper_ax(a)
    fig_bars.savefig(f"{layer_dir}/stacked_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig_bars)

    # 6. Persistence diagram + barcode ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    visualizer.plot_persistence_diagram(
        ax=ax1,
        title="",
        pts=20,
    )
    _paper_ax(ax1)
    plt.tight_layout()
    fig1.savefig(f"{layer_dir}/persistence_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    visualizer.plot_persistence_barcode(
        ax=ax2,
        title="",
        pts=20,
    )
    _paper_ax(ax2)
    plt.tight_layout()
    fig2.savefig(f"{layer_dir}/persistence_barcode.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(f"      Layer {layer_idx}: ✓  →  {layer_dir}/")


# ── single experiment runner ───────────────────────────────────────────────────

def run_experiment(
    exp_key: str,
    exp_label: str,
    model,
    dataloader,
    device: torch.device,
    noise_fn=None,
):
    """
    Extracts embeddings + metrics for one experiment, then writes both
    filter/ and no_filter/ visualizations for all 12 layers.
    """
    exp_dir = os.path.join(OUTPUT_ROOT, exp_key)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'#'*65}")
    print(f"# Experiment : {exp_label}")
    print(f"# Output dir : {exp_dir}")
    print(f"{'#'*65}\n")

    layer_embeddings, labels, metrics = extract_embeddings_and_metrics(
        model, dataloader, device, noise_fn=noise_fn
    )

    # ── save metrics ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}   "
          f"F1 macro: {metrics['f1_macro']:.4f}   "
          f"F1 weighted: {metrics['f1_weighted']:.4f}\n")

    # ── visualize: no_filter first, then filter ────────────────────────────────
    n_layers = len(layer_embeddings)
    for filter_clusters, subfolder in [(False, "no_filter"), (True, "filter")]:
        print(f"\n  ── {subfolder} ──")
        for layer_idx in range(n_layers):
            emb       = layer_embeddings[f"layer_{layer_idx}"]
            layer_dir = os.path.join(exp_dir, subfolder, f"layer_{layer_idx}")
            process_layer(layer_idx, emb, labels, exp_label, layer_dir, filter_clusters)

    print(f"\n  ✓  {exp_label} complete → {exp_dir}/")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("=" * 65)
    print("  Unified ViT CIFAR-10 HOLE Analysis")
    print("=" * 65)
    print(f"  Output root : {OUTPUT_ROOT}/")
    print(f"  Device      : {DEVICE}")
    print(f"  Samples     : {N_PER_CLASS} per class  ({N_PER_CLASS * 10} total)")
    print(f"  Seed        : np=42, torch=42 (noisy experiments)")
    print("=" * 65 + "\n")

    # ── load full-precision model once ────────────────────────────────────────
    print(f"Loading full-precision model from {MODEL_PATH} …")
    model_fp = ViTForImageClassification.from_pretrained(MODEL_PATH)
    model_fp = model_fp.to(DEVICE)
    model_fp.eval()
    print(f"Full-precision model ready on {DEVICE}\n")

    # ── load dataset + build dataloader (used by all experiments) ─────────────
    print("Loading CIFAR-10 test set …")
    test_ds = load_dataset("cifar10", split="test")

    proc      = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    size      = proc.size["height"]
    transform = Compose([
        Resize(size), CenterCrop(size), ToTensor(),
        Normalize(mean=proc.image_mean, std=proc.image_std),
    ])

    def transform_fn(examples):
        examples["pixel_values"] = [transform(img.convert("RGB")) for img in examples["img"]]
        return examples

    test_ds.set_transform(transform_fn)

    print(f"Selecting {N_PER_CLASS} samples/class with seed=42 …")
    indices = select_balanced_subset(test_ds, n_per_class=N_PER_CLASS, seed=42)
    subset  = Subset(test_ds, indices)
    loader  = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"Dataloader ready: {len(indices)} samples\n")

    # ── experiment 1: balanced (clean, full-precision) ────────────────────────
    run_experiment(
        exp_key="balanced",
        exp_label="Balanced – clean (FP32)",
        model=model_fp,
        dataloader=loader,
        device=DEVICE,
        noise_fn=None,
    )

    # ── experiments 2-6: noise variants ───────────────────────────────────────
    for noise_key, cfg in NOISE_CONFIGS.items():
        run_experiment(
            exp_key=noise_key,
            exp_label=cfg["label"],
            model=model_fp,
            dataloader=loader,
            device=DEVICE,
            noise_fn=cfg["fn"],
        )

    # ── experiment 7: INT8 quantized (clean) ──────────────────────────────────
    print("\nBuilding INT8 dynamically-quantized model …")
    model_q = ViTForImageClassification.from_pretrained(MODEL_PATH)
    model_q.eval()
    model_q = torch.quantization.quantize_dynamic(model_q, {nn.Linear}, dtype=torch.qint8)
    # quantized models must run on CPU
    loader_cpu = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    run_experiment(
        exp_key="quantized",
        exp_label="Quantized – clean (INT8)",
        model=model_q,
        dataloader=loader_cpu,
        device=torch.device("cpu"),
        noise_fn=None,
    )

    print(f"\n{'='*65}")
    print(f"  ✓  All experiments complete.")
    print(f"  ✓  Results saved under: {OUTPUT_ROOT}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
