"""
BERT Sentiment Classification + HOLE Topological Analysis
==========================================================

Loads a pretrained BERT model fine-tuned for sentiment analysis,
extracts hidden-state embeddings from each encoder layer, and runs
the full HOLE visualization pipeline on those embeddings.

Experiments
-----------
  balanced   – clean text, full-precision model

Every experiment produces BOTH filter/ and no_filter/ sub-trees.

Output layout
-------------
  bert_sentiment_hole_outputs/
    balanced/
      metrics.json
      filter/   layer_0/ … layer_11/
      no_filter/ layer_0/ … layer_11/

Reproducibility
---------------
  np.random.seed(42)
  torch.manual_seed(42)
"""

import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import hole
from hole.visualization.cluster_flow import ClusterFlowAnalyzer, FlowVisualizer
from hole.visualization.persistence_vis import plot_dimensionality_reduction


# ── configuration ──────────────────────────────────────────────────────────────

MODEL_NAME = "textattack/bert-base-uncased-SST-2"
OUTPUT_ROOT = "bert_sentiment_hole_outputs"
N_PER_CLASS = 200
BATCH_SIZE = 32
MAX_SEQ_LEN = 128

SENTIMENT_CLASSES = {
    0: "negative",
    1: "positive",
}


# ── data helpers ───────────────────────────────────────────────────────────────


def select_balanced_subset(
    dataset, label_key: str = "label", n_classes: int = 2,
    n_per_class: int = 40, seed: int = 42,
) -> np.ndarray:
    """Pick a balanced subset with `n_per_class` samples per label."""
    all_labels = np.array([dataset[i][label_key] for i in range(len(dataset))])
    np.random.seed(seed)
    indices = []
    for c in range(n_classes):
        cls_idx = np.where(all_labels == c)[0]
        n_pick = min(n_per_class, len(cls_idx))
        indices.extend(np.random.choice(cls_idx, n_pick, replace=False))
    return np.array(indices)


def collate_fn(batch):
    """Stack pre-tokenized fields into a batch dict."""
    return {
        "input_ids": torch.stack([torch.tensor(b["input_ids"]) for b in batch]),
        "attention_mask": torch.stack(
            [torch.tensor(b["attention_mask"]) for b in batch]
        ),
        "labels": torch.tensor([b["label"] for b in batch]),
    }


# ── embedding extraction ──────────────────────────────────────────────────────


def extract_embeddings_and_metrics(model, dataloader, device, n_classes: int = 2):
    """
    Forward-pass through the model collecting:
      - CLS-token embeddings from every BERT encoder layer
      - Classification metrics (accuracy, F1, precision, recall)
    """
    model.eval()

    layer_bufs = None
    labels_buf = []
    preds_buf = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            lbl = batch["labels"].cpu().numpy()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

            # outputs.hidden_states is a tuple of (n_layers + 1,) tensors
            # each of shape (batch, seq_len, hidden_dim).
            # Index 0 = embedding layer output, 1..12 = encoder layers.
            hidden_states = outputs.hidden_states

            preds_buf.append(outputs.logits.argmax(dim=-1).cpu().numpy())
            labels_buf.append(lbl)

            # Collect CLS token (position 0) from each encoder layer
            if layer_bufs is None:
                layer_bufs = {
                    f"layer_{i}": [] for i in range(len(hidden_states) - 1)
                }

            for i in range(1, len(hidden_states)):
                cls_emb = hidden_states[i][:, 0, :].cpu().numpy()
                layer_bufs[f"layer_{i - 1}"].append(cls_emb)

    labels = np.concatenate(labels_buf)
    preds = np.concatenate(preds_buf)

    layer_embeddings = {k: np.vstack(v) for k, v in layer_bufs.items()}

    class_names = [SENTIMENT_CLASSES[i] for i in range(n_classes)]
    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True
    )

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "f1_micro": float(f1_score(labels, preds, average="micro")),
        "precision_macro": float(precision_score(labels, preds, average="macro")),
        "recall_macro": float(recall_score(labels, preds, average="macro")),
        "per_class": {
            cls: {k: round(float(v), 4) for k, v in vals.items() if k != "support"}
            for cls, vals in report.items()
            if cls in class_names
        },
    }

    return layer_embeddings, labels, metrics


# ── per-layer HOLE visualizations ─────────────────────────────────────────────


def process_layer(
    layer_key: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    exp_label: str,
    layer_dir: str,
    filter_clusters: bool,
):
    """Run the full HOLE visualization suite on one layer's CLS embeddings."""
    os.makedirs(layer_dir, exist_ok=True)
    os.makedirs(f"{layer_dir}/core", exist_ok=True)

    filter_tag = "Filter" if filter_clusters else "No Filter"

    visualizer = hole.HOLEVisualizer(
        point_cloud=embeddings, distance_metric="cosine"
    )

    # 1. PCA visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_dimensionality_reduction(
        embeddings,
        method="pca",
        labels=labels,
        ax=ax,
        title=f"{layer_key} – PCA ({exp_label}, {filter_tag})",
        point_size=50,
        alpha=0.7,
    )
    plt.savefig(f"{layer_dir}/pca_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Heatmap dendrogram
    hv = visualizer.get_persistence_dendrogram_visualizer(
        distance_matrix=visualizer.distance_matrix
    )
    hv.compute_persistence()
    hv.plot_dendrogram_with_heatmap(figsize=(16, 8), cmap="gray")
    plt.savefig(f"{layer_dir}/heatmap_dendrogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Cluster evolution analysis
    analyzer = ClusterFlowAnalyzer(visualizer.distance_matrix, max_thresholds=8)
    clust_evo = analyzer.compute_cluster_evolution(
        labels,
        filter_small_clusters=filter_clusters,
        min_cluster_size=5,
        metric_name="Cosine",
    )

    thresholds = sorted(
        [float(t) for t in clust_evo["labels_"]["Cosine"].keys()]
    )
    mid_t = thresholds[1] if len(thresholds) > 1 else thresholds[0]

    # 4. Blob visualization
    bv = visualizer.get_blob_visualizer(
        figsize=(10, 8), outlier_percentage=0.0, show_contours=False
    )
    fig = bv.plot_pca_with_cluster_hulls(
        embeddings,
        labels,
        mid_t,
        save_path=f"{layer_dir}/blob_visualization.png",
        title=f"{layer_key} – Blob ({exp_label}, {filter_tag}, t={mid_t:.3f})",
        metric="cosine",
    )
    plt.close(fig)

    # 5. Sankey flow + stacked bar evolution
    fv = FlowVisualizer(figsize=(18, 10), class_names=SENTIMENT_CLASSES)

    plt.close(
        fv.plot_sankey_flow(
            clust_evo,
            save_path=f"{layer_dir}/sankey_flow.png",
            title=f"{layer_key} – DN-Cosine ({exp_label}, {filter_tag})",
            show_true_labels_text=False,
            show_filtration_text=False,
        )
    )

    plt.close(
        fv.plot_stacked_bar_evolution(
            clust_evo,
            save_path=f"{layer_dir}/stacked_bars.png",
            title=f"{layer_key} – DN-Cosine ({exp_label}, {filter_tag})",
            show_true_labels_text=False,
            show_filtration_text=False,
        )
    )

    # 6. Persistence diagram + barcode
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    visualizer.plot_persistence_diagram(
        ax=ax1,
        title=f"{layer_key} – Persistence Diagram ({exp_label})",
        pts=20,
    )
    plt.tight_layout()
    fig1.savefig(
        f"{layer_dir}/persistence_diagram.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    visualizer.plot_persistence_barcode(
        ax=ax2,
        title=f"{layer_key} – Persistence Barcode ({exp_label})",
        pts=20,
    )
    plt.tight_layout()
    fig2.savefig(
        f"{layer_dir}/persistence_barcode.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig2)

    print(f"      {layer_key}: done  →  {layer_dir}/")


# ── experiment runner ──────────────────────────────────────────────────────────


def run_experiment(
    exp_key: str,
    exp_label: str,
    model,
    dataloader,
    device: torch.device,
):
    """Run one complete experiment: extract embeddings, compute metrics, generate HOLE visualizations."""
    exp_dir = os.path.join(OUTPUT_ROOT, exp_key)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'#' * 65}")
    print(f"# Experiment : {exp_label}")
    print(f"# Output dir : {exp_dir}")
    print(f"{'#' * 65}\n")

    layer_embeddings, labels, metrics = extract_embeddings_and_metrics(
        model, dataloader, device
    )

    metrics_path = os.path.join(exp_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")
    print(
        f"  Accuracy: {metrics['accuracy']:.4f}   "
        f"F1 macro: {metrics['f1_macro']:.4f}   "
        f"F1 weighted: {metrics['f1_weighted']:.4f}\n"
    )

    for filter_clusters, subfolder in [(False, "no_filter"), (True, "filter")]:
        print(f"\n  ── {subfolder} ──")
        for layer_key, emb in layer_embeddings.items():
            layer_dir = os.path.join(exp_dir, subfolder, layer_key)
            process_layer(
                layer_key, emb, labels, exp_label, layer_dir, filter_clusters
            )

    print(f"\n  ✓  {exp_label} complete → {exp_dir}/")


# ── main ───────────────────────────────────────────────────────────────────────


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("=" * 65)
    print("  BERT Sentiment Classification – HOLE Analysis")
    print("=" * 65)
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Output root : {OUTPUT_ROOT}/")
    print(f"  Device      : {DEVICE}")
    print(f"  Samples     : {N_PER_CLASS} per class  ({N_PER_CLASS * 2} total)")
    print(f"  Max seq len : {MAX_SEQ_LEN}")
    print("=" * 65 + "\n")

    # ── load model + tokenizer ────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}\n")

    # ── load and tokenize dataset ─────────────────────────────────────────────
    # SST-2 has 2 classes (negative=0, positive=1)
    print("Loading SST-2 dataset …")
    dataset = load_dataset("glue", "sst2", split="validation")

    print(f"Selecting {N_PER_CLASS} samples/class with seed=42 …")
    indices = select_balanced_subset(
        dataset, label_key="label", n_classes=2,
        n_per_class=N_PER_CLASS, seed=42,
    )
    subset = Subset(dataset, indices.tolist())

    # Pre-tokenize the subset
    def tokenize_item(item):
        enc = tokenizer(
            item["sentence"],
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        return {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "label": item["label"],
        }

    tokenized = [tokenize_item(subset[i]) for i in range(len(subset))]

    loader = DataLoader(
        tokenized, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    print(f"Dataloader ready: {len(tokenized)} samples\n")

    # ── run experiment ────────────────────────────────────────────────────────
    run_experiment(
        exp_key="balanced",
        exp_label="Balanced – Sentiment (FP32)",
        model=model,
        dataloader=loader,
        device=DEVICE,
    )

    print(f"\n{'=' * 65}")
    print(f"  ✓  All experiments complete.")
    print(f"  ✓  Results saved under: {OUTPUT_ROOT}/")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
