"""
BERT Named Entity Recognition + HOLE Topological Analysis
==========================================================

Loads a pretrained BERT model fine-tuned for NER (token classification),
extracts per-entity-type embeddings from each encoder layer, and runs
the full HOLE visualization pipeline on those embeddings.

The script uses the CoNLL-2003 English NER dataset with BIO tags:
  O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC

For HOLE analysis we collapse BIO prefixes into entity types
(O, PER, ORG, LOC, MISC) so clusters correspond to semantic categories.

Experiments
-----------
  balanced   – clean text, full-precision model

Output layout
-------------
  bert_ner_hole_outputs/
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
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

import hole
from hole.visualization.cluster_flow import ClusterFlowAnalyzer, FlowVisualizer
from hole.visualization.persistence_vis import plot_dimensionality_reduction


# ── configuration ──────────────────────────────────────────────────────────────

MODEL_NAME = "dslim/bert-base-NER"
OUTPUT_ROOT = "bert_ner_hole_outputs"
N_SENTENCES = 150
BATCH_SIZE = 16
MAX_SEQ_LEN = 128

# CoNLL-2003 NER fine-grained label set (BIO scheme)
NER_LABELS_FINE = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}

# Collapsed entity-type labels for HOLE cluster analysis
ENTITY_TYPES = {
    0: "O",
    1: "PER",
    2: "ORG",
    3: "LOC",
    4: "MISC",
}

# Map CoNLL-2003 fine-grained → collapsed index
FINE_TO_COLLAPSED = {
    0: 0,  # O → O
    1: 1,  # B-PER → PER
    2: 1,  # I-PER → PER
    3: 2,  # B-ORG → ORG
    4: 2,  # I-ORG → ORG
    5: 3,  # B-LOC → LOC
    6: 3,  # I-LOC → LOC
    7: 4,  # B-MISC → MISC
    8: 4,  # I-MISC → MISC
}

# dslim/bert-base-NER label → collapsed entity types
# The model outputs: {0: O, 1: B-MISC, 2: I-MISC, 3: B-PER, 4: I-PER,
#                     5: B-ORG, 6: I-ORG, 7: B-LOC, 8: I-LOC}
# CoNLL-2003 ground-truth uses: {0: O, 1: B-PER, 2: I-PER, 3: B-ORG, 4: I-ORG,
#                                 5: B-LOC, 6: I-LOC, 7: B-MISC, 8: I-MISC}
MODEL_PRED_TO_COLLAPSED = {
    0: 0,  # O → O
    1: 4,  # B-MISC → MISC
    2: 4,  # I-MISC → MISC
    3: 1,  # B-PER → PER
    4: 1,  # I-PER → PER
    5: 2,  # B-ORG → ORG
    6: 2,  # I-ORG → ORG
    7: 3,  # B-LOC → LOC
    8: 3,  # I-LOC → LOC
}


# ── data helpers ───────────────────────────────────────────────────────────────


def tokenize_and_align_labels(examples, tokenizer, label_key="ner_tags", max_length=128):
    """
    Tokenize sentences and align NER labels to sub-word tokens.
    Uses the standard approach: first sub-token gets the original label,
    continuation sub-tokens get -100 (ignored in loss/metrics).
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    all_labels = []
    for i, labels in enumerate(examples[label_key]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # special tokens
            elif word_id != prev_word_id:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(-100)  # sub-word continuation
            prev_word_id = word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def collate_fn(batch):
    """Stack pre-tokenized NER fields into a batch dict."""
    return {
        "input_ids": torch.stack([torch.tensor(b["input_ids"]) for b in batch]),
        "attention_mask": torch.stack(
            [torch.tensor(b["attention_mask"]) for b in batch]
        ),
        "labels": torch.stack([torch.tensor(b["labels"]) for b in batch]),
    }


# ── embedding extraction ──────────────────────────────────────────────────────


def extract_embeddings_and_metrics(model, dataloader, device):
    """
    Forward-pass collecting:
      - Token-level embeddings from every BERT encoder layer
        (only tokens with valid labels, i.e. not -100)
      - Token-level classification metrics

    Returns:
      layer_embeddings : dict[str, np.ndarray]  – shape (N_tokens, hidden_dim) per layer
      token_labels     : np.ndarray             – collapsed entity-type indices per token
      metrics          : dict                   – accuracy, F1, etc.
    """
    model.eval()

    layer_bufs = None
    labels_buf = []
    preds_buf = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            lbl = batch["labels"]  # (batch, seq_len) with -100 for ignored

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states  # tuple of (n_layers+1,)
            logits = outputs.logits  # (batch, seq_len, n_labels)
            batch_preds = logits.argmax(dim=-1).cpu()  # (batch, seq_len)

            # Mask: only keep tokens with real labels (not -100)
            mask = lbl != -100  # (batch, seq_len)

            # Flatten and filter
            flat_labels = lbl[mask].numpy()
            flat_preds = batch_preds[mask].numpy()

            labels_buf.append(flat_labels)
            preds_buf.append(flat_preds)

            # Collect token embeddings from each encoder layer
            if layer_bufs is None:
                layer_bufs = {
                    f"layer_{i}": [] for i in range(len(hidden_states) - 1)
                }

            for i in range(1, len(hidden_states)):
                hs = hidden_states[i].cpu()  # (batch, seq_len, hidden)
                # Apply mask to extract only valid-label token embeddings
                valid_embs = hs[mask].numpy()  # (n_valid_tokens, hidden)
                layer_bufs[f"layer_{i - 1}"].append(valid_embs)

    all_labels = np.concatenate(labels_buf)
    all_preds = np.concatenate(preds_buf)

    layer_embeddings = {k: np.vstack(v) for k, v in layer_bufs.items()}

    # Collapse BIO → entity type for HOLE analysis
    collapsed_labels = np.array([FINE_TO_COLLAPSED[l] for l in all_labels])
    collapsed_preds = np.array([MODEL_PRED_TO_COLLAPSED[p] for p in all_preds])

    # Metrics on collapsed labels
    class_names = [ENTITY_TYPES[i] for i in range(len(ENTITY_TYPES))]
    report = classification_report(
        collapsed_labels, collapsed_preds,
        target_names=class_names, output_dict=True, zero_division=0,
    )

    metrics = {
        "accuracy_fine": float(accuracy_score(all_labels, all_preds)),
        "accuracy_collapsed": float(accuracy_score(collapsed_labels, collapsed_preds)),
        "f1_macro": float(
            f1_score(collapsed_labels, collapsed_preds, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(collapsed_labels, collapsed_preds, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(
                collapsed_labels, collapsed_preds, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                collapsed_labels, collapsed_preds, average="macro", zero_division=0
            )
        ),
        "per_class": {
            cls: {k: round(float(v), 4) for k, v in vals.items() if k != "support"}
            for cls, vals in report.items()
            if cls in class_names
        },
    }

    return layer_embeddings, collapsed_labels, metrics


# ── sub-sample tokens for HOLE (embeddings can be huge) ───────────────────────


def subsample_tokens(embeddings, labels, max_tokens=500, seed=42):
    """
    If there are more tokens than `max_tokens`, sample a balanced subset
    so HOLE distance-matrix computation stays tractable.
    """
    n = len(labels)
    if n <= max_tokens:
        return embeddings, labels

    np.random.seed(seed)
    unique_labels = np.unique(labels)
    per_label = max(max_tokens // len(unique_labels), 5)

    indices = []
    for lbl in unique_labels:
        lbl_idx = np.where(labels == lbl)[0]
        n_pick = min(per_label, len(lbl_idx))
        indices.extend(np.random.choice(lbl_idx, n_pick, replace=False))

    indices = np.array(indices)
    return embeddings[indices], labels[indices]


# ── per-layer HOLE visualizations ─────────────────────────────────────────────


def process_layer(
    layer_key: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    exp_label: str,
    layer_dir: str,
    filter_clusters: bool,
):
    """Run the full HOLE visualization suite on one layer's token embeddings."""
    os.makedirs(layer_dir, exist_ok=True)
    os.makedirs(f"{layer_dir}/core", exist_ok=True)

    # Sub-sample so distance matrices stay manageable
    emb, lbl = subsample_tokens(embeddings, labels, max_tokens=75)

    filter_tag = "Filter" if filter_clusters else "No Filter"

    visualizer = hole.HOLEVisualizer(
        point_cloud=emb, distance_metric="cosine"
    )

    # 1. PCA visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_dimensionality_reduction(
        emb,
        method="pca",
        labels=lbl,
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
        lbl,
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
        emb,
        lbl,
        mid_t,
        save_path=f"{layer_dir}/blob_visualization.png",
        title=f"{layer_key} – Blob ({exp_label}, {filter_tag}, t={mid_t:.3f})",
        metric="cosine",
    )
    plt.close(fig)

    # 5. Sankey flow + stacked bar evolution
    fv = FlowVisualizer(figsize=(18, 10), class_names=ENTITY_TYPES)

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

    print(f"      {layer_key}: done  ({len(emb)} tokens)  →  {layer_dir}/")


# ── experiment runner ──────────────────────────────────────────────────────────


def run_experiment(
    exp_key: str,
    exp_label: str,
    model,
    dataloader,
    device: torch.device,
):
    """Run one complete NER experiment: extract embeddings, metrics, generate HOLE visualizations."""
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
        f"  Accuracy (collapsed): {metrics['accuracy_collapsed']:.4f}   "
        f"F1 macro: {metrics['f1_macro']:.4f}   "
        f"F1 weighted: {metrics['f1_weighted']:.4f}"
    )

    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Token distribution: { {ENTITY_TYPES[u]: int(c) for u, c in zip(unique, counts)} }\n")

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
    print("  BERT NER Classification – HOLE Analysis")
    print("=" * 65)
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Output root : {OUTPUT_ROOT}/")
    print(f"  Device      : {DEVICE}")
    print(f"  Sentences   : {N_SENTENCES}")
    print(f"  Max seq len : {MAX_SEQ_LEN}")
    print("=" * 65 + "\n")

    # ── load model + tokenizer ────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}\n")

    # ── load CoNLL-2003 English NER dataset ────────────────────────────────
    print("Loading CoNLL-2003 (English) NER dataset …")
    dataset = load_dataset("conll2003", split="test", trust_remote_code=True)

    # Take a subset of sentences
    np.random.seed(42)
    n_total = len(dataset)
    n_pick = min(N_SENTENCES, n_total)
    sent_indices = np.random.choice(n_total, n_pick, replace=False)
    sent_indices.sort()
    subset = dataset.select(sent_indices.tolist())
    print(f"Selected {n_pick} sentences from test split\n")

    # Tokenize with sub-word alignment
    print("Tokenizing with sub-word label alignment …")
    tokenized = tokenize_and_align_labels(
        subset, tokenizer, label_key="ner_tags", max_length=MAX_SEQ_LEN,
    )

    # Build list of dicts for the dataloader
    data_list = []
    for i in range(len(subset)):
        data_list.append(
            {
                "input_ids": tokenized["input_ids"][i],
                "attention_mask": tokenized["attention_mask"][i],
                "labels": tokenized["labels"][i],
            }
        )

    loader = DataLoader(
        data_list, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    print(f"Dataloader ready: {len(data_list)} sentences\n")

    # ── run experiment ────────────────────────────────────────────────────────
    run_experiment(
        exp_key="balanced",
        exp_label="Balanced – NER (FP32)",
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
