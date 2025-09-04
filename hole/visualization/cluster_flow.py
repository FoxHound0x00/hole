"""
Flow Visualization for Persistent Homology Cluster Evolution

This module provides Sankey diagrams and stacked bar charts to show how clusters
evolve through different death thresholds in persistent homology filtration.
Based on the reference ComponentEvolutionVisualizer implementation.
"""

import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import gudhi as gd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch
from tqdm import tqdm

# Import distance_matrix from core
from ..core.distance_metrics import distance_matrix


class ClusterFlowAnalyzer:
    """Analyzes cluster evolution through persistent homology filtration."""

    def __init__(self, distance_matrix: np.ndarray, max_thresholds: int = 8):
        """
        Initialize with distance matrix.

        Args:
            distance_matrix: 2D symmetric distance matrix
            max_thresholds: Maximum number of thresholds to analyze
        """
        self.distance_matrix = distance_matrix
        self.n_points = distance_matrix.shape[0]
        self.max_thresholds = max_thresholds

        # Will be computed
        self.persistence = None
        self.death_thresholds = None
        self.cluster_evolution = None

    def compute_cluster_evolution(
        self, true_labels: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute cluster evolution through different death thresholds.
        Returns data in the format expected by ComponentEvolutionVisualizer.

        Args:
            true_labels: Optional true labels for comparison

        Returns:
            Dictionary containing components_ and labels_ in the expected format
        """
        print("Computing persistent homology...")

        # Create Rips complex and compute persistence
        rips_complex = gd.RipsComplex(distance_matrix=self.distance_matrix)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
        self.persistence = simplex_tree.persistence()

        # Extract death thresholds for 0-dimensional features (connected components)
        death_thresholds = []
        for dim, (birth, death) in self.persistence:
            if dim == 0 and death != float("inf"):
                death_thresholds.append(death)

        # Sort all thresholds
        all_thresholds = sorted(set(death_thresholds))
        print(f"Found {len(all_thresholds)} total death thresholds")

        # Select 4 specific thresholds for meaningful 5-stage evolution
        selected_thresholds = self._select_meaningful_thresholds(
            all_thresholds, true_labels
        )

        print(f"Selected thresholds: {[f'{t:.4f}' for t in selected_thresholds]}")

        # Initialize components_ and labels_ dictionaries
        components_ = {"Euclidean": {}}
        labels_ = {"Euclidean": {}}

        for threshold in tqdm(selected_thresholds, desc="Processing thresholds"):
            print(f"  Processing threshold: {threshold:.4f}")

            # Create adjacency matrix for this threshold
            adj_matrix = (self.distance_matrix <= threshold).astype(int)
            np.fill_diagonal(adj_matrix, 0)  # Remove self-loops

            # Find connected components using NetworkX
            graph = nx.from_numpy_array(adj_matrix)
            components = list(nx.connected_components(graph))

            # Create cluster labels
            cluster_labels = np.zeros(self.n_points, dtype=int)
            for cluster_id, component in enumerate(components):
                for node in component:
                    cluster_labels[node] = cluster_id

            # Store in the expected format
            components_["Euclidean"][str(threshold)] = len(components)
            labels_["Euclidean"][str(threshold)] = cluster_labels

        return {
            "components_": components_,
            "labels_": labels_,
            "true_labels": true_labels,
        }

    def _select_meaningful_thresholds(
        self, all_thresholds: List[float], true_labels: Optional[np.ndarray] = None
    ) -> List[float]:
        """
        Select 4 meaningful thresholds for 5-stage visualization:
        Stage 1: True labels (not a threshold)
        Stage 2: Initial clusters (very small threshold - many clusters)
        Stage 3: Clusters similar to true labels (threshold where clusters roughly match CIFAR-10)
        Stage 4: Intermediate merging (between similar and final)
        Stage 5: Final single cluster
        """
        if len(all_thresholds) < 4:
            print("Warning: Not enough thresholds, using all available")
            return all_thresholds

        # Stage 2: Initial clusters - use smallest threshold (many small clusters)
        initial_threshold = all_thresholds[0]

        # Stage 5: Final cluster - find threshold where we get 1 cluster
        final_threshold = None
        for threshold in reversed(all_thresholds):
            # Test this threshold
            adj_matrix = (self.distance_matrix <= threshold).astype(int)
            np.fill_diagonal(adj_matrix, 0)
            graph = nx.from_numpy_array(adj_matrix)
            n_components = nx.number_connected_components(graph)
            if n_components == 1:
                final_threshold = threshold
                break

        if final_threshold is None:
            final_threshold = all_thresholds[-1]

        # Stage 3: Find threshold where clusters are most similar to true labels
        similar_threshold = self._find_similar_to_true_labels(
            all_thresholds, true_labels
        )

        # Stage 4: Intermediate threshold - between similar and final
        intermediate_candidates = [
            t for t in all_thresholds if similar_threshold < t < final_threshold
        ]
        if intermediate_candidates:
            # Pick middle of the candidates
            intermediate_threshold = intermediate_candidates[
                len(intermediate_candidates) // 2
            ]
        else:
            # Fallback: pick something between similar and final
            intermediate_threshold = (similar_threshold + final_threshold) / 2
            # Find closest actual threshold
            intermediate_threshold = min(
                all_thresholds, key=lambda x: abs(x - intermediate_threshold)
            )

        selected = [
            initial_threshold,
            similar_threshold,
            intermediate_threshold,
            final_threshold,
        ]

        # Remove duplicates and sort
        selected = sorted(list(set(selected)))

        print("Selected thresholds breakdown:")
        print(f"  Initial (many clusters): {selected[0]:.4f}")
        if len(selected) > 1:
            print(f"  Similar to true labels: {selected[1]:.4f}")
        if len(selected) > 2:
            print(f"  Intermediate merging: {selected[2]:.4f}")
        if len(selected) > 3:
            print(f"  Final single cluster: {selected[3]:.4f}")

        return selected

    def _find_similar_to_true_labels(
        self, all_thresholds: List[float], true_labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Find threshold where clusters best match true labels - where data points
        are grouped together with their original class labels (with some outliers).
        Uses clustering purity/homogeneity to find best match.
        """
        if true_labels is None:
            # Fallback: use threshold that gives ~10 clusters
            target_clusters = 10
            best_threshold = all_thresholds[len(all_thresholds) // 3]
            best_score = float("inf")

            for threshold in all_thresholds:
                adj_matrix = (self.distance_matrix <= threshold).astype(int)
                np.fill_diagonal(adj_matrix, 0)
                graph = nx.from_numpy_array(adj_matrix)
                n_components = nx.number_connected_components(graph)

                score = abs(n_components - target_clusters)
                if score < best_score:
                    best_score = score
                    best_threshold = threshold

            return best_threshold

        print("Finding threshold where data points cluster with their true labels...")

        best_threshold = all_thresholds[len(all_thresholds) // 3]  # Default fallback
        best_score = 0.0  # We want to maximize clustering quality

        # Test thresholds to find one where clusters best match true labels
        for threshold in all_thresholds:
            adj_matrix = (self.distance_matrix <= threshold).astype(int)
            np.fill_diagonal(adj_matrix, 0)
            graph = nx.from_numpy_array(adj_matrix)
            components = list(nx.connected_components(graph))

            # Create cluster labels
            cluster_labels = np.zeros(self.n_points, dtype=int)
            for cluster_id, component in enumerate(components):
                for node in component:
                    cluster_labels[node] = cluster_id

            # Calculate clustering quality metrics
            purity_score = self._calculate_purity(true_labels, cluster_labels)
            homogeneity_score = self._calculate_homogeneity(true_labels, cluster_labels)

            # Combined score: prioritize high purity and reasonable homogeneity
            # Purity measures how "pure" each cluster is (same true label)
            # Homogeneity measures how well each true class is in one cluster
            combined_score = 0.7 * purity_score + 0.3 * homogeneity_score

            print(
                f"    Threshold {threshold:.4f}: {len(components)} clusters, purity={purity_score:.3f}, homogeneity={homogeneity_score:.3f}, combined={combined_score:.3f}"
            )

            if combined_score > best_score:
                best_score = combined_score
                best_threshold = threshold

        print(f"    Best threshold: {best_threshold:.4f} (score: {best_score:.3f})")
        return best_threshold

    def _calculate_purity(
        self, true_labels: np.ndarray, cluster_labels: np.ndarray
    ) -> float:
        """
        Calculate clustering purity: for each cluster, what fraction belongs to the most common true class.
        High purity means clusters contain mostly points from the same true class.
        """
        if len(true_labels) != len(cluster_labels):
            return 0.0

        total_correct = 0
        total_points = len(true_labels)

        # For each cluster, find the most common true label
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            # Get all points in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]

            if len(cluster_true_labels) > 0:
                # Find most common true label in this cluster
                unique_labels, counts = np.unique(
                    cluster_true_labels, return_counts=True
                )
                max_count = np.max(counts)
                total_correct += max_count

        purity = total_correct / total_points if total_points > 0 else 0.0
        return purity

    def _calculate_homogeneity(
        self, true_labels: np.ndarray, cluster_labels: np.ndarray
    ) -> float:
        """
        Calculate clustering homogeneity: for each true class, what fraction is in the most common cluster.
        High homogeneity means each true class is mostly in one cluster.
        """
        if len(true_labels) != len(cluster_labels):
            return 0.0

        total_correct = 0
        total_points = len(true_labels)

        # For each true class, find the most common cluster
        unique_true_labels = np.unique(true_labels)

        for true_label in unique_true_labels:
            # Get all points with this true label
            true_mask = true_labels == true_label
            true_cluster_labels = cluster_labels[true_mask]

            if len(true_cluster_labels) > 0:
                # Find most common cluster for this true label
                unique_clusters, counts = np.unique(
                    true_cluster_labels, return_counts=True
                )
                max_count = np.max(counts)
                total_correct += max_count

        homogeneity = total_correct / total_points if total_points > 0 else 0.0
        return homogeneity


class ComponentEvolutionVisualizer:
    """
    A class for visualizing component evolution through death thresholds.
    Based on the reference implementation from the user's Jupyter notebook.
    """

    def __init__(self, components_, labels_, class_names=None):
        """
        Initialize the component evolution visualizer.

        Args:
            components_: Dictionary of components at each threshold
            labels_: Dictionary of labels at each threshold
            class_names: Optional dictionary mapping class indices to names
        """
        self.components_ = components_
        self.labels_ = labels_
        # Initialize with None - will be created when needed
        self.color_mapping = None

        # Default class names if none provided
        self.class_names = class_names or {
            0: "Cluster_0",
            1: "Cluster_1",
            2: "Cluster_2",
            3: "Cluster_3",
            4: "Cluster_4",
            5: "Cluster_5",
            6: "Cluster_6",
            7: "Cluster_7",
            8: "Cluster_8",
            9: "Cluster_9",
        }

    def _generate_distinct_colors(self, n_colors, original_labels=None):
        """Generate n visually distinct colors using discrete colormaps."""
        colors = []

        # Use discrete colormaps for better distinction
        discrete_colormaps = [
            plt.cm.tab10,  # 10 distinct colors
            plt.cm.tab20,  # 20 distinct colors
            plt.cm.tab20b,  # 20 more distinct colors
            plt.cm.tab20c,  # 20 more distinct colors
            plt.cm.Set1,  # 9 distinct colors
            plt.cm.Set2,  # 8 distinct colors
            plt.cm.Set3,  # 12 distinct colors
            plt.cm.Pastel1,  # 9 pastel colors
            plt.cm.Pastel2,  # 8 pastel colors
            plt.cm.Dark2,  # 8 dark colors
            plt.cm.Paired,  # 12 paired colors
        ]

        # Special handling for noise/unclustered points (-1)
        # noise_color = (0.5, 0.5, 0.5, 1.0)  # Gray color for noise

        # If we have original labels, assign colors for them first
        if original_labels is not None:
            # Get unique original labels (excluding noise if present)
            unique_original = sorted(
                [label for label in set(original_labels) if label != -1]
            )

            # Assign colors from tab10 for original labels
            if len(unique_original) <= 10:
                base_cmap = plt.cm.tab10
                for i, label in enumerate(unique_original):
                    colors.append(base_cmap(i))
            else:
                # Use tab20 for more labels
                base_cmap = plt.cm.tab20
                for i, label in enumerate(unique_original):
                    colors.append(base_cmap(i % 20))

        # Fill remaining colors from discrete colormaps
        cmap_idx = 0

        while len(colors) < n_colors and cmap_idx < len(discrete_colormaps):
            cmap = discrete_colormaps[cmap_idx]

            # Get the number of colors in this colormap
            if hasattr(cmap, "N"):
                n_cmap_colors = cmap.N
            else:
                n_cmap_colors = 256  # Default for continuous maps used discretely

            # Sample discrete colors from the colormap
            for i in range(min(20, n_cmap_colors)):  # Limit to 20 colors per map
                if len(colors) >= n_colors:
                    break

                color = cmap(i / max(1, min(20, n_cmap_colors) - 1))

                # Ensure color is sufficiently different from existing colors
                is_unique = True
                for existing_color in colors:
                    # Calculate color distance in RGB space
                    dist = np.sqrt(
                        sum((color[j] - existing_color[j]) ** 2 for j in range(3))
                    )
                    if dist < 0.15:  # Minimum color distance threshold
                        is_unique = False
                        break

                if is_unique:
                    colors.append(color)

            cmap_idx += 1

        # If we still need more colors, use HSV generation as fallback
        if len(colors) < n_colors:
            golden_ratio = 0.618033988749895
            for i in range(len(colors), n_colors):
                # Use golden ratio for good color distribution
                hue = (i * golden_ratio) % 1.0
                saturation = 0.7 + (i % 3) * 0.1  # 0.7, 0.8, 0.9
                value = 0.8 + (i % 2) * 0.1  # 0.8, 0.9

                # Convert HSV to RGB
                rgb = mcolors.hsv_to_rgb([hue, saturation, value])
                rgba = (*rgb, 1.0)
                colors.append(rgba)

        return colors[:n_colors]

    def _create_color_mapping(self, key, thresholds, original_labels=None):
        """Create a consistent color mapping for all components across all thresholds."""
        all_component_ids = set()

        # Collect all component IDs from original labels
        if original_labels is not None:
            all_component_ids.update(original_labels)

        # Collect all component IDs from all thresholds
        for threshold in thresholds:
            threshold_str = str(threshold)
            if threshold_str in self.labels_[key]:
                all_component_ids.update(self.labels_[key][threshold_str])

        # Sort component IDs for consistent ordering, putting -1 (noise) first if present
        sorted_components = sorted(all_component_ids)
        if -1 in sorted_components:
            sorted_components.remove(-1)
            sorted_components = [-1] + sorted_components

        n_components = len(sorted_components)

        print(f"Creating color mapping for {n_components} components")

        # Generate distinct colors
        distinct_colors = self._generate_distinct_colors(
            n_components, set(original_labels) if original_labels is not None else None
        )

        # Create mapping from component ID to unique color
        color_mapping = {}
        for i, comp_id in enumerate(sorted_components):
            if comp_id == -1:
                # Special gray color for noise/unclustered points
                color_mapping[comp_id] = (0.5, 0.5, 0.5, 1.0)
            else:
                # Use the generated distinct colors for regular clusters
                color_idx = i if -1 not in sorted_components else i - 1
                if color_idx < len(distinct_colors):
                    color_mapping[comp_id] = distinct_colors[color_idx]
                else:
                    # Fallback to a default color if we run out
                    color_mapping[comp_id] = (0.3, 0.3, 0.3, 1.0)

        return color_mapping

    def plot_sankey(
        self,
        key,
        original_labels=None,
        ax=None,
        title=None,
        gray_second_layer=True,
        show_true_labels_text=True,
        show_filtration_text=True,
    ):
        """
        Create a 5-stage Sankey diagram:
        Stage 1: True Labels
        Stage 2: Initial PH clusters (many small clusters)
        Stage 3: Clusters similar to true labels
        Stage 4: Intermediate merging
        Stage 5: Final single cluster
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 10))

        if key not in self.components_ or key not in self.labels_:
            ax.text(
                0.5,
                0.5,
                f"No data found for key: {key}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Check if we have original labels
        has_original = original_labels is not None
        if not has_original:
            ax.text(
                0.5,
                0.5,
                "No true labels provided for 5-stage visualization",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Get the 4 thresholds (for stages 2-5)
        thresholds = sorted([float(t) for t in self.components_[key].keys()])

        if len(thresholds) < 4:
            ax.text(
                0.5,
                0.5,
                f"Need 4 thresholds for 5-stage visualization, got {len(thresholds)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Create consistent color mapping
        self.color_mapping = self._create_color_mapping(
            key, thresholds, original_labels
        )

        # Define the 5 stages with actual threshold values
        stage_names = ["True Labels"] + [f"{t:.4f}" for t in thresholds]
        n_stages = 5

        # Calculate positions for 5 stages
        x_positions = [0.1 + i * 0.2 for i in range(n_stages)]

        # Track node positions for each stage
        node_positions = {}
        flows = []

        # STAGE 1: True labels
        print("Creating Stage 1: True labels")
        original_counts = Counter(original_labels)
        total_points = len(original_labels)

        y_start, y_end = 0.1, 0.9
        total_height = y_end - y_start
        current_y = y_start
        node_positions[0] = {}

        for comp_id in sorted(original_counts.keys()):
            count = original_counts[comp_id]
            height = (count / total_points) * total_height

            node_positions[0][comp_id] = {
                "x": x_positions[0],
                "y": current_y + height / 2,
                "height": height,
                "count": count,
                "color": self.color_mapping[comp_id],
                "y_start": current_y,
                "y_end": current_y + height,
            }
            current_y += height

        # STAGES 2-5: Process each threshold
        for stage_idx, threshold in enumerate(thresholds):
            actual_stage = stage_idx + 1  # Stages 1-4 (index 1-4)
            threshold_str = str(threshold)

            if threshold_str not in self.labels_[key]:
                continue

            print(f"Creating Stage {actual_stage + 1}: Threshold {threshold:.4f}")

            labels_at_threshold = self.labels_[key][threshold_str]
            component_counts = Counter(labels_at_threshold)

            # Calculate positions for this stage
            current_y = y_start
            node_positions[actual_stage] = {}

            for comp_id in sorted(component_counts.keys()):
                count = component_counts[comp_id]
                height = (count / total_points) * total_height

                # Use gray for second layer (first filtration stage) if requested
                if (
                    gray_second_layer and actual_stage == 1
                ):  # Second layer (first threshold)
                    color = (0.7, 0.7, 0.7, 1.0)  # Light gray
                else:
                    color = self.color_mapping[comp_id]

                node_positions[actual_stage][comp_id] = {
                    "x": x_positions[actual_stage],
                    "y": current_y + height / 2,
                    "height": height,
                    "count": count,
                    "color": color,
                    "y_start": current_y,
                    "y_end": current_y + height,
                }
                current_y += height

        # Calculate flows between consecutive stages
        for stage in range(n_stages - 1):
            from_stage = stage
            to_stage = stage + 1

            # Get labels for both stages
            if from_stage == 0:
                labels1 = original_labels
            else:
                threshold_idx = from_stage - 1
                if threshold_idx < len(thresholds):
                    labels1 = self.labels_[key][str(thresholds[threshold_idx])]
                else:
                    continue

            threshold_idx = to_stage - 1
            if threshold_idx < len(thresholds):
                labels2 = self.labels_[key][str(thresholds[threshold_idx])]
            else:
                continue

            # Calculate flows
            flow_mapping = defaultdict(lambda: defaultdict(int))
            for point_idx, (comp1, comp2) in enumerate(zip(labels1, labels2)):
                flow_mapping[comp1][comp2] += 1

            # Create flow objects
            for comp1, comp2_dict in flow_mapping.items():
                for comp2, count in comp2_dict.items():
                    if (
                        comp1 in node_positions[from_stage]
                        and comp2 in node_positions[to_stage]
                    ):
                        flows.append(
                            {
                                "from_stage": from_stage,
                                "to_stage": to_stage,
                                "from_comp": comp1,
                                "to_comp": comp2,
                                "count": count,
                                "from_node": node_positions[from_stage][comp1],
                                "to_node": node_positions[to_stage][comp2],
                            }
                        )

        # Draw flows (behind nodes)
        max_flow = max([f["count"] for f in flows]) if flows else 1

        for flow in flows:
            thickness = max(0.003, min(0.04, (flow["count"] / max_flow) * 0.06))

            from_node = flow["from_node"]
            to_node = flow["to_node"]

            # Connection points
            x1 = from_node["x"] + 0.01
            y1 = from_node["y"]
            x2 = to_node["x"] - 0.01
            y2 = to_node["y"]

            # Create smooth bezier curve
            control_distance = (x2 - x1) * 0.3
            cx1 = x1 + control_distance
            cx2 = x2 - control_distance

            # Generate curve points
            n_points = 20
            t_values = np.linspace(0, 1, n_points)

            curve_x, curve_y = [], []
            for t in t_values:
                bx = (
                    (1 - t) ** 3 * x1
                    + 3 * (1 - t) ** 2 * t * cx1
                    + 3 * (1 - t) * t**2 * cx2
                    + t**3 * x2
                )
                by = (
                    (1 - t) ** 3 * y1
                    + 3 * (1 - t) ** 2 * t * y1
                    + 3 * (1 - t) * t**2 * y2
                    + t**3 * y2
                )
                curve_x.append(bx)
                curve_y.append(by)

            # Create flow polygon
            upper_y = [y + thickness / 2 for y in curve_y]
            lower_y = [y - thickness / 2 for y in curve_y]

            flow_x = curve_x + curve_x[::-1]
            flow_y = upper_y + lower_y[::-1]

            ax.fill(
                flow_x,
                flow_y,
                color=from_node["color"],
                alpha=0.6,
                edgecolor="none",
                zorder=1,
            )

        # Draw nodes (on top of flows)
        for stage_idx in range(n_stages):
            if stage_idx not in node_positions:
                continue

            for comp_id, node in node_positions[stage_idx].items():
                # Draw node rectangle - make wider for better text visibility
                rect = FancyBboxPatch(
                    (node["x"] - 0.012, node["y_start"]),
                    0.024,
                    node["height"],
                    boxstyle="round,pad=0.001",
                    facecolor=node["color"],
                    edgecolor="black",
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=2,
                )
                ax.add_patch(rect)

                # Add labels based on flags
                if (
                    node["height"] > 0.015
                ):  # Slightly lower threshold for better visibility
                    show_text = False
                    if stage_idx == 0:
                        # Stage 1: True labels - controlled by show_true_labels_text
                        show_text = show_true_labels_text
                        if comp_id in self.class_names:
                            label_text = self.class_names[comp_id]
                            font_size = 7 if len(label_text) > 6 else 8
                        else:
                            label_text = f"{comp_id}"
                            font_size = 8
                    else:
                        # Other stages: Filtration stages - controlled by show_filtration_text
                        show_text = show_filtration_text
                        label_text = f"{comp_id}"
                        font_size = 8

                    if show_text:
                        ax.text(
                            node["x"],
                            node["y"],
                            label_text,
                            ha="center",
                            va="center",
                            fontsize=font_size,
                            fontweight="bold",
                            color="white",
                            rotation=0,
                        )

        # Add stage labels with threshold values - BIGGER text for paper
        for i, stage_name in enumerate(stage_names):
            ax.text(
                x_positions[i],
                0.05,
                stage_name,
                ha="center",
                va="center",
                fontsize=14,  # Bigger for paper
                rotation=0,
                fontweight="normal",  # Remove bold as requested
            )

        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Cluster Evolution Stages", fontsize=16)  # Bigger for paper
        ax.set_ylabel("Component Size (Normalized)", fontsize=16)  # Bigger for paper

        plot_title = f"Sankey Diagram - {title if title else key}"
        ax.set_title(plot_title, fontsize=18, pad=20)  # Bigger for paper

        # Remove ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        return ax

    def plot_stacked_bars(
        self,
        key,
        original_labels=None,
        ax=None,
        title=None,
        gray_second_layer=True,
        show_true_labels_text=True,
        show_filtration_text=True,
    ):
        """
        Create a 5-stage stacked bar chart:
        Stage 1: True labels
        Stage 2: Initial PH clusters (many small clusters)
        Stage 3: Clusters similar to true labels
        Stage 4: Intermediate merging
        Stage 5: Final single cluster
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 10))

        if key not in self.components_ or key not in self.labels_:
            ax.text(
                0.5,
                0.5,
                f"No data found for key: {key}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Check if we have original labels
        has_original = original_labels is not None
        if not has_original:
            ax.text(
                0.5,
                0.5,
                "No true labels provided for 5-stage visualization",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Get the 4 thresholds (for stages 2-5)
        thresholds = sorted([float(t) for t in self.components_[key].keys()])

        if len(thresholds) < 4:
            ax.text(
                0.5,
                0.5,
                f"Need 4 thresholds for 5-stage visualization, got {len(thresholds)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            return ax

        # Create consistent color mapping if not already created
        if self.color_mapping is None:
            self.color_mapping = self._create_color_mapping(
                key, thresholds, original_labels
            )

        # Define the stages with actual threshold values
        # Add small gap after true labels, then stick filtration stages together
        stage_names = ["True Labels", ""] + [f"{t:.4f}" for t in thresholds]
        x_positions = np.array(
            [0, 1.5, 2.0, 3.0, 4.0, 5.0]
        )  # Smaller gap after true labels, then consecutive
        bar_width = 1.0  # Make bars stick together by using full width
        # n_stages = len(stage_names)
        # Get unique true labels
        # unique_labels = sorted(set(original_labels))

        # Process each stage
        stage_data = []

        # Stage 1: True labels
        original_counts = Counter(original_labels)
        stage_data.append(("True Labels", original_counts))

        # Stage 2: Empty separator (white bar)
        stage_data.append(("", {}))

        # Stages 3-6: Each threshold
        for i, threshold in enumerate(thresholds):
            threshold_str = str(threshold)
            if threshold_str in self.labels_[key]:
                labels_at_threshold = self.labels_[key][threshold_str]
                component_counts = Counter(labels_at_threshold)
                stage_data.append((f"{threshold:.4f}", component_counts))

        # Create stacked bars
        for stage_idx, (stage_name, component_counts) in enumerate(stage_data):
            if stage_idx == 1:  # Empty separator - just skip, no visible bar
                continue

            if not component_counts:
                continue

            # Stack components in a single bar
            bottom = 0
            for comp_id in sorted(component_counts.keys()):
                count = component_counts[comp_id]

                # Use gray for second layer (initial PH clusters) if requested
                if (
                    gray_second_layer and stage_idx == 2
                ):  # Second threshold layer (first filtration stage)
                    color = (0.7, 0.7, 0.7, 1.0)  # Light gray
                else:
                    color = self.color_mapping[comp_id]

                # Create bar segment
                ax.bar(
                    x_positions[stage_idx],
                    count,
                    bottom=bottom,
                    width=bar_width,
                    color=color,
                    alpha=0.85,
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Add component label if significant and flags allow
                if (
                    count > sum(component_counts.values()) * 0.04
                ):  # Show labels for >4% of total
                    show_text = False
                    if stage_idx == 0:
                        # Stage 1: True labels - controlled by show_true_labels_text
                        show_text = show_true_labels_text
                        label_text = f"{comp_id}"
                        font_size = (
                            12 if len(label_text) <= 6 else 10
                        )  # Bigger for paper
                    elif stage_idx > 1:  # Skip empty separator stage (stage_idx == 1)
                        # Filtration stages - controlled by show_filtration_text
                        show_text = show_filtration_text
                        label_text = f"{comp_id}"
                        font_size = 12  # Bigger for paper

                    if show_text:
                        ax.text(
                            x_positions[stage_idx],
                            bottom + count / 2,
                            label_text,
                            ha="center",
                            va="center",
                            fontsize=font_size,
                            fontweight="normal",  # Remove bold as requested
                            color="white",
                            rotation=0,
                        )

                bottom += count

        # Customize plot - BIGGER fonts for paper
        ax.set_xlabel("Cluster Evolution Stages", fontsize=16)  # Bigger for paper
        ax.set_ylabel("Component Size", fontsize=16)  # Bigger for paper
        # ax.set_title(
        #     f"Stacked Bar Chart - {title if title else key}", fontsize=18, pad=20  # Bigger for paper
        # )

        # Set x-axis labels with threshold values (skip the empty separator)
        ax.set_xticks(
            [x_positions[0]] + list(x_positions[2:])
        )  # Skip separator position
        ax.set_xticklabels(
            [stage_names[0]] + stage_names[2:], fontsize=14, rotation=0
        )  # Bigger for paper

        # Set x-axis limits to avoid any weird spacing around the invisible separator
        ax.set_xlim(-0.5, 5.5)

        # Remove all visual elements except the bars
        ax.set_yticks([])
        ax.grid(False)  # Explicitly turn off grid
        for spine in ax.spines.values():
            spine.set_visible(False)

        return ax


class FlowVisualizer:
    """
    High-level flow visualization class that wraps ComponentEvolutionVisualizer
    with more user-friendly interface.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (20, 12),
        dpi: int = 800,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the flow visualizer.

        Args:
            figsize: Figure size for plots
            dpi: DPI for saved plots
            class_names: Optional dictionary mapping class indices to names
        """
        self.figsize = figsize
        self.dpi = dpi
        self.class_names = class_names

    def plot_sankey_flow(
        self,
        cluster_evolution: Dict,
        save_path: Optional[str] = None,
        # title: str = "5-Stage Cluster Evolution",
        title: str = None,
        show_true_labels_text: bool = True,
        show_filtration_text: bool = True,
    ) -> plt.Figure:
        """
        Plot a 5-stage Sankey diagram showing cluster evolution. using ComponentEvolutionVisualizer

        Args:
            cluster_evolution: Dictionary from ClusterFlowAnalyzer.compute_cluster_evolution()
            save_path: Path to save the figure
            title: Title for the plot
            show_true_labels_text: Whether to show text labels in true labels blocks
            show_filtration_text: Whether to show text labels in filtration stage blocks

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.tight_layout()

        # Extract components and labels
        components_ = cluster_evolution.get("components_", {})
        labels_ = cluster_evolution.get("labels_", {})
        true_labels = cluster_evolution.get("true_labels", None)

        # Create visualizer
        visualizer = ComponentEvolutionVisualizer(
            components_, labels_, self.class_names
        )

        # Plot Sankey diagram
        for key in components_.keys():
            visualizer.plot_sankey(
                key,
                true_labels,
                ax,
                title,
                show_true_labels_text=show_true_labels_text,
                show_filtration_text=show_filtration_text,
            )
            break  # avoid plotting all distance metrics in one plot

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved Sankey flow diagram: {save_path}")

        return fig

    def plot_stacked_bar_evolution(
        self,
        cluster_evolution: Dict,
        save_path: Optional[str] = None,
        title: str = "5-Stage Cluster Evolution",
        show_true_labels_text: bool = True,
        show_filtration_text: bool = True,
    ) -> plt.Figure:
        """
        Plot a stacked bar chart showing cluster evolution using ComponentEvolutionVisualizer

        Args:
            cluster_evolution: Dictionary from ClusterFlowAnalyzer.compute_cluster_evolution()
            save_path: Path to save the figure
            title: Title for the plot
            show_true_labels_text: Whether to show text labels in true labels blocks
            show_filtration_text: Whether to show text labels in filtration stage blocks

        Returns:
            matplotlib Figure object
        """
        # fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 0.6))
        fig, ax = plt.subplots(figsize=self.figsize)
        plt.tight_layout()

        # Extract components and labels
        components_ = cluster_evolution.get("components_", {})
        labels_ = cluster_evolution.get("labels_", {})
        true_labels = cluster_evolution.get("true_labels", None)

        # Create visualizer
        visualizer = ComponentEvolutionVisualizer(
            components_, labels_, self.class_names
        )

        # Plot stacked bars
        for key in components_.keys():
            visualizer.plot_stacked_bars(
                key,
                true_labels,
                ax,
                title,
                show_true_labels_text=show_true_labels_text,
                show_filtration_text=show_filtration_text,
            )
            break  # so that we only plot one distance metric in one plot

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            print(f"Saved stacked bar evolution chart: {save_path}")

        return fig


def analyze_activation_flows(
    activation_file: str,
    output_dir: str,
    model_name: str,
    condition_name: str,
    true_labels: Optional[np.ndarray] = None,
    max_points: int = 100,
    max_thresholds: int = 6,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Analyze cluster flow evolution for activations using 5-stage ComponentEvolutionVisualizer..

    Args:
        activation_file: Path to activation .npy file
        output_dir: Output directory for flow visualizations
        model_name: Model name
        condition_name: Condition name
        true_labels: True class labels
        max_points: Maximum points to use for analysis
        max_thresholds: Maximum number of thresholds to plot
        class_names: Optional dictionary mapping class indices to names

    Returns:
        Dictionary with flow analysis results
    """
    print(f"Analyzing cluster flow evolution for {model_name} - {condition_name}")

    # Load activations
    try:
        all_activations = np.load(activation_file, allow_pickle=True).item()
        if not isinstance(all_activations, dict):
            print(f"Warning: Expected dictionary, got {type(all_activations)}")
            return {}
    except Exception as e:
        print(f"Error loading {activation_file}: {e}")
        return {}

    # Create output directory
    flow_output_dir = os.path.join(output_dir, f"{model_name}_{condition_name}")
    os.makedirs(flow_output_dir, exist_ok=True)

    # Initialize flow visualizer
    flow_viz = FlowVisualizer(figsize=(40, 20), class_names=class_names)

    # Default class names if none provided
    if class_names is None:
        class_names = {i: f"Class_{i}" for i in range(10)}

    # Import required modules
    from mst_proc import MSTProcessor

    results = {}

    # Process each layer
    for layer_name, activation_data in all_activations.items():
        print(f"  Processing layer: {layer_name}")

        # Handle different activation shapes
        if len(activation_data.shape) == 3:
            # [batch_size, seq_len, hidden_dim] - use class token
            pc = activation_data[:, 0, :]
        elif len(activation_data.shape) == 2:
            # [batch_size, hidden_dim] - already flattened
            pc = activation_data
        else:
            print(f"    Warning: Unexpected shape {activation_data.shape}, skipping...")
            continue

        # Subsample if too many points
        if pc.shape[0] > max_points:
            indices = np.random.choice(pc.shape[0], max_points, replace=False)
            pc = pc[indices]
            layer_labels = true_labels[indices] if true_labels is not None else None
        else:
            layer_labels = true_labels

        if layer_labels is None:
            print(f"    Warning: No labels for {layer_name}, skipping...")
            continue

        # Clean layer name for filename
        clean_layer_name = layer_name.replace("/", "_").replace(".", "_")

        # Initialize MST processor for distance calculations
        mst_obj = MSTProcessor()

        try:
            # Compute distance matrices
            X_pca = mst_obj.pca_utils(pc)

            distance_matrices = {
                "Euclidean": distance_matrix(pc),
                "Mahalanobis": mst_obj.fast_maha(X_pca),
                "Cosine": mst_obj.cosine_gen(pc),
            }

            # Add density normalized versions
            distance_matrices[
                "Density_Normalized_Euclidean"
            ] = mst_obj.density_normalizer(
                X=X_pca, dists=distance_matrices["Euclidean"], k=5
            )
            distance_matrices[
                "Density_Normalized_Mahalanobis"
            ] = mst_obj.density_normalizer(
                X=X_pca, dists=distance_matrices["Mahalanobis"], k=5
            )

            layer_results = {}

            # Process each distance metric
            for dist_name, dist_matrix in distance_matrices.items():
                print(f"    Processing {dist_name} distance metric...")

                try:
                    # Create title
                    if condition_name.lower() in ["inference", "clean"]:
                        if model_name.lower() == "original":
                            title_prefix = f"ViT Model - {dist_name} - {layer_name}"
                        else:
                            title_prefix = f"ViT {model_name.replace('_', ' ').title()} - {dist_name} - {layer_name}"
                    else:
                        if model_name.lower() == "original":
                            title_prefix = f"ViT Model - {condition_name.replace('_', ' ').title()} - {dist_name} - {layer_name}"
                        else:
                            title_prefix = f"ViT {model_name.replace('_', ' ').title()} - {condition_name.replace('_', ' ').title()} - {dist_name} - {layer_name}"

                    # Compute cluster evolution
                    analyzer = ClusterFlowAnalyzer(
                        dist_matrix, max_thresholds=max_thresholds
                    )
                    cluster_evolution = analyzer.compute_cluster_evolution(layer_labels)

                    # Save Sankey diagram
                    sankey_path = os.path.join(
                        flow_output_dir, f"{clean_layer_name}_{dist_name}_sankey.png"
                    )
                    sankey_fig = flow_viz.plot_sankey_flow(
                        cluster_evolution,
                        save_path=sankey_path,
                        # title=f"Sankey Diagram - {title_prefix}",
                    )
                    plt.close(sankey_fig)

                    # Save stacked bar chart
                    bars_path = os.path.join(
                        flow_output_dir,
                        f"{clean_layer_name}_{dist_name}_stacked_bars.png",
                    )
                    bars_fig = flow_viz.plot_stacked_bar_evolution(
                        cluster_evolution,
                        save_path=bars_path,
                        title=f"Stacked Bar Chart - {title_prefix}",
                    )
                    plt.close(bars_fig)

                    # Store results
                    layer_results[dist_name] = {
                        "cluster_evolution": cluster_evolution,
                        "sankey_path": sankey_path,
                        "bars_path": bars_path,
                    }

                except Exception as e:
                    print(f"      Error processing {dist_name}: {e}")
                    continue

            results[layer_name] = layer_results

        except Exception as e:
            print(f"    Error processing layer {layer_name}: {e}")
            continue

    return results


if __name__ == "__main__":
    print("Flow Visualization: 5-stage component evolution through persistent homology")
    print(
        "Stages: True Labels → Initial Clusters → Similar to True → Intermediate → Final Cluster"
    )
