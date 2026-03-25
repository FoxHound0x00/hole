# Plan: Fix Sankey & Blob Visualization Bugs

Based on comprehensive code analysis, 10 critical bugs and 9 interpretability issues were identified. **7 major bugs have been fixed** (cluster ID renumbering, color mapping, threshold precision & validation). **Remaining work** focuses on naming improvements (outlier → minority class filtering), hull/contour alignment, and documentation.

---

## TL;DR
The Sankey and blob visualizations have been significantly improved with fixes for cluster ID renumbering, color mapping across all thresholds, threshold precision, and visual connections between diagrams. Remaining issues include filter mask behavior (intentional design), outlier detection naming (should be "minority class filtering"), hull/contour alignment, and general interpretability improvements for stage naming and thresholds.

---

## Remaining Steps

### **1. Document Filter Mask Behavior** 
[cluster_flow.py](hole/visualization/cluster_flow.py#L114-L149): The filter mask is computed at middle threshold but applied to ALL thresholds. **NOTE**: User confirmed this is intentional - "noise datapoint in middle cluster is noise in all".

**Status**: Working as intended, documented in docstring.

---

### **2. Rename "Outlier" Detection to "Minority Class Filtering"**
[scatter_hull.py](hole/visualization/scatter_hull.py#L419-L451): Removes ALL points of minority classes (not statistical outliers), misleading users about what's being shown.

**Fix approach**: Rename to `minority_class_threshold` and `_filter_minority_classes()`. Update all variable names (`minority_mask` not `outlier_mask`). Add proper docstrings.

---

### **3. Fix Hull/Contour Mismatch**
[scatter_hull.py](hole/visualization/scatter_hull.py#L903-L930): Convex hull drawn around ALL cluster points, but contours only for non-outlier points, creating visual mismatch.

**Fix approach**: Draw hull around non-outlier points only (consistent with contours) OR draw contours for all points (consistent with hull).

---

### **4. Improve Remaining Interpretability Issues**
- Add consistent stage naming/numbering throughout code and docs
- Explain `gray_second_layer` purpose

---

## Verification

After implementation:
1. Run `vit_inference_quantized.py` with `filter_small_clusters=True` - verify behavior
2. Confirm hull boundaries align with contour regions
3. Check minority class filtering behavior is clearly labeled
4. Test with edge cases: very few points, single cluster, all same label

---

## Decisions (Updated)

**Filter Strategy**: ✓ **RESOLVED** - Intentional design: filter at middle threshold, apply to all stages. Rationale: noise in middle cluster → noise everywhere.

**Cluster Renumbering**: ✓ **IMPLEMENTED** - Keep original IDs to preserve color mapping.

**Color Assignment**: ✓ **IMPLEMENTED** - Unified mapping across all stages based on most common true label at each threshold.

**Outlier Handling**: Rename existing minority filtering vs implement real statistical outliers → **Recommend rename first, then add statistical outliers as separate feature**

---

## Detailed Bug Analysis (from subagent investigation)

### CRITICAL BUGS

#### **BUG 1: Filter Mask Applied Incorrectly Across All Thresholds**
**Location**: hole/visualization/cluster_flow.py#L114-L149

**Description**: The `filter_small_clusters` feature has a severe logic error. The filter mask is computed based on cluster sizes at the middle threshold (stage 3, `selected_thresholds[1]`), but then applied to ALL thresholds including earlier ones.

**Code**:
```python
# Line 114-127: Compute filter based on MIDDLE threshold only
if filter_small_clusters and len(selected_thresholds) >= 3:
    middle_threshold = str(selected_thresholds[1])
    middle_labels = all_cluster_labels[middle_threshold]
    cluster_sizes = Counter(middle_labels)
    small_clusters = {cid for cid, count in cluster_sizes.items() if count <= min_cluster_size}
    filter_mask = np.array([label not in small_clusters for label in middle_labels])

# Line 130-144: Apply same filter to ALL thresholds
for threshold in selected_thresholds:
    if filter_small_clusters:
        filtered_labels = cluster_labels[filter_mask]  # Wrong!
```

**Impact**: 
- A point that's in a SMALL cluster at stage 3 might be in a LARGE cluster at stage 2
- Filtering it out at stage 2 is incorrect and distorts the cluster evolution
- The Sankey diagram will show fewer points at early stages than actually exist
- Flow calculations between stages become incorrect because point counts don't match

---

#### **BUG 2: Cluster ID Renumbering Breaks Color Mapping**
**Location**: hole/visualization/cluster_flow.py#L132-L144

**Description**: After filtering, cluster IDs are renumbered to be consecutive (0, 1, 2, ...), but the color mapping created in `_create_color_mapping()` uses the ORIGINAL cluster IDs. ✓ **FIXED**

---

#### **BUG 5: Outlier Detection Removes Entire Minority Classes**
**Location**: hole/visualization/scatter_hull.py#L419-L451

**Description**: The outlier detection marks ALL points of a minority class within a cluster as outliers, rather than detecting statistical outliers.

**Code**:
```python
for class_id, count in zip(unique_classes, class_counts):
    class_percentage = count / cluster_size
    if class_percentage < percentage:
        # This class is an outlier class in this cluster
        cluster_indices = np.where(cluster_mask)[0]
        class_mask_in_cluster = cluster_true_labels == class_id
        outlier_class_indices = cluster_indices[class_mask_in_cluster]
        outlier_mask[outlier_class_indices] = True
```

**Impact**:
- If a class makes up 8% of a cluster (below 10% threshold), ALL points of that class are marked as outliers
- This is NOT statistical outlier detection, it's class filtering

---

#### **BUG 6: Hull and Contour Mismatch**
**Location**: hole/visualization/scatter_hull.py#L916-L930

**Description**: In `plot_pca_with_cluster_hulls()`, the convex hull is drawn around ALL points in a cluster, but contours are only drawn for non-outlier classes.

**Impact**:
- Hull boundary doesn't match what the contours represent
- Creates a "dead zone" in the hull where no contours are drawn

---

### INTERPRETABILITY ISSUES

#### **ISSUE 1: Stage Naming and Numbering Inconsistency**
Throughout the code, stages are referred to with inconsistent terminology:
- "True Labels" vs "Stage 1" vs "stage 0 in code"
- "Second layer" vs "Stage 2" vs "First threshold" vs "stage_idx=0"
- "Middle threshold" vs "Stage 3" vs "thresholds[1]"

---

#### **ISSUE 3: "Outlier" Detection Is Actually Class Filtering**Current naming suggests statistical outlier detection, but actually filters entire minority classes.

---

#### **ISSUE 6: Gray Second Layer Purpose Not Explained**
The `gray_second_layer` parameter grays out the initial clusters stage (stage 2) to "de-emphasize" it, but the purpose isn't explained in the visualization or docstrings.

---

#### **ISSUE 7: "Middle Threshold" and "Fourth Threshold" Are Arbitrary Names**
These names don't convey semantic meaning. "Middle" suggests median, but it's actually the "similar to true labels" threshold.

---

#### **ISSUE 8: Stacked Bar "Empty Separator" Stage Is Confusing**
An empty stage is added between true labels and thresholds, creating visual gap that looks like a missing bar rather than intentional spacing.

---

### DESIGN ISSUES

#### **DESIGN ISSUE 1: Hardcoded Magic Numbers**
- Line 761: `if node["height"] > 0.015` - hardcoded threshold for showing labels  
- Line 873: `if count > sum(component_counts.values()) * 0.04` - hardcoded 4% threshold
- Line 261: `combined_score = 0.7 * purity_score + 0.3 * homogeneity_score` - hardcoded weights

---

#### **DESIGN ISSUE 2: Tight Coupling Between Stages**
The entire pipeline assumes exactly 5 stages with 4 thresholds. If you want different stage counts, code breaks.

---

#### **DESIGN ISSUE 3: No Validation of Input Data**
Neither `compute_cluster_evolution()` nor visualization functions validate input data dimensions, leading to cryptic errors.

---

## PRIORITY SUMMARY

**Remaining Issues**:
1. Bug 1: Filter mask - intentional design, properly documented
2. Bug 5: Outlier detection removes entire classes - needs renaming to minority_class_filter
3. Bug 6: Hull/contour mismatch - needs alignment decision
4. Issue 1: Stage naming inconsistency - makes code hard to follow
5. Issue 3: "Outlier" naming misleading - should be "minority class filtering"
6. Issue 6: Gray second layer purpose unclear
7. Issue 7: "Middle threshold" naming doesn't convey semantic meaning
8. Issue 8: Stacked bar empty separator confusing
9. Design issues 1-3: Magic numbers, tight coupling, missing validation

**Completed Fixes** ✓:
- Cluster ID renumbering removed (preserves color mapping)
- Stage indexing simplified
- Color matching across ALL thresholds (not just middle)
- Threshold precision fixed (f"{:.8f}")
- Threshold validation with interpolation
- Filter behavior documented
- Legend for contour colors added
- Cluster hulls colored by most common true label
- Cluster ID labels added
- Visual connection between Sankey and Blob established