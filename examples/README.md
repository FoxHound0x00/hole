# HOLE Examples

This directory contains examples demonstrating the HOLE library functionality.

## Quick Start Examples

### 1. `simple_example.py`
**Recommended starting point** - Basic usage with persistence diagrams and PCA visualization.
- 20 lines of code
- Shows fundamental HOLE workflow
- Good for beginners

### 2. `distance_metrics_example.py` 
Compare different distance metrics (Euclidean, cosine, Manhattan).
- 30 lines of code  
- Demonstrates metric selection
- Shows impact of different distance functions

### 3. `cluster_analysis_example.py`
Analyze cluster evolution through persistent homology.
- 40 lines of code
- Shows cluster flow analysis
- Good for understanding topological clustering

## Advanced Example

### 4. `comprehensive_example.py`
**Advanced users only** - Comprehensive demonstration of ALL library features.
- 425 lines of code
- Multiple distance metrics
- All visualization types
- Can be overwhelming for beginners

## Running Examples

```bash
cd examples/

# Start with the simple example
python simple_example.py

# Try different distance metrics  
python distance_metrics_example.py

# Explore cluster analysis
python cluster_analysis_example.py

# For advanced features (takes longer to run)
python comprehensive_example.py
```

## Output

Examples will generate PNG files showing the visualizations. These are automatically saved to the current directory.

## Requirements

All examples require the HOLE library to be installed with its dependencies:
- numpy
- matplotlib  
- scikit-learn
- gudhi
- scipy
- seaborn
