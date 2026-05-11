"""
Library-wide default constants.

Centralised here so the same magic number isn't repeated across modules.
Each constant has a one-line comment explaining what it controls — when in
doubt, prefer adding a constructor argument over reading a constant directly.
"""

# Default seed for any reduction or sampling that should be deterministic.
# Override per call via the explicit `random_state` argument on the relevant
# constructor (PCA, t-SNE, MDS, MSTProcessor, ...).
DEFAULT_RANDOM_STATE: int = 42

# Default upper threshold for MSTProcessor edge filtering. Tuned empirically
# for normalised distance matrices; raise it for unscaled embeddings.
DEFAULT_MST_THRESHOLD: float = 35.0

# Padding (as a fraction of axis range) added around scatter / hull plots so
# data does not touch the figure edge.
DEFAULT_PLOT_PADDING: float = 0.15

# Quantile of off-diagonal pairwise distances used as the auto Rips
# `max_edge_length`. Larger -> more topological detail, more compute.
DEFAULT_RIPS_QUANTILE: float = 0.95
