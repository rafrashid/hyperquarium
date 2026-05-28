"""
hac_pipeline/config/config.py
------------------------------
Central configuration for the HAC pipeline.
Only this file should need editing before a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HACConfig:
    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    pixels_per_roi: int = 1000
    random_seed: int = 42

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------
    nan_col_threshold: float = 0.5
    """Drop feature columns where NaN fraction exceeds this value.
    pilot dataset:      0.5  (GLCM/specdiv window/plot >= 51 are ~100% NaN)
    reefcompare dataset: 0.75 (cutoff shifts to window/plot size ~75)
    """

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    pca_variance_threshold: float = 0.95

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    k_values: list = field(default_factory=lambda: [2, 5, 10, 20])
    """Fixed K values to evaluate. k_rois (N unique ROIs) and silhouette-best
    K are added automatically at runtime."""

    silhouette_k_min: int = 2
    silhouette_k_max: int = 50

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    majority_vote_min_pixels: int = 10

    # ------------------------------------------------------------------
    # Feature separation plot
    # ------------------------------------------------------------------
    n_top_separation_features: int = 20

    # ------------------------------------------------------------------
    # Dendrogram annotation
    # ------------------------------------------------------------------
    poor_roi_prop_correct_threshold: float = 0.5

    # ------------------------------------------------------------------
    # Paths — derived at runtime in hac.py; do not set manually
    # ------------------------------------------------------------------
    xgb_shap_dir: Path = None
    output_dir: Path = None
    held_out_summary_path: Path = None
