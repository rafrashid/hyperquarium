"""
hac_pipeline/config/config.py
------------------------------
Central configuration for the HAC pipeline.
Only this file should need editing before a run.
Paths (xgb_shap_dir, output_dir) are derived at runtime in hac.py from --spectra;
do not set them here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HACConfig:
    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    pixels_per_roi: int = 500
    """Number of pixels sampled per ROI via stratified random sampling.
    ROIs with fewer available pixels include all rows (with a logged warning)."""

    random_seed: int = 42
    """Fixed seed for reproducible stratified sampling. Drawn fresh each run."""

    # ------------------------------------------------------------------
    # Feature selection
    # ------------------------------------------------------------------
    n_top_features: int = 50
    """Top-N features selected by mean |SHAP| across turf_algae Level 4 classes."""

    turf_label_substring: str = "turf_algae"
    """Substring used to filter Level 4 SHAP columns to turf classes only."""

    # ------------------------------------------------------------------
    # PCA pre-reduction
    # ------------------------------------------------------------------
    pca_variance_threshold: float = 0.95
    """Fraction of variance retained after PCA. Controls number of components."""

    # ------------------------------------------------------------------
    # NaN handling
    # ------------------------------------------------------------------
    nan_col_threshold: float = 0.5
    """Drop feature columns where the fraction of NaN rows exceeds this value.
    Empirically motivated by the NaN distribution in the compiled parquet:
    - pilot dataset: GLCM/specdiv window/plot sizes >= 51 are ~100% NaN → 0.5
    - reefcompare dataset: cutoff is around window/plot size 75 → adjust accordingly.
    """

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------
    k_values: list = field(default_factory=lambda: [2, 5, 10, 20])
    """HAC dendrogram cut heights (number of clusters) to evaluate.
    Silhouette-best K is added automatically if not already present."""

    silhouette_k_min: int = 2
    silhouette_k_max: int = 50
    """Range of K values swept to find silhouette-best K."""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    majority_vote_min_pixels: int = 10
    """Minimum pixel count per ROI required to report a majority vote result.
    ROIs below this threshold are logged as warnings and excluded from metrics."""

    # ------------------------------------------------------------------
    # Feature separation
    # ------------------------------------------------------------------
    n_top_separation_features: int = 20
    """Number of top-ranked features shown in the feature separation dot plot (7e)."""

    # ------------------------------------------------------------------
    # Paths — derived at runtime in hac.py; do not set manually
    # ------------------------------------------------------------------
    xgb_shap_dir: Path = None
    """Path to outputs/spectra_{X}/level_4/ containing feature_importance_shap.csv.
    Set automatically from --spectra argument."""

    output_dir: Path = None
    """Path to outputs/hac/spectra_{X}/. Set automatically from --spectra argument."""

    held_out_summary_path: Path = None
    """Path to held_out_accuracy_summary.csv from xgb_pipeline.
    Used to annotate dendrogram and cluster accuracy plot.
    Set automatically from --spectra argument."""

    # ------------------------------------------------------------------
    # Poorly-classified ROI annotation
    # ------------------------------------------------------------------
    poor_roi_prop_correct_threshold: float = 0.5
    """ROIs with prop_correct below this value across all spectra types are
    highlighted in red on the dendrogram. Read from held_out_accuracy_summary.csv."""