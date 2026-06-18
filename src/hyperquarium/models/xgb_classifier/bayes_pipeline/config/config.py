"""
config/config.py — single edit point for the Bayesian ROI-level feature analysis.

Mirrors xgb_pipeline's dataclass-config convention: this is the only file that
should need editing before a run. Each run targets ONE dataset (pilot OR
reefcompare) as a separate invocation — there is no dataset axis inside the harness.

See knowledge base: [[bayesian-roi-feature-analysis]].
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Feature naming patterns (shared with xgb_pipeline / feature-family-classification)
# --------------------------------------------------------------------------- #
GLCM_PATTERN = re.compile(r"^(energy|entropy|homogeneity|contrast)_window_(\d+)$")
SDIV_PATTERN = re.compile(r"^sdiv_(.+)_plot_(\d+)$")  # greedy; captures full measure

# Metadata columns present in the compiled parquet (never features).
METADATA_COLUMNS = [
    "roi_ID", "scan_ID", "dataset", "exposure",
    "n_valid_pixels", "line", "sample", "label",
]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass
class PathConfig:
    """All input/output paths. Edit per dataset run."""
    # Input: the compiled pixel-level parquet (one dataset).
    compiled_parquet: Path = Path("data/compiled_dataset.parquet")
    # Input: labelset mapping CSV (defines L1/L2/L3 remapping + inclusion filter).
    labelset_mapping: Path = Path("data/labelset_mapping.csv")
    # Output root for this run. No pilot/ reefcompare/ nesting — the run IS the dataset.
    output_root: Path = Path("outputs/bayes")

    @property
    def roi_summary(self) -> Path:
        """Cached ROI-level aggregation (written once by aggregate stage)."""
        return self.output_root / "roi_summary.parquet"


# --------------------------------------------------------------------------- #
# Feature selection — "7 features at one representative size"
# --------------------------------------------------------------------------- #
@dataclass
class FeatureConfig:
    """
    The 7 conceptual features: 3 specdiv (gamma, alpha, beta) + 4 GLCM.

    Each exists at ~11 window/plot sizes in the parquet; this analysis picks ONE
    representative size per family. GLCM window size and specdiv plot size are
    independent knobs (they share the 7..203 scale but are conceptually different).
    Gamma uses the specdiv plot size.
    """
    glcm_window: int = 35  # representative GLCM window size (7..203)
    specdiv_plot: int = 35  # representative specdiv plot size (7..203)

    glcm_metrics: tuple[str, ...] = ("energy", "entropy", "homogeneity", "contrast")
    # specdiv measure stems as they appear in compiled column names (post gamma-norm).
    # alpha_local / beta_local are the gamma-normalised stems; gamma is standalone.
    sdiv_local_measures: tuple[str, ...] = ("alpha_local", "beta_local")
    sdiv_gamma_measure: str = "gamma"
    # Mirrors compile_features SPECDIV_INCLUDE_GAMMA: the gamma column
    # (sdiv_gamma_plot_<size>) only exists if the parquet was compiled with gamma.
    # Default False to match the current parquet (SPECDIV_INCLUDE_GAMMA = False).
    # Set True once a gamma-inclusive parquet is compiled to restore the full
    # 7-feature design.
    include_gamma: bool = False

    def feature_columns(self) -> dict[str, str]:
        """
        Map short feature name -> actual parquet column name at the chosen size.

        Returns e.g. {"homogeneity": "homogeneity_window_25",
                      "alpha": "sdiv_alpha_local_plot_25", ...}
        Gamma is included only if include_gamma is True (column must exist in parquet).
        """
        cols: dict[str, str] = {}
        for m in self.glcm_metrics:
            cols[m] = f"{m}_window_{self.glcm_window}"
        for m in self.sdiv_local_measures:
            short = m.split("_")[0]  # alpha_local -> alpha
            cols[short] = f"sdiv_{m}_plot_{self.specdiv_plot}"
        if self.include_gamma:
            cols[self.sdiv_gamma_measure] = f"sdiv_gamma_plot_{self.specdiv_plot}"
        return cols

    @property
    def feature_names(self) -> list[str]:
        """Short names, stable order: specdiv (gamma if included) then 4 GLCM."""
        specdiv = (["gamma"] if self.include_gamma else []) + ["alpha", "beta"]
        return [*specdiv, *self.glcm_metrics]


# --------------------------------------------------------------------------- #
# Label hierarchy
# --------------------------------------------------------------------------- #
@dataclass
class LabelConfig:
    """
    Level mapping. Labels reach the analysis via labelset_mapping.csv, applied with
    remap_labels() (data/labels.py). Level_0 acts as the inclusion filter (rows whose
    raw label is absent from the mapping are dropped) — same convention as
    compile_features / plot_summary_figures.

    L4 (ROI identity) is intentionally excluded here — this analysis IS at ROI level.
    """
    levels: tuple[int, ...] = (1, 2, 3)
    raw_label_col: str = "label"
    # The focal class. All turf-referenced contrasts are relative to this.
    turf_class: str = "turf_algae"
    # Mapping CSV column names.
    mapping_raw_col: str = "Level_0"
    level_col_fmt: str = "Level_{level}"  # Level_1, Level_2, Level_3
    # Labelset selection — mirrors train.py's --labelset / LABEL_MAPPING_DATASET.
    # labelset_mapping.csv has a `labelset` column; remap_labels filters to one.
    labelset: str = "reefcompare"  # default matches train.py
    labelset_col: str = "labelset"


# --------------------------------------------------------------------------- #
# Priors (locked — see Steps 2 & 3 in the knowledge base)
# --------------------------------------------------------------------------- #
@dataclass
class PriorConfig:
    """Weakly-informative priors on grand-standardised features. Independent per class."""
    mu_sd: float = 2.0  # mu[c]      ~ Normal(0, 2)
    log_sigma_sd: float = 0.5  # log sigma[c] ~ Normal(0, 0.5)  (log link, positivity)


# --------------------------------------------------------------------------- #
# Sampler
# --------------------------------------------------------------------------- #
@dataclass
class SamplerConfig:
    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.95  # raised: thin-class sigma geometry can be tricky
    random_seed: int = 42
    # nutpie is faster but non-blocking; fall back to PyMC default NUTS if absent.
    use_nutpie: bool = True


# --------------------------------------------------------------------------- #
# Analysis / reporting
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisConfig:
    """
    Thin-class handling: FIT all classes (threshold = 1, keep max data), but every
    output table carries n_roi so each sigma / ratio self-documents how much data
    backs it. The threshold is a *reporting/visualisation* filter, NOT a fitting
    filter. A 1-ROI class returns its sigma PRIOR unchanged (within-class SS = 0).
    """
    min_rois_for_sigma: int = 1  # keep ALL classes in the fit
    # Recommended cutoff below which a sigma is prior-dominated; used by downstream
    # tables/figures to grey-out / annotate, NOT to drop from the fit.
    reliable_sigma_n_roi: int = 15
    credible_interval: float = 0.94  # ArviZ default HDI
    # Diagnostics thresholds that flag a cell for inspection.
    rhat_warn: float = 1.01
    ess_warn: int = 400


# --------------------------------------------------------------------------- #
# Top-level config bundle
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    priors: PriorConfig = field(default_factory=PriorConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)


# Convenience singleton (import-and-edit, like xgb_pipeline).
CONFIG = Config()
