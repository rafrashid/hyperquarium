"""
config/config.py
Central configuration for all pipeline parameters, paths, and model settings.
Edit this file before running the pipeline.

Column naming conventions (from your compiled tables):
  Spectral bands    : <wavelength>_nm          e.g. 425_nm, 583_nm
  GLCM              : <metric>_window_<size>   e.g. contrast_window_7, energy_window_203
  Spectral diversity: sdiv_<type>_<measure>_plot_<size>
                                               e.g. sdiv_alpha_sdiv_plot_7, sdiv_beta_lcsd_plot_203
  Metadata (excluded from features):
                      roi_ID, exposure, n_valid_pixels, scan_ID, dataset, label, line, sample
"""

import re
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"

# Pre-built input tables — one parquet file per spectra type.
# Each file contains: spectral features + GLCM + spectral diversity + metadata + label.
# Spatial features (GLCM, sdiv) are computed from the corresponding spectra:
#   A: reflectance spectra        | GLCM + sdiv from A
#   B: 2nd derivative of A        | GLCM + sdiv from A
#   C: L2-norm of A               | GLCM + sdiv from C
#   D: 2nd derivative of C        | GLCM + sdiv from C
# These files are assumed to exist on disk already — no generation step in the pipeline.
SPECTRA_FILES = {
    "A": DATA_DIR / "spectra_A.parquet",
    "B": DATA_DIR / "spectra_B.parquet",
    "C": DATA_DIR / "spectra_C.parquet",
    "D": DATA_DIR / "spectra_D.parquet",
}


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

# Raw label column — as it appears in your compiled tables
RAW_LABEL_COLUMN = "label"

# Metadata columns — excluded from features entirely (not spatial features, not spectral)
# 'line' and 'sample' are pixel coordinates; 'dataset' is the labelset origin tag
METADATA_COLUMNS = {"roi_ID", "exposure", "n_valid_pixels", "scan_ID", "dataset",
                    "label", "line", "sample"}

# Label mapping file — CSV with columns: labelset, Level_0, Level_1, Level_2, Level_3
LABEL_MAPPING_FILE = DATA_DIR / "labelset_mapping.csv"

# In the mapping file, Level_0 is the source label column to match against RAW_LABEL_COLUMN
LABEL_MAPPING_LEVEL0_COL = "Level_0"

# Dataset filter — only rows where the 'labelset' column matches this value
LABEL_MAPPING_DATASET = "pilot"

# Label columns produced after remapping (added to the DataFrame by remap_labels())
LABEL_COLUMNS = {
    1: "label_level1",
    2: "label_level2",
    3: "label_level3",
}

# Turf algae class name — must match exactly the value in Level_3 of the mapping file
TURF_ALGAE_CLASS = "turf_algae"


# ---------------------------------------------------------------------------
# Feature column identification
# ---------------------------------------------------------------------------
# These patterns are used throughout the pipeline to classify columns into
# feature families for SHAP grouping, scale-response curves, and auditing.

# Spectral band pattern: <integer>_nm  e.g. 425_nm, 703_nm
SPECTRAL_PATTERN = re.compile(r"^\d+_nm$")

# GLCM pattern: <metric>_window_<size>  e.g. contrast_window_7, energy_window_203
# Valid metrics: energy, entropy, homogeneity, contrast
GLCM_PATTERN = re.compile(r"^(energy|entropy|homogeneity|contrast)_window_(\d+)$")

# Spectral diversity pattern: sdiv_<sdiv_measure>_plot_<size>
# The measure name is everything between sdiv_ and _plot_
# e.g. sdiv_alpha_sdiv_plot_7  -> measure='alpha_sdiv', size=7
#      sdiv_beta_lcsd_plot_203 -> measure='beta_lcsd',  size=203
SDIV_PATTERN = re.compile(r"^sdiv_(.+)_plot_(\d+)$")

# Window sizes confirmed from your data — 11 sizes, each approximately doubling area
WINDOW_SIZES = [7, 9, 13, 17, 25, 35, 51, 71, 101, 143, 203]

# GLCM metric names (for labelling plots and reports)
GLCM_METRICS = ["energy", "entropy", "homogeneity", "contrast"]

# Spectral diversity measure names (the full measure identifier between sdiv_ and _plot_)
SDIV_METRICS = ["alpha_sdiv", "beta_lcsd"]


def classify_column(col: str) -> str:
    """
    Classifies a column name into one of four feature families, or 'metadata'.

    Returns one of: 'spectral', 'glcm', 'sdiv', 'label', 'metadata'

    Used by get_feature_columns() and SHAP grouping functions.
    """
    if col in METADATA_COLUMNS:
        return "metadata"
    if col in LABEL_COLUMNS.values():
        return "label"
    if SPECTRAL_PATTERN.match(col):
        return "spectral"
    if GLCM_PATTERN.match(col):
        return "glcm"
    if SDIV_PATTERN.match(col):
        return "sdiv"
    return "unknown"


def extract_window_size(col: str) -> int | None:
    """
    Extracts the integer window size from a GLCM or sdiv column name.

    Returns the size as int, or None if the column is not a spatial feature.

    Examples:
        contrast_window_71      -> 71
        sdiv_alpha_sdiv_plot_25 -> 25
        425_nm                  -> None
    """
    m = GLCM_PATTERN.match(col) or SDIV_PATTERN.match(col)
    return int(m.group(2)) if m else None


# ---------------------------------------------------------------------------
# Output format
# ---------------------------------------------------------------------------
# "parquet" — recommended for large datasets (26M rows); compact and fast; requires pyarrow
# "csv"     — universally readable; larger file size; preferred when sharing with ecologists
# Applies to tabular outputs (SHAP values, feature importance, large DataFrames).
# Plots are always PNG. Metadata/metrics are always JSON.
OUTPUT_FORMAT: str = "parquet"


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

@dataclass
class SplitConfig:
    train_frac: float = 0.98
    val_frac: float = 0.01
    test_frac: float = 0.01
    random_seed: int = 42
    # Stratify on the finest label level to ensure rare classes appear in all splits
    stratify_level: int = 1


SPLIT = SplitConfig()


# ---------------------------------------------------------------------------
# XGBoost hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class XGBConfig:
    tree_method: str = "hist"  # Always hist at this scale
    device: str = "cpu"  # Set to "cuda" if GPU node is available

    max_depth: int = 6
    eta: float = 0.05  # Learning rate / shrinkage
    subsample: float = 0.8
    colsample_bytree: float = 0.7  # 0.6–0.8 recommended for 180+ features
    min_child_weight: int = 10  # Increase for rare classes
    gamma: float = 0.1
    reg_lambda: float = 1.0  # L2 regularisation
    reg_alpha: float = 0.0  # L1 regularisation

    num_boost_round: int = 2000  # Upper bound — early stopping cuts this short
    early_stopping_rounds: int = 50
    verbose_eval: int = 100  # Log eval metric every N rounds

    seed: int = 42


XGB = XGBConfig()


# ---------------------------------------------------------------------------
# Level-specific settings
# ---------------------------------------------------------------------------

@dataclass
class LevelConfig:
    level: int
    n_classes: int
    objective: str
    eval_metric: str
    run_unweighted: bool = False  # If True, also trains without class weights


LEVEL_CONFIGS = {
    3: LevelConfig(
        level=3, n_classes=2,
        objective="binary:logistic", eval_metric="aucpr",
        run_unweighted=True,  # Hypothesis baseline: natural decision boundary
    ),
    2: LevelConfig(
        level=2, n_classes=5,
        objective="multi:softprob", eval_metric="mlogloss",
    ),
    1: LevelConfig(
        level=1, n_classes=8,
        objective="multi:softprob", eval_metric="mlogloss",
    ),
}


# ---------------------------------------------------------------------------
# SHAP settings
# ---------------------------------------------------------------------------

@dataclass
class SHAPConfig:
    n_top_features: int = 20
    shap_sample_size: int | None = 50_000  # Set None to use full test set
    random_seed: int = 42


SHAP_CFG = SHAPConfig()

# ---------------------------------------------------------------------------
# Pipeline control
# ---------------------------------------------------------------------------

SPECTRA_TYPES = ["A", "B", "C", "D"]
LEVELS = [3, 2, 1]  # Coarse to fine
