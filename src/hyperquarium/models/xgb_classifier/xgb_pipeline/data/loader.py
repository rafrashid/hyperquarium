"""
data/loader.py
Data loading, label remapping, train/val/test splitting,
class weight computation, and DMatrix creation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from config.config import (
    SplitConfig, LABEL_COLUMNS, SPLIT,
    RAW_LABEL_COLUMN, LABEL_MAPPING_FILE, LABEL_MAPPING_DATASET,
    LABEL_MAPPING_LEVEL0_COL, METADATA_COLUMNS, classify_column,
    ROI_ID_COLUMN,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.io import save_csv, save_json, load_dataframe, load_spectra_file, save_dataframe

logger = logging.getLogger(__name__)

# Module-level store for ROI mapping — avoids pandas attribute assignment warning
_roi_mapping_store: dict = {}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_spectra(path: str | Path) -> pd.DataFrame:
    """
    Loads a spectra dataset from parquet or CSV.
    Format is detected automatically from the file extension.

    Args:
        path: Path to the data file (.parquet or .csv).

    Returns:
        DataFrame with feature columns and raw label column (pre-remapping).
    """
    return load_spectra_file(path)


# ---------------------------------------------------------------------------
# Label remapping
# ---------------------------------------------------------------------------

def load_label_mapping(
        mapping_file: str | Path = LABEL_MAPPING_FILE,
        dataset: str = LABEL_MAPPING_DATASET,
) -> pd.DataFrame:
    """
    Loads the label mapping file and filters to the specified dataset.

    Args:
        mapping_file: Path to the labelset_mapping.csv file.
        dataset:      Value to filter on in the 'labelset' column (e.g. 'pilot').

    Returns:
        Filtered DataFrame with columns: Level_0, Level_1, Level_2, Level_3.

    Raises:
        FileNotFoundError: If the mapping file does not exist.
        ValueError:        If no rows match the dataset filter or required columns are missing.
    """
    mapping_file = Path(mapping_file)
    if not mapping_file.exists():
        raise FileNotFoundError(f"Label mapping file not found: {mapping_file}")

    mapping = pd.read_csv(mapping_file)

    # Support both 'labelset' and 'dataset' as the filter column name
    filter_col = None
    for candidate in ("labelset", "dataset"):
        if candidate in mapping.columns:
            filter_col = candidate
            break
    if filter_col is None:
        raise ValueError(
            f"Mapping file must have a 'labelset' or 'dataset' column. "
            f"Found: {list(mapping.columns)}"
        )

    filtered = mapping[mapping[filter_col] == dataset].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows found for dataset='{dataset}' in column '{filter_col}'. "
            f"Available values: {mapping[filter_col].unique().tolist()}"
        )

    required_cols = [LABEL_MAPPING_LEVEL0_COL, "Level_1", "Level_2", "Level_3"]
    missing = [c for c in required_cols if c not in filtered.columns]
    if missing:
        raise ValueError(f"Mapping file is missing required columns: {missing}")

    logger.info(
        f"Loaded label mapping — dataset='{dataset}' | "
        f"{len(filtered)} label(s): {filtered[LABEL_MAPPING_LEVEL0_COL].tolist()}"
    )
    return filtered[required_cols].reset_index(drop=True)


def remap_labels(
        df: pd.DataFrame,
        mapping_file: str | Path = LABEL_MAPPING_FILE,
        dataset: str = LABEL_MAPPING_DATASET,
        raw_label_col: str = RAW_LABEL_COLUMN,
        drop_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Remaps raw labels in the dataset to the three hierarchy levels
    (Level_1, Level_2, Level_3) using the label mapping file.

    The mapping file is filtered to `dataset` rows only. Each raw label
    in `raw_label_col` is looked up in the mapping's Level_0 column and
    replaced with the corresponding Level_1, Level_2, Level_3 values,
    which are written into the pipeline label columns defined in LABEL_COLUMNS.

    Args:
        df:            Dataset DataFrame containing a raw label column.
        mapping_file:  Path to labelset_mapping.csv.
        dataset:       Dataset filter value (e.g. 'pilot').
        raw_label_col: Column in df containing the raw labels to remap.
        drop_unmapped: If True (default), drops rows whose raw label is not
                       in the mapping and logs a warning. If False, raises
                       a ValueError instead.

    Returns:
        DataFrame with three new columns added:
            label_level1, label_level2, label_level3
        The original raw label column is preserved.

    Raises:
        KeyError:   If raw_label_col does not exist in df.
        ValueError: If drop_unmapped=False and unmapped labels are found.
    """
    if raw_label_col not in df.columns:
        raise KeyError(
            f"Raw label column '{raw_label_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    mapping = load_label_mapping(mapping_file, dataset)

    # Build lookup dict: raw label -> {level: mapped_value}
    lookup = {
        row[LABEL_MAPPING_LEVEL0_COL]: {
            LABEL_COLUMNS[1]: row["Level_1"],
            LABEL_COLUMNS[2]: row["Level_2"],
            LABEL_COLUMNS[3]: row["Level_3"],
        }
        for _, row in mapping.iterrows()
    }

    # Check for unmapped raw labels
    raw_labels = df[raw_label_col].unique()
    mapped_set = set(lookup.keys())
    unmapped = [lbl for lbl in raw_labels if lbl not in mapped_set]

    if unmapped:
        msg = (
            f"Found {len(unmapped)} raw label(s) not in the '{dataset}' mapping: {unmapped}. "
            f"Valid labels are: {sorted(mapped_set)}"
        )
        if drop_unmapped:
            logger.warning(msg + " — dropping affected rows.")
            df = df[df[raw_label_col].isin(mapped_set)].copy()
        else:
            raise ValueError(msg)

    # Apply remapping — only for levels present in the mapping file (1, 2, 3)
    # Level 4 is constructed separately below from Level 2 label + roi_ID
    out = df.copy()
    mapping_level_cols = [LABEL_COLUMNS[l] for l in (1, 2, 3) if l in LABEL_COLUMNS]
    for level_col in mapping_level_cols:
        out[level_col] = out[raw_label_col].map(
            {k: v[level_col] for k, v in lookup.items()}
        )

    # Log resulting class distribution at each level (1, 2, 3 only — level 4 logged separately)
    for level, col in LABEL_COLUMNS.items():
        if col not in out.columns:
            continue
        counts = out[col].value_counts().sort_values(ascending=False)
        total = len(out)
        logger.info(f"Level {level} ({col}) class distribution after remapping:")
        for cls, n in counts.items():
            logger.info(f"    {cls:<30} {n:>8,}  ({n / total * 100:.2f}%)")

    # Construct Level 4 label: {level2_label}_ROI_{running_count}
    # Running count restarts from 001 for each Level 2 class, so labels look like:
    #   turf_algae_ROI_001, turf_algae_ROI_002, ..., coral_ROI_001, coral_ROI_002, ...
    # This makes labels human-readable and comparable within the Level 2 hierarchy.
    # The mapping from original roi_ID to the new label is exported as a CSV.
    level1_col = LABEL_COLUMNS[1]
    level2_col = LABEL_COLUMNS[2]
    level4_col = LABEL_COLUMNS[4]

    if ROI_ID_COLUMN in out.columns:
        # Build roi_ID -> level4_label mapping
        # Group by Level 2 class, assign running count per ROI within that class
        roi_map = (
            out[[level2_col, ROI_ID_COLUMN]]
            .drop_duplicates()
            .sort_values([level2_col, ROI_ID_COLUMN])
        )
        roi_map["_rank"] = (
            roi_map.groupby(level2_col)[ROI_ID_COLUMN]
            .transform(lambda x: (pd.factorize(x)[0] + 1))
        )
        roi_map[level4_col] = (
                roi_map[level2_col].astype(str)
                + "_ROI_"
                + roi_map["_rank"].apply(lambda n: f"{n:03d}")
        )
        roi_map = roi_map.drop(columns="_rank")

        # Also attach Level 1 for the full mapping table
        level1_per_roi = (
            out[[ROI_ID_COLUMN, level1_col]]
            .drop_duplicates()
        )
        roi_map = roi_map.merge(level1_per_roi, on=ROI_ID_COLUMN, how="left")

        # Apply to DataFrame
        id_to_label = dict(zip(roi_map[ROI_ID_COLUMN], roi_map[level4_col]))
        out[level4_col] = out[ROI_ID_COLUMN].map(id_to_label)

        n_rois = out[level4_col].nunique()
        logger.info(
            f"Level 4 labels constructed — {n_rois} unique ROIs "
            f"(format: {{level2_label}}_ROI_NNN)"
        )

        # Log ROI count per Level 2 class
        roi_per_class = out.groupby(level2_col)[level4_col].nunique()
        for cls, n in roi_per_class.sort_index().items():
            logger.info(f"    {cls:<30} {n:>4} ROIs")

        # Store mapping table in module-level dict for export by the caller
        # Columns: roi_ID, label_level1, label_level2, label_level4
        _roi_mapping_store['current'] = roi_map[[
            ROI_ID_COLUMN, level1_col, level2_col, level4_col
        ]].sort_values([level2_col, level4_col]).reset_index(drop=True)

    else:
        logger.warning(
            f"Column '{ROI_ID_COLUMN}' not found — Level 4 labels not constructed. "
            f"Check ROI_ID_COLUMN in config.py."
        )

    logger.info(
        f"Label remapping complete — {len(out):,} rows retained "
        f"({'all' if not unmapped else f'{len(df) - len(out):,} dropped'})"
    )
    return out


def validate_mapping(
        df: pd.DataFrame,
        mapping_file: str | Path = LABEL_MAPPING_FILE,
        dataset: str = LABEL_MAPPING_DATASET,
        raw_label_col: str = RAW_LABEL_COLUMN,
) -> dict:
    """
    Dry-run validation of the label mapping against the dataset.
    Reports which raw labels are mapped, which are missing, and what
    the resulting class distributions will look like — without modifying df.

    Useful to run before the full pipeline to catch label mismatches early.

    Args:
        df:            Dataset DataFrame.
        mapping_file:  Path to labelset_mapping.csv.
        dataset:       Dataset filter value.
        raw_label_col: Raw label column name in df.

    Returns:
        Dict with keys: 'mapped', 'unmapped', 'level_distributions'.
    """
    mapping = load_label_mapping(mapping_file, dataset)
    mapped_set = set(mapping[LABEL_MAPPING_LEVEL0_COL].tolist())
    raw_labels = df[raw_label_col].unique().tolist()
    mapped = sorted([l for l in raw_labels if l in mapped_set])
    unmapped = sorted([l for l in raw_labels if l not in mapped_set])

    print(f"\n{'=' * 55}")
    print(f"Label mapping validation — dataset: '{dataset}'")
    print(f"{'=' * 55}")
    print(f"Raw labels in dataset : {len(raw_labels)}")
    print(f"  Mapped              : {len(mapped)}  {mapped}")
    print(f"  Unmapped (no entry) : {len(unmapped)}  {unmapped if unmapped else 'none'}")

    # Simulate distributions
    lookup = {row[LABEL_MAPPING_LEVEL0_COL]: row for _, row in mapping.iterrows()}
    level_distributions = {}
    for level_col in ["Level_1", "Level_2", "Level_3"]:
        pipeline_col = LABEL_COLUMNS[{"Level_1": 1, "Level_2": 2, "Level_3": 3}[level_col]]
        simulated = df[raw_label_col].map(
            {k: v[level_col] for k, v in lookup.items()}
        ).dropna()
        dist = simulated.value_counts()
        level_distributions[level_col] = dist.to_dict()
        total = dist.sum()
        print(f"\n  {level_col} ({pipeline_col}):")
        for cls, n in dist.items():
            print(f"    {cls:<30} {n:>8,}  ({n / total * 100:.2f}%)")

    print(f"{'=' * 55}\n")
    return {"mapped": mapped, "unmapped": unmapped, "level_distributions": level_distributions}


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_data(
        df: pd.DataFrame,
        level: int,
        cfg: SplitConfig = SPLIT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split.
    Stratification uses cfg.stratify_level (finest level) to ensure rare
    classes appear proportionally in all three splits.

    Args:
        df:    Full dataset (must already have label columns from remap_labels()).
        level: Target label level for logging class distributions.
        cfg:   SplitConfig instance.

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(cfg.train_frac + cfg.val_frac + cfg.test_frac - 1.0) < 1e-6, \
        "Split fractions must sum to 1.0"

    stratify_col = LABEL_COLUMNS[cfg.stratify_level]
    label_col = LABEL_COLUMNS[level]

    logger.info(
        f"Splitting — level {level} | "
        f"train {cfg.train_frac:.0%} / val {cfg.val_frac:.0%} / test {cfg.test_frac:.0%} | "
        f"stratify on: {stratify_col}"
    )

    val_test_frac = cfg.val_frac + cfg.test_frac
    train_df, valtest_df = train_test_split(
        df,
        test_size=val_test_frac,
        stratify=df[stratify_col],
        random_state=cfg.random_seed,
    )

    relative_test_frac = cfg.test_frac / val_test_frac
    val_df, test_df = train_test_split(
        valtest_df,
        test_size=relative_test_frac,
        stratify=valtest_df[stratify_col],
        random_state=cfg.random_seed,
    )

    logger.info(
        f"Split sizes — train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}"
    )
    for split_name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        _log_class_distribution(sdf, label_col, split_name)

    return train_df, val_df, test_df


def _log_class_distribution(df: pd.DataFrame, label_col: str, split_name: str) -> None:
    counts = df[label_col].value_counts().sort_index()
    props = (counts / len(df) * 100).round(2)
    logger.info(f"  {split_name} class distribution:")
    for cls in counts.index:
        logger.info(f"    {cls:<30} {counts[cls]:,} ({props[cls]}%)")


# ---------------------------------------------------------------------------
# Feature / label extraction
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns feature column names by classifying each column using the naming
    conventions defined in config.py (spectral, glcm, sdiv).

    Excludes: metadata columns (roi_ID, exposure, n_valid_pixels, scan_ID,
    dataset, label, line, sample), remapped label columns, and any column
    classified as 'unknown' (logged as a warning).

    Args:
        df: Full dataset DataFrame (after remapping).

    Returns:
        List of feature column names (spectral + glcm + sdiv only).
    """
    feature_cols = []
    unknown_cols = []
    for col in df.columns:
        family = classify_column(col)
        if family in ("spectral", "glcm", "sdiv"):
            feature_cols.append(col)
        elif family == "unknown":
            unknown_cols.append(col)
        # metadata and label are silently excluded

    if unknown_cols:
        logger.warning(
            f"{len(unknown_cols)} column(s) did not match any known naming convention "
            f"and were excluded from features: {unknown_cols}"
        )

    logger.info(
        f"Feature columns identified: {len(feature_cols)} total "
        f"({sum(1 for c in feature_cols if classify_column(c) == 'spectral')} spectral, "
        f"{sum(1 for c in feature_cols if classify_column(c) == 'glcm')} glcm, "
        f"{sum(1 for c in feature_cols if classify_column(c) == 'sdiv')} sdiv)"
    )
    return feature_cols


def encode_labels(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Integer-encodes string class labels using a LabelEncoder fit on train only.

    Args:
        train_df, val_df, test_df: Split DataFrames.
        level: Label hierarchy level.

    Returns:
        (y_train, y_val, y_test, fitted LabelEncoder)
    """
    label_col = LABEL_COLUMNS[level]
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col])
    y_val = le.transform(val_df[label_col])
    y_test = le.transform(test_df[label_col])
    logger.info(f"Label encoding level {level} — classes: {list(le.classes_)}")
    return y_train, y_val, y_test, le


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Computes per-sample weights as inverse class frequency, normalised so the
    mean weight equals 1.0 (avoids inflating the effective learning rate).

    Args:
        y: Integer-encoded label array.

    Returns:
        Array of per-sample weights, same length as y.
    """
    classes, counts = np.unique(y, return_counts=True)
    freq = counts / len(y)
    inv_freq = 1.0 / freq
    inv_freq /= inv_freq.mean()
    weight_map = dict(zip(classes, inv_freq))
    weights = np.array([weight_map[label] for label in y])
    logger.info(f"Class weights: { {c: round(w, 3) for c, w in weight_map.items()} }")
    return weights


# ---------------------------------------------------------------------------
# DMatrix creation
# ---------------------------------------------------------------------------

def make_dmatrix(
        df: pd.DataFrame,
        feature_cols: list[str],
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        ref: xgb.QuantileDMatrix | None = None,
) -> xgb.DMatrix | xgb.QuantileDMatrix:
    """
    Creates a DMatrix or QuantileDMatrix from a DataFrame.

    XGBoost requires that val/test QuantileDMatrices are built with ref=dtrain
    so they share the same quantile cuts. Omitting ref raises:
      ValueError: Training dataset should be used as a reference...

    Args:
        df:            DataFrame containing feature columns.
        feature_cols:  Feature column names to use.
        y:             Integer-encoded label array.
        sample_weight: Optional per-sample weights (training only).
        ref:           Reference QuantileDMatrix for val/test sets.
                       None  -> new QuantileDMatrix (use for train).
                       False -> plain DMatrix (use for SHAP subsamples).
                       dtrain -> referenced QuantileDMatrix (use for val/test).

    Returns:
        QuantileDMatrix (train/val/test) or DMatrix (SHAP subsamples).
    """
    X = df[feature_cols].values

    if ref is False:
        dm = xgb.DMatrix(X, label=y, feature_names=feature_cols)
        label = "DMatrix"
    elif ref is None:
        dm = xgb.QuantileDMatrix(X, label=y, weight=sample_weight,
                                 feature_names=feature_cols)
        label = "QuantileDMatrix (train)"
    else:
        dm = xgb.QuantileDMatrix(X, label=y, ref=ref, feature_names=feature_cols)
        label = "QuantileDMatrix (ref)"

    logger.info(f"Created {label} — shape: ({dm.num_row():,}, {dm.num_col()})")
    return dm


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_roi_mapping(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Exports the ROI mapping table produced by remap_labels() to CSV.
    Columns: roi_ID, label_level1, label_level2, label_level4

    This table is the key reference for interpreting Level 4 model outputs —
    it links the human-readable running-count label back to the original roi_ID
    and its position in the Level 2 hierarchy.

    Args:
        df:      DataFrame returned by remap_labels() (must have _roi_mapping attr).
        out_dir: Output directory — saved as roi_label_mapping.csv.
    """
    mapping = _roi_mapping_store.get("current")
    if mapping is None:
        logger.warning("No ROI mapping found — was remap_labels() called?")
        return
    save_csv(mapping, out_dir / "roi_label_mapping.csv", index=False)
    logger.info(f"ROI label mapping exported: {out_dir / 'roi_label_mapping.csv'} "
                f"({len(mapping)} ROIs)")


def save_split_metadata(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        le: LabelEncoder,
        level: int,
        out_dir: Path,
) -> None:
    """
    Saves split sizes and class mapping to JSON for reproducibility.
    """
    label_col = LABEL_COLUMNS[level]
    meta = {
        "level": level,
        "label_column": label_col,
        "class_mapping": {str(i): cls for i, cls in enumerate(le.classes_)},
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "train_class_counts": train_df[label_col].value_counts().to_dict(),
        "val_class_counts": val_df[label_col].value_counts().to_dict(),
        "test_class_counts": test_df[label_col].value_counts().to_dict(),
    }
    save_json(meta, out_dir / "split_metadata.json")


# ---------------------------------------------------------------------------
# ROI-level subsampling
# ---------------------------------------------------------------------------

def subsample_turf_rois(
        df: pd.DataFrame,
        turf_algae_class: str = "turf_algae",
        random_seed: int = 42,
        spectra: str | None = None,
) -> pd.DataFrame:
    """
    Reduces turf algae ROI count to match the next most abundant Level 2 class.
    Subsampling is at the ROI level — all rows belonging to selected ROIs are kept.
    This is a one-time fixed subsample (reproducible via random_seed).

    The ceiling is determined by the largest non-turf Level 2 class ROI count,
    making the subsampling decision data-driven rather than arbitrary.

    Held-out turf ROIs are saved to data/turf_held_out_seed{seed}_spectra{X}.parquet.
    Spectra label is included in the filename to avoid overwriting when processing
    multiple spectra types.

    Args:
        df:               DataFrame after remap_labels() has been called.
        turf_algae_class: Level 2 label for turf algae.
        random_seed:      Random seed for reproducibility.
        spectra:          Spectra type label e.g. 'A' — included in held-out filename.
                          If None, no spectra suffix is added (not recommended when
                          processing multiple spectra types).

    Returns:
        Subsampled DataFrame with turf algae ROIs reduced to match ceiling.
    """
    level2_col = LABEL_COLUMNS[2]
    level4_col = LABEL_COLUMNS[4]

    # Use roi_ID directly if Level 4 not yet constructed
    roi_col = level4_col if level4_col in df.columns else ROI_ID_COLUMN

    if roi_col not in df.columns:
        raise KeyError(f"ROI column '{roi_col}' not found. Run remap_labels() first.")

    # Count ROIs per Level 2 class
    roi_counts = (
        df.groupby(level2_col)[roi_col]
        .nunique()
        .sort_values(ascending=False)
    )
    logger.info("ROI counts per Level 2 class (before subsampling):")
    for cls, n in roi_counts.items():
        logger.info(f"    {cls:<30} {n:>4} ROIs")

    if turf_algae_class not in roi_counts.index:
        logger.warning(f"'{turf_algae_class}' not found in Level 2 labels — no subsampling applied.")
        return df

    # Ceiling = largest non-turf ROI count
    non_turf_counts = roi_counts.drop(turf_algae_class)
    ceiling = int(non_turf_counts.max())
    n_turf_rois = int(roi_counts[turf_algae_class])

    if n_turf_rois <= ceiling:
        logger.info(
            f"Turf algae ROI count ({n_turf_rois}) already at or below ceiling ({ceiling}) "
            f"— no subsampling needed."
        )
        return df

    # Sample ROIs to keep
    turf_rois = (
        df[df[level2_col] == turf_algae_class][roi_col]
        .unique()
    )
    rng = np.random.default_rng(random_seed)
    selected_rois = rng.choice(turf_rois, size=ceiling, replace=False)
    selected_set = set(selected_rois)

    # Keep all non-turf rows + only selected turf ROI rows
    mask = (df[level2_col] != turf_algae_class) | (df[roi_col].isin(selected_set))
    out = df[mask].copy()
    held_out = df[~mask].copy()  # Unselected turf ROIs — saved as unseen data

    logger.info(
        f"Turf algae ROIs subsampled: {n_turf_rois} -> {ceiling} "
        f"(ceiling = largest non-turf class: '{non_turf_counts.idxmax()}')"
    )
    logger.info(f"Rows retained : {len(out):,} of {len(df):,} ({len(out) / len(df) * 100:.1f}%)")
    logger.info(f"Rows held out : {len(held_out):,} ({len(held_out) / len(df) * 100:.1f}%) — unselected turf ROIs")

    # Save held-out turf ROIs to data/ for use with predict.py
    if len(held_out) > 0:
        from config.config import DATA_DIR
        spectra_suffix = f"_spectra{spectra}" if spectra is not None else ""
        held_out_path = Path(DATA_DIR) / f"turf_held_out_seed{random_seed}{spectra_suffix}.parquet"
        held_out.to_parquet(held_out_path, index=False)
        logger.info(f"Held-out turf ROIs saved: {held_out_path}")

    return out