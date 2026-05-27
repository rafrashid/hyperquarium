"""
hac_pipeline/data/loader.py
----------------------------
Load the compiled parquet, apply label remapping, filter to turf_algae pixels,
and draw a stratified random sample of pixels_per_roi rows per ROI.

Mirrors the remap_labels() convention from xgb_pipeline/data/loader.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label remapping
# ---------------------------------------------------------------------------

RAW_LABEL_COLUMN = "label"
LABEL_MAPPING_LEVEL0_COL = "Level_0"
LABEL_LEVEL3_COL = "label_level3"
LABEL_LEVEL4_COL = "label_level4"
ROI_ID_COL = "roi_ID"

# Actual column names in labelset_mapping.csv -> internal names used throughout pipeline
_MAPPING_LEVEL_COLS = {
    "Level_1": "label_level1",
    "Level_2": "label_level2",
    "Level_3": "label_level3",
}

TURF_LEVEL3_VALUE = "turf_algae"


def remap_labels(
        df: pd.DataFrame,
        mapping_path: Path,
        labelset: str = "pilot",
) -> pd.DataFrame:
    """Apply label hierarchy mapping to raw label column.

    Loads labelset_mapping.csv (columns: labelset, Level_0, Level_1, Level_2, Level_3),
    filters to the given labelset, and maps raw labels to label_level1/2/3.
    Constructs label_level4 as label_level2 + running ROI count per Level_2 class,
    consistent with xgb_pipeline convention.

    Rows with unmapped labels are dropped with a warning.
    """
    mapping = pd.read_csv(mapping_path)
    mapping = mapping[mapping["labelset"] == labelset].copy()

    if mapping.empty:
        raise ValueError(
            f"No rows found in {mapping_path} for labelset='{labelset}'. "
            f"Available labelsets: {pd.read_csv(mapping_path)['labelset'].unique().tolist()}"
        )

    # Rename Level_1/2/3 -> label_level1/2/3
    mapping = mapping.rename(columns=_MAPPING_LEVEL_COLS)

    lookup = mapping.set_index(LABEL_MAPPING_LEVEL0_COL)[
        ["label_level1", "label_level2", "label_level3"]
    ].to_dict(orient="index")

    before = len(df)
    mapped_rows = df[RAW_LABEL_COLUMN].map(lookup)
    unmapped = mapped_rows.isna()
    if unmapped.any():
        dropped_labels = df.loc[unmapped, RAW_LABEL_COLUMN].unique().tolist()
        logger.warning(
            f"Dropping {unmapped.sum()} rows with unmapped labels: {dropped_labels}"
        )
    df = df[~unmapped].copy()
    mapped_df = pd.json_normalize(mapped_rows[~unmapped])
    mapped_df.index = df.index
    df = pd.concat([df, mapped_df], axis=1)

    # Construct label_level4: label_level2 + running ROI count per Level_2 class
    # Sort by roi_ID for deterministic ordering
    df = df.sort_values([ROI_ID_COL]).copy()
    roi_counts: dict[str, dict[str, int]] = {}
    level4_labels = []
    for _, row in df.iterrows():
        l2 = row["label_level2"]
        roi = row[ROI_ID_COL]
        if l2 not in roi_counts:
            roi_counts[l2] = {}
        if roi not in roi_counts[l2]:
            roi_counts[l2][roi] = len(roi_counts[l2]) + 1
        level4_labels.append(f"{l2}_ROI_{roi_counts[l2][roi]:03d}")
    df[LABEL_LEVEL4_COL] = level4_labels

    logger.info(
        f"Label remapping complete: {before} → {len(df)} rows "
        f"({before - len(df)} dropped). "
        f"Level 3 classes: {df[LABEL_LEVEL3_COL].nunique()}. "
        f"Level 4 ROIs: {df[LABEL_LEVEL4_COL].nunique()}."
    )
    return df


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_turf_sample(
        parquet_path: Path,
        mapping_path: Path,
        pixels_per_roi: int,
        random_seed: int,
        labelset: str = "pilot",
        nan_col_threshold: float = 0.5,
) -> pd.DataFrame:
    """Load parquet, remap labels, filter to turf_algae, stratified sample.

    Parameters
    ----------
    parquet_path : Path
        Compiled pixel-level parquet (same file used by xgb_pipeline).
    mapping_path : Path
        Path to labelset_mapping.csv.
    pixels_per_roi : int
        Target number of pixels per ROI. ROIs with fewer pixels include all rows.
    random_seed : int
        Random seed for reproducible sampling.
    labelset : str
        Labelset name passed to remap_labels().

    Returns
    -------
    pd.DataFrame
        Sampled turf_algae pixels with all feature columns and metadata.
    """
    logger.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")

    df = remap_labels(df, mapping_path=mapping_path, labelset=labelset)

    # Filter to turf_algae only
    turf_mask = df[LABEL_LEVEL3_COL] == TURF_LEVEL3_VALUE
    df_turf = df[turf_mask].copy()
    logger.info(
        f"Turf algae pixels: {len(df_turf):,} "
        f"({turf_mask.mean() * 100:.1f}% of total). "
        f"ROIs: {df_turf[ROI_ID_COL].nunique()}."
    )

    if df_turf.empty:
        raise ValueError(
            f"No turf_algae pixels found after label remapping. "
            f"Check labelset='{labelset}' and label_level3 mapping."
        )

    # Drop columns where >50% of turf rows are NaN — these are large window/plot
    # sizes (GLCM ≥ window_51, specdiv ≥ plot_51) that are essentially empty.
    # Then drop remaining rows with any NaN (genuine border-effect pixels).
    nan_frac = df_turf.isna().mean()
    high_nan_cols = nan_frac[nan_frac > nan_col_threshold].index.tolist()
    if high_nan_cols:
        logger.warning(
            f"Dropping {len(high_nan_cols)} columns with >50% NaN "
            f"(large window/plot sizes): {high_nan_cols}"
        )
        df_turf = df_turf.drop(columns=high_nan_cols)

    n_rows_before = len(df_turf)
    df_turf = df_turf.dropna(axis=0)
    n_rows_dropped = n_rows_before - len(df_turf)
    if n_rows_dropped:
        logger.warning(
            f"Dropped {n_rows_dropped:,} rows with remaining NaN values "
            f"({n_rows_dropped / n_rows_before * 100:.1f}% of turf pixels)."
        )

    if df_turf.empty:
        raise ValueError(
            "No turf_algae pixels remain after NaN removal. "
            "Check feature extraction outputs for systematic missing data."
        )

    # Stratified sample
    df_sample = _stratified_sample(
        df_turf,
        roi_col=ROI_ID_COL,
        n=pixels_per_roi,
        seed=random_seed,
    )

    logger.info(
        f"Stratified sample: {len(df_sample):,} pixels across "
        f"{df_sample[ROI_ID_COL].nunique()} ROIs."
    )
    return df_sample


def _stratified_sample(
        df: pd.DataFrame,
        roi_col: str,
        n: int,
        seed: int,
) -> pd.DataFrame:
    """Sample up to n rows per ROI. ROIs with fewer than n rows include all."""
    frames = []
    short_rois = []

    for roi_id, group in df.groupby(roi_col):
        if len(group) <= n:
            short_rois.append((roi_id, len(group)))
            frames.append(group)
        else:
            frames.append(group.sample(n=n, random_state=seed, replace=False))

    if short_rois:
        for roi_id, count in short_rois:
            logger.warning(
                f"ROI '{roi_id}' has only {count} pixels (< {n}); "
                f"including all available rows."
            )

    return pd.concat(frames, ignore_index=True)