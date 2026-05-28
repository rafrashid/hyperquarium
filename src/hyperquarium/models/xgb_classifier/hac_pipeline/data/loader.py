"""
hac_pipeline/data/loader.py
----------------------------
Load parquet, remap labels, filter to turf_algae, drop high-NaN columns,
drop NaN rows, stratified sample, return feature column list.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_LABEL_COLUMN = "label"
LABEL_MAPPING_LEVEL0_COL = "Level_0"
LABEL_LEVEL3_COL = "label_level3"
LABEL_LEVEL4_COL = "label_level4"
ROI_ID_COL = "roi_ID"
TURF_LEVEL3_VALUE = "turf_algae"

# Actual column names in labelset_mapping.csv -> internal names
_MAPPING_LEVEL_COLS = {
    "Level_1": "label_level1",
    "Level_2": "label_level2",
    "Level_3": "label_level3",
}

# Metadata columns — excluded from NaN dropping and feature matrix
_META_COLS = {
    ROI_ID_COL, "scan_ID", "dataset", "exposure", "n_valid_pixels",
    "line", "sample", RAW_LABEL_COLUMN,
    "label_level1", "label_level2", "label_level3", "label_level4",
}


def remap_labels(
        df: pd.DataFrame,
        mapping_path: Path,
        labelset: str = "pilot",
) -> pd.DataFrame:
    """Map raw labels to label_level1/2/3 and construct label_level4 from scratch."""
    mapping = pd.read_csv(mapping_path)
    mapping = mapping[mapping["labelset"] == labelset].copy()
    if mapping.empty:
        raise ValueError(
            f"No rows for labelset='{labelset}' in {mapping_path}. "
            f"Available: {pd.read_csv(mapping_path)['labelset'].unique().tolist()}"
        )

    mapping = mapping.rename(columns=_MAPPING_LEVEL_COLS)
    lookup = mapping.set_index(LABEL_MAPPING_LEVEL0_COL)[
        ["label_level1", "label_level2", "label_level3"]
    ].to_dict(orient="index")

    before = len(df)
    mapped_rows = df[RAW_LABEL_COLUMN].map(lookup)
    unmapped = mapped_rows.isna()
    if unmapped.any():
        logger.warning(
            f"Dropping {unmapped.sum()} rows with unmapped labels: "
            f"{df.loc[unmapped, RAW_LABEL_COLUMN].unique().tolist()}"
        )
    df = df[~unmapped].copy()
    mapped_df = pd.json_normalize(mapped_rows[~unmapped])
    mapped_df.index = df.index
    df = pd.concat([df, mapped_df], axis=1)

    # Construct label_level4: label_level2 + running ROI count per Level_2 class
    df = df.sort_values(ROI_ID_COL).copy()
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
        f"Label remapping: {before} → {len(df)} rows. "
        f"Level 3: {df[LABEL_LEVEL3_COL].nunique()} classes. "
        f"Level 4: {df[LABEL_LEVEL4_COL].nunique()} ROIs."
    )
    return df


def load_turf_sample(
        parquet_path: Path,
        mapping_path: Path,
        pixels_per_roi: int,
        random_seed: int,
        nan_col_threshold: float,
        labelset: str = "pilot",
) -> tuple[pd.DataFrame, list[str]]:
    """Load, filter, clean, and sample turf_algae pixels.

    Returns
    -------
    df_sample : pd.DataFrame
        Sampled turf pixels with feature and metadata columns.
    feature_cols : list[str]
        Ordered list of numeric feature columns surviving NaN removal.
        These are the columns passed to PCA — all valid features, no SHAP pre-selection.
    nan_fractions : dict[str, float]
        NaN fraction per column before dropping, for audit logging.
    """
    logger.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns.")

    df = remap_labels(df, mapping_path=mapping_path, labelset=labelset)

    # Filter to turf_algae
    turf_mask = df[LABEL_LEVEL3_COL] == TURF_LEVEL3_VALUE
    df_turf = df[turf_mask].copy()
    logger.info(
        f"Turf pixels: {len(df_turf):,} ({turf_mask.mean() * 100:.1f}% of total). "
        f"ROIs: {df_turf[ROI_ID_COL].nunique()}."
    )
    if df_turf.empty:
        raise ValueError("No turf_algae pixels found after label remapping.")

    # Identify feature columns (all numeric, non-metadata)
    feat_cols = [
        c for c in df_turf.columns
        if c not in _META_COLS and pd.api.types.is_numeric_dtype(df_turf[c])
    ]

    # Compute NaN fractions before dropping (for audit)
    nan_fractions = {
        c: float(df_turf[c].isna().mean()) for c in feat_cols
    }

    # Drop high-NaN feature columns
    high_nan = [c for c in feat_cols if nan_fractions[c] > nan_col_threshold]
    if high_nan:
        logger.warning(
            f"Dropping {len(high_nan)} feature columns with >{nan_col_threshold * 100:.0f}% NaN "
            f"(large window/plot sizes). First 5: {high_nan[:5]}"
        )
        df_turf = df_turf.drop(columns=high_nan)
        feat_cols = [c for c in feat_cols if c not in high_nan]

    # Drop rows with any NaN in remaining feature columns
    n_before = len(df_turf)
    nan_row_mask = df_turf[feat_cols].isna().any(axis=1)
    df_turf = df_turf[~nan_row_mask]
    n_dropped = n_before - len(df_turf)
    if n_dropped:
        logger.warning(
            f"Dropped {n_dropped:,} rows with NaN in feature columns "
            f"({n_dropped / n_before * 100:.1f}% of turf pixels)."
        )

    if df_turf.empty:
        raise ValueError(
            "No turf_algae pixels remain after NaN removal. "
            "Check feature extraction for systematic missing data."
        )

    logger.info(
        f"Clean turf pixels: {len(df_turf):,} across "
        f"{df_turf[ROI_ID_COL].nunique()} ROIs. "
        f"Feature columns: {len(feat_cols)}."
    )

    # Stratified sample
    df_sample = _stratified_sample(df_turf, roi_col=ROI_ID_COL,
                                   n=pixels_per_roi, seed=random_seed)
    logger.info(
        f"Stratified sample: {len(df_sample):,} pixels across "
        f"{df_sample[ROI_ID_COL].nunique()} ROIs."
    )
    return df_sample, feat_cols, nan_fractions


def _stratified_sample(
        df: pd.DataFrame,
        roi_col: str,
        n: int,
        seed: int,
) -> pd.DataFrame:
    frames = []
    for roi_id, group in df.groupby(roi_col):
        if len(group) <= n:
            logger.warning(
                f"ROI '{roi_id}' has only {len(group)} pixels (< {n}); "
                f"including all."
            )
            frames.append(group)
        else:
            frames.append(group.sample(n=n, random_state=seed, replace=False))
    return pd.concat(frames, ignore_index=True)
