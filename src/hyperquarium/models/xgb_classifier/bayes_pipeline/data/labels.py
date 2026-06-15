"""
data/labels.py — apply labelset_mapping.csv to the raw `label` column.

Mirrors the label-mapping convention shared by compile_features / plot_summary /
loader: Level_0 is the inclusion filter (raw labels absent from the mapping are
dropped), and Level_1..Level_3 give the hierarchical remapping.

L3 here is binary turf vs non-turf. If the mapping already encodes that, it is used
as-is; otherwise it is derived from the turf class name as a fallback.
"""
from __future__ import annotations

import pandas as pd
from bayes_pipeline.config.config import LabelConfig


def load_mapping(path, cfg: LabelConfig) -> pd.DataFrame:
    """Load the labelset mapping CSV. Expects Level_0 + Level_1..Level_3 columns."""
    mapping = pd.read_csv(path)
    if cfg.mapping_raw_col not in mapping.columns:
        raise ValueError(
            f"Mapping CSV missing '{cfg.mapping_raw_col}' (the inclusion-filter column). "
            f"Found: {list(mapping.columns)}"
        )
    return mapping


def remap_labels(df: pd.DataFrame, mapping: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    """
    Add class_L1 / class_L2 / class_L3 columns to df by joining on the raw label.

    Rows whose raw label is not present in the mapping's Level_0 column are dropped
    (Level_0 = inclusion filter). Returns a new DataFrame.
    """
    raw = cfg.raw_label_col
    if raw not in df.columns:
        raise ValueError(f"Input frame missing raw label column '{raw}'.")

    level_cols = [cfg.level_col_fmt.format(level=l) for l in cfg.levels]
    missing = [c for c in level_cols if c not in mapping.columns]
    if missing:
        raise ValueError(f"Mapping CSV missing level columns: {missing}")

    keep = mapping[[cfg.mapping_raw_col, *level_cols]].copy()
    keep = keep.rename(columns={cfg.mapping_raw_col: raw})

    merged = df.merge(keep, on=raw, how="inner")  # inner join = inclusion filter

    # Rename Level_k -> class_Lk for downstream clarity.
    rename = {cfg.level_col_fmt.format(level=l): f"class_L{l}" for l in cfg.levels}
    merged = merged.rename(columns=rename)
    return merged
