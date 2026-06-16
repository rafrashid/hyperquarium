"""
data/labels.py — apply labelset_mapping.csv to the raw `label` column.

Mirrors the label-mapping convention shared by train.py / loader.py / compile_features:
- labelset_mapping.csv has a `labelset` column; remap_labels() filters to ONE labelset
  (e.g. "pilot" or "reefcompare") before building the lookup — same as the main pipeline.
- Level_0 is the inclusion filter (raw labels absent from the filtered mapping are dropped).
- Level_1..Level_3 give the hierarchical remapping.

Convention match: train.py calls `remap_labels(df, dataset=args.labelset)`; this module
exposes the same `dataset=` parameter so the bayes pipeline behaves identically.
"""
from __future__ import annotations

import pandas as pd
from bayes_pipeline.config.config import LabelConfig


def load_mapping(path, cfg: LabelConfig, dataset: str | None = None) -> pd.DataFrame:
    """
    Load labelset_mapping.csv and (if the CSV has a `labelset` column) filter to one
    labelset. `dataset` defaults to cfg.labelset when None.
    """
    mapping = pd.read_csv(path)
    if cfg.mapping_raw_col not in mapping.columns:
        raise ValueError(
            f"Mapping CSV missing '{cfg.mapping_raw_col}' (the inclusion-filter column). "
            f"Found: {list(mapping.columns)}"
        )

    dataset = dataset if dataset is not None else cfg.labelset
    if cfg.labelset_col in mapping.columns:
        before = len(mapping)
        mapping = mapping[mapping[cfg.labelset_col] == dataset].copy()
        if mapping.empty:
            available = pd.read_csv(path)[cfg.labelset_col].unique().tolist()
            raise ValueError(
                f"No rows in mapping for labelset '{dataset}'. "
                f"Available: {available}"
            )
        # (silent on count; caller logs)
    # If no labelset column, the CSV is single-labelset — use as-is.
    return mapping


def remap_labels(df: pd.DataFrame, cfg: LabelConfig, mapping_path,
                 dataset: str | None = None) -> pd.DataFrame:
    """
    Add class_L1 / class_L2 / class_L3 to df by joining on the raw label, after
    filtering the mapping to one labelset.

    Signature mirrors the main pipeline's intent: pass the labelset via `dataset`
    (defaults to cfg.labelset). Rows whose raw label is absent from the filtered
    mapping's Level_0 are dropped (inclusion filter). Returns a new DataFrame.
    """
    raw = cfg.raw_label_col
    if raw not in df.columns:
        raise ValueError(f"Input frame missing raw label column '{raw}'.")

    mapping = load_mapping(mapping_path, cfg, dataset=dataset)

    level_cols = [cfg.level_col_fmt.format(level=l) for l in cfg.levels]
    missing = [c for c in level_cols if c not in mapping.columns]
    if missing:
        raise ValueError(f"Mapping CSV missing level columns: {missing}")

    keep = mapping[[cfg.mapping_raw_col, *level_cols]].copy()
    keep = keep.rename(columns={cfg.mapping_raw_col: raw})

    merged = df.merge(keep, on=raw, how="inner")  # inner join = inclusion filter

    rename = {cfg.level_col_fmt.format(level=l): f"class_L{l}" for l in cfg.levels}
    merged = merged.rename(columns=rename)
    return merged
