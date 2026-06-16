"""
data/aggregate.py — the once-per-run heavy stage.

Reads the compiled pixel-level parquet (2.6M / 26M rows), applies the labelset
mapping, and collapses each of the 7 chosen feature columns to ONE row per ROI:

    roi_ID | class_L1 | class_L2 | class_L3 | <feat>_mean | <feat>_se | <feat>_n  (×7)

This is the only stage that touches millions of rows. Its output (roi_summary.parquet)
is cached so all 21 downstream fits reuse it without re-aggregating.

Because every pixel in an ROI shares one label, the class columns are constant within
ROI and travel for free via `first()`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from bayes_pipeline.config.config import Config
from bayes_pipeline.data.labels import remap_labels
from bayes_pipeline.utils.io import save_parquet
from bayes_pipeline.utils.logger import get_logger


def _resolve_feature_columns(df: pd.DataFrame, cfg: Config, logger) -> dict[str, str]:
    """Map short feature name -> actual column, checking presence in the parquet."""
    wanted = cfg.features.feature_columns()
    present, missing = {}, {}
    for short, col in wanted.items():
        if col in df.columns:
            present[short] = col
        else:
            missing[short] = col
    if missing:
        raise ValueError(
            "These feature columns are not in the parquet (check glcm_window / "
            f"specdiv_plot in FeatureConfig): {missing}\n"
            f"Available sample: {[c for c in df.columns if 'window' in c or 'plot' in c][:12]}"
        )
    logger.info("Resolved %d feature columns: %s", len(present), present)
    return present


def aggregate_to_roi(cfg: Config, dataset: str | None = None, logger=None) -> pd.DataFrame:
    """
    Run the full aggregation and cache to cfg.paths.roi_summary.
    Returns the ROI-level summary DataFrame.

    `dataset` selects the labelset (defaults to cfg.labels.labelset), mirroring
    train.py's --labelset. labelset_mapping.csv is filtered to this labelset.
    """
    logger = logger or get_logger()
    paths, labels = cfg.paths, cfg.labels
    dataset = dataset if dataset is not None else labels.labelset

    logger.info("Loading compiled parquet: %s", paths.compiled_parquet)
    feat_cols_wanted = list(cfg.features.feature_columns().values())
    meta_needed = [labels.raw_label_col, "roi_ID"]
    # Read only the columns we need — keeps the big read as light as possible.
    cols_to_read = meta_needed + feat_cols_wanted
    df = pd.read_parquet(paths.compiled_parquet, columns=cols_to_read, engine="pyarrow")
    logger.info("Loaded %d pixel rows.", len(df))

    logger.info("Applying labelset mapping (labelset=%s): %s",
                dataset, paths.labelset_mapping)
    df = remap_labels(df, labels, paths.labelset_mapping, dataset=dataset)
    logger.info("After inclusion filter: %d rows, %d ROIs.",
                len(df), df["roi_ID"].nunique())

    feat_map = _resolve_feature_columns(df, cfg, logger)
    class_cols = [f"class_L{l}" for l in labels.levels]

    logger.info("Aggregating to one row per ROI ...")
    grouped = df.groupby("roi_ID", sort=False)

    # Class columns are constant within ROI -> first().
    summary = grouped[class_cols].first()

    # Per-feature mean, se, n. se = std / sqrt(n) with ddof=1.
    for short, col in feat_map.items():
        g = grouped[col]
        mean = g.mean()
        std = g.std(ddof=1)
        n = g.count()
        se = std / np.sqrt(n)
        summary[f"{short}_mean"] = mean
        summary[f"{short}_se"] = se
        summary[f"{short}_n"] = n

    summary = summary.reset_index()
    logger.info("ROI summary: %d ROIs × %d cols.", len(summary), summary.shape[1])

    save_parquet(summary, paths.roi_summary)
    logger.info("Cached ROI summary -> %s", paths.roi_summary)
    return summary


def load_or_build_summary(cfg: Config, dataset: str | None = None,
                          force: bool = False, logger=None) -> pd.DataFrame:
    """Load cached roi_summary.parquet if present, else build it (for `dataset`)."""
    logger = logger or get_logger()
    if cfg.paths.roi_summary.exists() and not force:
        logger.info("Loading cached ROI summary: %s", cfg.paths.roi_summary)
        return pd.read_parquet(cfg.paths.roi_summary, engine="pyarrow")
    return aggregate_to_roi(cfg, dataset=dataset, logger=logger)
