#!/usr/bin/env python3
"""
scripts/aggregate.py — standalone aggregation stage (the heavy, big-RAM step).

Run once per dataset before fitting. Reads the compiled parquet, applies the
labelset mapping, collapses to one row per ROI, and caches roi_summary.parquet.

Usage:
    python scripts/aggregate.py \
        --parquet data/compiled_dataset.parquet \
        --mapping data/labelset_mapping.csv \
        --output  outputs/bayes \
        [--glcm-window 25] [--specdiv-plot 25] [--force]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config.config import CONFIG
from data.aggregate import aggregate_to_roi
from utils.logger import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/bayes"))
    ap.add_argument("--labelset", type=str, default="reefcompare",
                    help="Labelset to filter labelset_mapping.csv (default: reefcompare)")
    ap.add_argument("--glcm-window", type=int, default=None)
    ap.add_argument("--specdiv-plot", type=int, default=None)
    ap.add_argument("--include-gamma", action="store_true",
                    help="Include gamma feature (parquet must have sdiv_gamma_plot_*).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = CONFIG
    cfg.paths.compiled_parquet = args.parquet
    cfg.paths.labelset_mapping = args.mapping
    cfg.paths.output_root = args.output
    cfg.labels.labelset = args.labelset
    if args.glcm_window is not None:
        cfg.features.glcm_window = args.glcm_window
    if args.specdiv_plot is not None:
        cfg.features.specdiv_plot = args.specdiv_plot
    if args.include_gamma:
        cfg.features.include_gamma = True

    logger = get_logger(logfile=args.output / "aggregate.log")
    if cfg.paths.roi_summary.exists() and not args.force:
        logger.info("ROI summary already exists (use --force to rebuild): %s",
                    cfg.paths.roi_summary)
        return
    aggregate_to_roi(cfg, dataset=args.labelset, logger=logger)


if __name__ == "__main__":
    main()
