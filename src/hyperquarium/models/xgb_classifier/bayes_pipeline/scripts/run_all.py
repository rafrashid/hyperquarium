#!/usr/bin/env python3
"""
scripts/run_all.py — run all 21 cells for one dataset.

Assumes aggregation has been run (or runs it if the cache is missing). Datasets are
separate invocations: set --output to a dataset-specific root and point --parquet /
--mapping at that dataset.

Usage:
    python scripts/run_all.py \
        --parquet data/compiled_dataset.parquet \
        --mapping data/labelset_mapping.csv \
        --output  outputs/bayes [--force-aggregate]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from bayes_pipeline.config.config import CONFIG
from bayes_pipeline.run_bayes import run_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/bayes"))
    ap.add_argument("--labelset", type=str, default="reefcompare",
                    help="Labelset to filter labelset_mapping.csv (default: reefcompare)")
    ap.add_argument("--glcm-window", type=int, default=None)
    ap.add_argument("--specdiv-plot", type=int, default=None)
    ap.add_argument("--force-aggregate", action="store_true")
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

    run_all(cfg, dataset=args.labelset, force_aggregate=args.force_aggregate)


if __name__ == "__main__":
    main()
