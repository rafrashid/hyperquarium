#!/usr/bin/env python3
"""
scripts/fit_one.py — prototype ONE cell end to end (prove before looping).

Runs the prior-predictive gate, fits, prints diagnostics, computes the Step-4
contrast tables, and (optionally) saves. Use this first on your machine to confirm
the Bambi/PyMC workflow before running the full 21-cell loop.

Usage:
    python scripts/fit_one.py --feature homogeneity --level 2 \
        --output outputs/bayes [--no-save] [--prior-check]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config.config import CONFIG
from data.aggregate import load_or_build_summary
from models.fit import prepare_cell, prior_predictive
from run_bayes import run_one_cell
from utils.logger import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/bayes"))
    ap.add_argument("--labelset", type=str, default="reefcompare",
                    help="Labelset to filter labelset_mapping.csv (default: reefcompare)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--prior-check", action="store_true",
                    help="Run prior-predictive gate and report the implied range.")
    args = ap.parse_args()

    cfg = CONFIG
    cfg.paths.output_root = args.output
    cfg.labels.labelset = args.labelset
    logger = get_logger()

    summary = load_or_build_summary(cfg, dataset=args.labelset, logger=logger)

    if args.prior_check:
        cell = prepare_cell(summary, args.feature, args.level, cfg)
        _, idata_prior = prior_predictive(cell, cfg)
        pp = idata_prior.prior_predictive["feat_z"].values
        logger.info("Prior-predictive feat_z range: [%.2f, %.2f], |max|=%.2f "
                    "(standardised; expect bulk within ~±6).",
                    float(pp.min()), float(pp.max()), float(abs(pp).max()))

    res = run_one_cell(summary, args.feature, args.level, cfg,
                       logger=logger, save=not args.no_save)

    print("\n=== per_class_sigma ===")
    print(res["tables"]["per_class_sigma"].to_string(index=False))
    print("\n=== turf_distances ===")
    print(res["tables"]["turf_distances"].to_string(index=False))
    print("\n=== turf_sigma_ratios ===")
    print(res["tables"]["turf_sigma_ratios"].to_string(index=False))
    print("\n=== sigma_vs_gap ===")
    print(res["tables"]["sigma_vs_gap"].to_string(index=False))
    print("\n=== diagnostics ===")
    print(res["diagnostics"])


if __name__ == "__main__":
    main()
