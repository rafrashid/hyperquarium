"""
run_bayes.py — orchestrator.

Provides:
  run_one_cell(...)  — fit + diagnose + contrasts + save for a single (feature, level).
  run_all(...)       — loop over the 21 cells (7 features × 3 levels) for one dataset.

Datasets are separate invocations: point cfg.paths at one dataset's parquet and run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from bayes_pipeline.analysis.contrasts import all_contrasts
from bayes_pipeline.analysis.diagnostics import diagnostics_summary, diagnostics_table
from bayes_pipeline.config.config import Config, CONFIG
from bayes_pipeline.data.aggregate import load_or_build_summary
from bayes_pipeline.models.fit import prepare_cell, fit_cell
from bayes_pipeline.utils.io import save_table, save_json, save_idata, ensure_dir
from bayes_pipeline.utils.logger import get_logger


def _cell_dir(cfg: Config, feature: str, level: int) -> Path:
    return cfg.paths.output_root / feature / f"level_{level}"


def run_one_cell(summary: pd.DataFrame, feature: str, level: int,
                 cfg: Config, logger=None, save: bool = True) -> dict:
    """Fit one cell end to end. Returns a dict of result tables + diagnostics."""
    logger = logger or get_logger()
    cell = prepare_cell(summary, feature, level, cfg)
    model, idata = fit_cell(cell, cfg, logger=logger)

    diag = diagnostics_summary(idata, cfg)
    if diag["n_flagged"]:
        logger.warning("[%s L%d] %d flagged params (max r_hat=%.3f, min ess=%.0f, div=%d)",
                       feature, level, diag["n_flagged"], diag["max_rhat"] or -1,
                       diag["min_ess_bulk"] or -1, diag["divergences"])

    tables = all_contrasts(idata, cell, cfg)

    if save:
        out = ensure_dir(_cell_dir(cfg, feature, level))
        save_idata(idata, out / "idata.nc")
        for name, df in tables.items():
            if not df.empty:
                save_table(df, out / f"{name}.csv")
        save_table(diagnostics_table(idata, cfg), out / "diagnostics.csv")
        save_json({
            "feature": feature, "level": level,
            "n_roi": len(cell.df),
            "n_classes": int(cell.df["class"].nunique()),
            "class_n_roi": cell.class_n_roi,
            "grand_mean": cell.grand_mean, "grand_sd": cell.grand_sd,
            "diagnostics": diag,
            "priors": {"mu_sd": cfg.priors.mu_sd, "log_sigma_sd": cfg.priors.log_sigma_sd},
        }, out / "metadata.json")
        logger.info("[%s L%d] saved -> %s", feature, level, out)

    return {"cell": cell, "idata": idata, "tables": tables, "diagnostics": diag}


def run_all(cfg: Config = CONFIG, dataset: str | None = None,
            force_aggregate: bool = False, logger=None) -> None:
    """Run all 21 cells for the currently configured dataset/labelset."""
    logger = logger or get_logger(logfile=cfg.paths.output_root / "run.log")
    dataset = dataset if dataset is not None else cfg.labels.labelset
    logger.info("=== Bayesian ROI feature analysis: full run ===")
    logger.info("Output root: %s | labelset: %s", cfg.paths.output_root, dataset)

    summary = load_or_build_summary(cfg, dataset=dataset,
                                    force=force_aggregate, logger=logger)

    failures = []
    for feature in cfg.features.feature_names:
        for level in cfg.labels.levels:
            try:
                run_one_cell(summary, feature, level, cfg, logger=logger)
            except Exception as exc:  # keep going; log the cell
                logger.exception("[%s L%d] FAILED: %s", feature, level, exc)
                failures.append((feature, level, str(exc)))

    if failures:
        logger.warning("Completed with %d failed cells: %s", len(failures), failures)
    else:
        logger.info("All 21 cells completed successfully.")


if __name__ == "__main__":
    run_all()
