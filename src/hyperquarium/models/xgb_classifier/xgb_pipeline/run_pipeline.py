"""
run_pipeline.py
Main entry point. Orchestrates data loading, training, evaluation, and SHAP analysis
across all spectra types and hierarchy levels.

Usage:
    python run_pipeline.py                         # Full pipeline
    python run_pipeline.py --spectra A             # Single spectra type
    python run_pipeline.py --spectra A --level 3   # Single spectra + level
    python run_pipeline.py --spectra A --level 3 --stage train
    python run_pipeline.py --spectra A --level 3 --stage evaluate
    python run_pipeline.py --spectra A --level 3 --stage shap
    python run_pipeline.py --cross-spectra         # Cross-spectra comparison only
"""

import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb

from config.config import (
    SPECTRA_FILES, SPECTRA_TYPES, LEVELS, LEVEL_CONFIGS,
    OUTPUT_DIR, LOG_DIR, XGB, SPLIT, SHAP_CFG, TURF_ALGAE_CLASS,
)
from data.loader import (
    remap_labels, load_spectra, split_data, get_feature_columns, encode_labels,
    compute_sample_weights, make_dmatrix, save_split_metadata,
)
from evaluation.evaluator import run_evaluation
from features.shap_analysis import run_shap_analysis, compare_shap_across_spectra
from models.trainer import (
    build_params, train_model, save_model, save_training_metadata, load_model,
)
from utils.io import make_output_dir
from utils.logger import get_logger

logger = get_logger("pipeline", LOG_DIR)


# ---------------------------------------------------------------------------
# Single model run
# ---------------------------------------------------------------------------

def run_one(
        spectra: str,
        level: int,
        stage: str = "all",
        weighted: bool = True,
) -> None:
    """
    Runs the pipeline for a single spectra type / level / weighting combination.

    Args:
        spectra:  Spectra type label ('A', 'B', 'C', 'D').
        level:    Hierarchy level (1, 2, or 3).
        stage:    Which stage to run: 'train', 'evaluate', 'shap', or 'all'.
        weighted: If False, skips sample weighting (Level 3 unweighted baseline).
    """
    out_dir = make_output_dir(OUTPUT_DIR, spectra, level, weighted)
    run_id = f"spectra_{spectra}_level{level}_{'weighted' if weighted else 'unweighted'}"
    lvl_cfg = LEVEL_CONFIGS[level]

    logger.info(f"{'=' * 60}")
    logger.info(f"Starting run: {run_id}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Stage: {stage}")

    # ---- Load, remap & split -----------------------------------------
    df = load_spectra(SPECTRA_FILES[spectra])
    df = remap_labels(df)  # Maps raw Level_0 labels -> label_level1/2/3
    train_df, val_df, test_df = split_data(df, level, SPLIT)
    feature_cols = get_feature_columns(df)
    y_train, y_val, y_test, le = encode_labels(train_df, val_df, test_df, level)
    save_split_metadata(train_df, val_df, test_df, le, level, out_dir)

    # ---- DMatrix ------------------------------------------------------
    sample_weight = compute_sample_weights(y_train) if weighted else None
    dtrain = make_dmatrix(train_df, feature_cols, y_train, sample_weight, ref=None)
    dval = make_dmatrix(val_df, feature_cols, y_val, ref=dtrain)
    dtest = make_dmatrix(test_df, feature_cols, y_test, ref=dtrain)

    # ---- Train --------------------------------------------------------
    if stage in ("train", "all"):
        params = build_params(XGB, lvl_cfg)
        booster, evals_result = train_model(dtrain, dval, params, XGB, run_id)
        save_model(booster, out_dir)
        save_training_metadata(booster, evals_result, params, le, out_dir, weighted)
    else:
        model_path = out_dir / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run --stage train first."
            )
        booster = load_model(model_path)
        from utils.io import load_json
        evals_result = load_json(out_dir / "evals_result.json")

    # ---- Evaluate -----------------------------------------------------
    if stage in ("evaluate", "all"):
        run_evaluation(
            booster=booster,
            dtest=dtest,
            y_test=y_test,
            evals_result=evals_result,
            le=le,
            n_classes=lvl_cfg.n_classes,
            eval_metric=lvl_cfg.eval_metric,
            turf_algae_class=TURF_ALGAE_CLASS,
            out_dir=out_dir,
        )

    # ---- SHAP ---------------------------------------------------------
    if stage in ("shap", "all"):
        # Subsample test set for SHAP if configured
        if SHAP_CFG.shap_sample_size and len(test_df) > SHAP_CFG.shap_sample_size:
            rng = np.random.default_rng(SHAP_CFG.random_seed)
            idx = rng.choice(len(test_df), SHAP_CFG.shap_sample_size, replace=False)
            shap_df = test_df.iloc[idx]
            shap_y = y_test[idx]
            logger.info(f"SHAP subsample: {SHAP_CFG.shap_sample_size:,} rows")
        else:
            shap_df = test_df
            shap_y = y_test

        dshap = make_dmatrix(shap_df, feature_cols, shap_y, ref=False)
        X_shap = shap_df[feature_cols].values

        run_shap_analysis(
            booster=booster,
            dmatrix=dshap,
            X=X_shap,
            feature_names=feature_cols,
            le=le,
            n_classes=lvl_cfg.n_classes,
            cfg=SHAP_CFG,
            out_dir=out_dir,
        )

    logger.info(f"Completed run: {run_id}")


# ---------------------------------------------------------------------------
# Cross-spectra comparison
# ---------------------------------------------------------------------------

def run_cross_spectra_comparison(level: int = 3) -> None:
    """
    Loads feature importance CSVs across all spectra types for a given level
    and produces a cross-spectra SHAP comparison.

    Args:
        level: Hierarchy level to compare across spectra (default: 3).
    """
    import pandas as pd
    cross_dir = Path(OUTPUT_DIR) / "cross_spectra" / f"level_{level}"
    cross_dir.mkdir(parents=True, exist_ok=True)

    importance_dfs = {}
    for spectra in SPECTRA_TYPES:
        imp_path = make_output_dir(OUTPUT_DIR, spectra, level) / "feature_importance_shap.csv"
        if imp_path.exists():
            importance_dfs[spectra] = pd.read_csv(imp_path, index_col=0)
            logger.info(f"Loaded importance: {imp_path}")
        else:
            logger.warning(f"Missing importance file for spectra {spectra} level {level}: {imp_path}")

    if len(importance_dfs) < 2:
        logger.error("Need at least 2 spectra types to compare. Run SHAP analysis first.")
        return

    compare_shap_across_spectra(importance_dfs, cross_dir)
    logger.info(f"Cross-spectra comparison saved to {cross_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost algal turf pipeline")
    parser.add_argument("--spectra", type=str, choices=SPECTRA_TYPES + ["all"], default="all",
                        help="Spectra type to run (default: all)")
    parser.add_argument("--level", type=int, choices=LEVELS + [0], default=0,
                        help="Hierarchy level to run; 0 = all (default: all)")
    parser.add_argument("--stage", type=str, choices=["train", "evaluate", "shap", "all"],
                        default="all", help="Pipeline stage to run (default: all)")
    parser.add_argument("--cross-spectra", action="store_true",
                        help="Run cross-spectra SHAP comparison only")
    parser.add_argument("--cross-level", type=int, default=3,
                        help="Level to use for cross-spectra comparison (default: 3)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cross_spectra:
        run_cross_spectra_comparison(level=args.cross_level)
        return

    spectra_to_run = SPECTRA_TYPES if args.spectra == "all" else [args.spectra]
    levels_to_run = LEVELS if args.level == 0 else [args.level]

    for spectra in spectra_to_run:
        for level in levels_to_run:
            lvl_cfg = LEVEL_CONFIGS[level]

            # Weighted run
            run_one(spectra, level, stage=args.stage, weighted=True)

            # Level 3 unweighted baseline (hypothesis control)
            if level == 3 and lvl_cfg.run_unweighted and args.stage in ("train", "all"):
                run_one(spectra, level, stage=args.stage, weighted=False)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()