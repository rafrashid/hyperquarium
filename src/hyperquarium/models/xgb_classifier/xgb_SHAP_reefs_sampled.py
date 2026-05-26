"""
shap.py
PBS-ready SHAP analysis script.

Usage:
    python3 shap.py <data_path> <level> <weighted> [--labelset pilot]

PBS usage:
    module load python3/3.14.4
    python3 shap.py "$DATA_PATH" "$LEVEL" "$WEIGHTED"
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHAP analysis for one trained XGBoost model."
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("weighted", type=str)
    parser.add_argument("--labelset", type=str, default="pilot")
    return parser.parse_args()


def parse_weighted(val: str) -> bool:
    if val.strip().lower() in ("true", "1", "yes"):  return True
    if val.strip().lower() in ("false", "0", "no"):  return False
    raise ValueError(f"weighted must be true/false, got: '{val}'")


def main() -> None:
    args = parse_args()
    weighted = parse_weighted(args.weighted)
    level = args.level
    data_path = Path(args.data_path)
    spectra = data_path.stem.split("_")[-1].upper()

    from utils.logger import get_logger
    from config.config import OUTPUT_DIR, LOG_DIR
    run_id = f"spectra_{spectra}_level{level}_{'weighted' if weighted else 'unweighted'}"
    logger = get_logger(f"shap_{run_id}", LOG_DIR)

    logger.info("=" * 60)
    logger.info(f"SHAP  {run_id}")
    logger.info(f"  data_path : {data_path}")
    logger.info(f"  level     : {level}")
    logger.info(f"  weighted  : {weighted}")
    logger.info("=" * 60)

    import numpy as np
    from config.config import LEVEL_CONFIGS, SPLIT, TURF_ALGAE_CLASS, SHAP_CFG, LABEL_COLUMNS
    from data.loader import (load_spectra, remap_labels, sample_held_out_rois, split_data,
                             get_feature_columns, encode_labels, make_dmatrix,
                             patch_level_configs)
    from models.trainer import load_model
    from evaluation.evaluator import predict_leaf
    from features.shap_analysis import run_shap_analysis, plot_pca_tsne
    from utils.io import make_output_dir

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    out_dir = make_output_dir(OUTPUT_DIR, spectra, level, weighted)
    model_path = out_dir / "model.json"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path} — run train.py first.")
        sys.exit(1)

    # ---- Load & prepare ---------------------------------------------------
    df = load_spectra(data_path)
    df = remap_labels(df, dataset=args.labelset)
    df = sample_held_out_rois(df, spectra=spectra, random_seed=42)

    # Patch n_classes for Levels 1, 2, 3 dynamically from data
    patch_level_configs(df)

    # Level 4: handled separately
    if level == 4:
        roi_map_path = out_dir / "roi_label_mapping.csv"
        if roi_map_path.exists():
            import pandas as _pd
            n_rois = _pd.read_csv(roi_map_path)["label_level4"].nunique()
            LEVEL_CONFIGS[4].n_classes = n_rois
            logger.info(f"Level 4 n_classes loaded from roi_label_mapping.csv: {n_rois}")
        else:
            logger.warning("roi_label_mapping.csv not found — falling back to df count.")
            LEVEL_CONFIGS[4].n_classes = df[LABEL_COLUMNS[4]].nunique()

    lvl_cfg = LEVEL_CONFIGS[level]

    train_df, val_df, test_df = split_data(df, level, SPLIT)
    feature_cols = get_feature_columns(df)
    y_train, y_val, y_test, le = encode_labels(train_df, val_df, test_df, level)

    booster = load_model(model_path)

    # ---- Subsample for SHAP -----------------------------------------------
    rng = np.random.default_rng(SHAP_CFG.random_seed)
    if SHAP_CFG.shap_sample_size and len(test_df) > SHAP_CFG.shap_sample_size:
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

    dleaf = make_dmatrix(shap_df, feature_cols, shap_y, ref=False)
    leaves = predict_leaf(booster, dleaf)
    plot_pca_tsne(
        embedding_matrix=leaves,
        y=shap_y,
        le=le,
        turf_algae_class=TURF_ALGAE_CLASS,
        sample_size=SHAP_CFG.shap_sample_size or 10_000,
        random_seed=SHAP_CFG.random_seed,
        out_dir=out_dir,
        title_suffix=f"spectra {spectra} — level {level}",
    )

    logger.info(f"SHAP analysis complete — outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
