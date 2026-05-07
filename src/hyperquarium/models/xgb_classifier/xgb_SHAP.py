"""
shap.py
PBS-ready SHAP analysis script. Loads data and a trained model, computes
SHAP values, feature importance, scale-response curve, dependence plots,
and PCA → t-SNE embedding visualisation.

Requires train.py to have been run first for the same spectra/level/weighted
combination.

Usage:
    python3 shap.py <data_path> <level> <weighted>

Arguments:
    data_path : Path to the spectra parquet or CSV file
                e.g. data/spectra_A.parquet
    level     : Hierarchy level (1, 2, or 3)
    weighted  : "true" / "false"  (must match the training run)

Examples:
    python3 shap.py data/spectra_A.parquet 3 true
    python3 shap.py data/spectra_B.parquet 1 true

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
    parser.add_argument("data_path", type=Path,
                        help="Path to spectra parquet/CSV file")
    parser.add_argument("level", type=int, choices=[1, 2, 3, 4],
                        help="Hierarchy level (1, 2, or 3)")
    parser.add_argument("weighted", type=str,
                        help="Apply sample weights: true / false")
    return parser.parse_args()


def parse_weighted(val: str) -> bool:
    if val.strip().lower() in ("true", "1", "yes"):
        return True
    if val.strip().lower() in ("false", "0", "no"):
        return False
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
    from config.config import LEVEL_CONFIGS, SPLIT, TURF_ALGAE_CLASS, SHAP_CFG
    from data.loader import (load_spectra, remap_labels, split_data,
                             get_feature_columns, encode_labels, make_dmatrix)
    from models.trainer import load_model
    from evaluation.evaluator import predict_leaf
    from features.shap_analysis import run_shap_analysis, plot_pca_tsne
    from utils.io import make_output_dir

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    lvl_cfg = LEVEL_CONFIGS[level]
    out_dir = make_output_dir(OUTPUT_DIR, spectra, level, weighted)

    model_path = out_dir / "model.json"
    if not model_path.exists():
        logger.error(
            f"Model not found: {model_path}\n"
            f"Run train.py first for spectra={spectra} level={level} weighted={weighted}"
        )
        sys.exit(1)

    # ---- Load & prepare ---------------------------------------------------
    df = load_spectra(data_path)
    df = remap_labels(df)

    # Level 4: derive n_classes dynamically from unique ROIs in data
    if level == 4:
        from config.config import LABEL_COLUMNS
        n_rois = df[LABEL_COLUMNS[4]].nunique()
        LEVEL_CONFIGS[4].n_classes = n_rois
        logger.info(f"Level 4 n_classes set dynamically: {n_rois}")
        from data.loader import save_roi_mapping
        save_roi_mapping(df, out_dir)

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

    # ---- SHAP analysis ----------------------------------------------------
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

    # ---- PCA → t-SNE on leaf embeddings -----------------------------------
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