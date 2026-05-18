"""
shap_cv.py
PBS-ready CV SHAP analysis script. Runs SHAP for one trained CV fold model.

Usage:
    python3 shap_cv.py <data_path> <level> <weighted> --fold N [--labelset pilot]

PBS array job usage:
    #PBS -J 0-4
    python3 scripts/shap_cv.py data/spectra_A.parquet 3 true --fold $PBS_ARRAY_INDEX
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SHAP analysis for one CV fold model."
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("weighted", type=str)
    parser.add_argument("--fold", type=int, required=True)
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
    fold = args.fold
    data_path = Path(args.data_path)
    spectra = data_path.stem.split("_")[-1].upper()

    import numpy as np
    from utils.logger import get_logger
    from config.config import OUTPUT_DIR, LOG_DIR, CV, SHAP_CFG
    run_id = f"cv_spectra_{spectra}_level{level}_fold{fold}_{'weighted' if weighted else 'unweighted'}"
    logger = get_logger(f"shap_{run_id}", LOG_DIR)

    logger.info("=" * 60)
    logger.info(f"SHAP CV  {run_id}")
    logger.info(f"  fold : {fold} / {CV.n_splits - 1}")
    logger.info("=" * 60)

    from config.config import LEVEL_CONFIGS, TURF_ALGAE_CLASS, LABEL_COLUMNS
    from data.loader import (load_spectra, remap_labels, subsample_turf_rois,
                             split_data_cv, get_feature_columns, make_dmatrix)
    from models.trainer import load_model
    from features.shap_analysis import run_shap_analysis
    from evaluation.evaluator import predict_leaf
    from features.shap_analysis import plot_pca_tsne
    from sklearn.preprocessing import LabelEncoder

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    suffix = "" if weighted else "_unweighted"
    out_dir = Path(OUTPUT_DIR) / f"spectra_{spectra}" / f"level_{level}_cv{suffix}" / f"fold_{fold}"
    model_path = out_dir / "model.json"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path} — run train_cv.py first.")
        sys.exit(1)

    lvl_cfg = LEVEL_CONFIGS[level]

    # ---- Load & prepare ---------------------------------------------------
    df = load_spectra(data_path)
    df = remap_labels(df, dataset=args.labelset)
    df = subsample_turf_rois(df, spectra=spectra, random_seed=42)

    if level == 4:
        n_rois = df[LABEL_COLUMNS[4]].nunique()
        LEVEL_CONFIGS[4].n_classes = n_rois

    feature_cols = get_feature_columns(df)
    _, val_df = split_data_cv(df, level, fold)

    label_col = LABEL_COLUMNS[level]
    le = LabelEncoder()
    le.fit(df[label_col])
    y_val = le.transform(val_df[label_col])

    booster = load_model(model_path)

    # ---- Subsample for SHAP -----------------------------------------------
    rng = np.random.default_rng(SHAP_CFG.random_seed)
    if SHAP_CFG.shap_sample_size and len(val_df) > SHAP_CFG.shap_sample_size:
        idx = rng.choice(len(val_df), SHAP_CFG.shap_sample_size, replace=False)
        shap_df = val_df.iloc[idx]
        shap_y = y_val[idx]
        logger.info(f"SHAP subsample: {SHAP_CFG.shap_sample_size:,} rows")
    else:
        shap_df = val_df
        shap_y = y_val

    dshap = make_dmatrix(shap_df, feature_cols, shap_y, ref=False)
    X_shap = shap_df[feature_cols].values

    # ---- SHAP analysis ----------------------------------------------------
    run_shap_analysis(
        booster=booster,
        dmatrix=dshap,
        X=X_shap,
        feature_names=feature_cols,
        le=le,
        n_classes=len(le.classes_),
        cfg=SHAP_CFG,
        out_dir=out_dir,
    )

    # ---- PCA → t-SNE ------------------------------------------------------
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
        title_suffix=f"spectra {spectra} — level {level} — CV fold {fold}",
    )

    logger.info(f"CV fold {fold} SHAP complete — outputs: {out_dir}")


if __name__ == "__main__":
    main()
