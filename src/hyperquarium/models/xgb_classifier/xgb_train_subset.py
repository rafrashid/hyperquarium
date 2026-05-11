"""
train.py
PBS-ready training script. Loads data, remaps labels, splits, and trains
one XGBoost model. Model and metadata are saved to outputs/.

Usage:
    python3 train.py <data_path> <level> <weighted>

Arguments:
    data_path : Path to the spectra parquet or CSV file
                e.g. data/spectra_A.parquet
    level     : Hierarchy level to train (1, 2, or 3)
    weighted  : Whether to apply inverse-frequency sample weights
                "true" / "false"  (case-insensitive)

Examples:
    python3 train.py data/spectra_A.parquet 3 true
    python3 train.py data/spectra_A.parquet 3 false
    python3 train.py data/spectra_B.parquet 1 true

PBS usage:
    module load python3/3.14.4
    python3 train.py "$DATA_PATH" "$LEVEL" "$WEIGHTED"
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one XGBoost model for the algal turf pipeline."
    )
    parser.add_argument("data_path", type=Path,
                        help="Path to spectra parquet/CSV file")
    parser.add_argument("level", type=int, choices=[1, 2, 3, 4],
                        help="Hierarchy level (1, 2, or 3)")
    parser.add_argument("weighted", type=str,
                        help="Apply sample weights: true / false")
    parser.add_argument("--labelset", type=str, default="pilot",
                        help="Label mapping labelset filter (default: pilot)")
    return parser.parse_args()


def parse_weighted(val: str) -> bool:
    if val.strip().lower() in ("true", "1", "yes"):
        return True
    if val.strip().lower() in ("false", "0", "no"):
        return False
    raise ValueError(f"weighted must be true/false, got: '{val}'")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    weighted = parse_weighted(args.weighted)
    level = args.level
    data_path = Path(args.data_path)

    # Derive spectra label from filename stem: spectra_A.parquet -> "A"
    spectra = data_path.stem.split("_")[-1].upper()

    # ---- Logging ----------------------------------------------------------
    from utils.logger import get_logger
    from config.config import OUTPUT_DIR, LOG_DIR
    run_id = f"spectra_{spectra}_level{level}_{'weighted' if weighted else 'unweighted'}"
    logger = get_logger(f"train_{run_id}", LOG_DIR)

    logger.info("=" * 60)
    logger.info(f"TRAIN  {run_id}")
    logger.info(f"  data_path : {data_path}")
    logger.info(f"  level     : {level}")
    logger.info(f"  weighted  : {weighted}")
    logger.info("=" * 60)

    # ---- Imports ----------------------------------------------------------
    from config.config import LEVEL_CONFIGS, XGB, SPLIT, OUTPUT_DIR
    from data.loader import (load_spectra, remap_labels, split_data,
                             get_feature_columns, encode_labels,
                             compute_sample_weights, make_dmatrix,
                             save_split_metadata, subsample_turf_rois)
    from models.trainer import build_params, patch_num_class, train_model, save_model, save_training_metadata
    from utils.io import make_output_dir

    # ---- Validate ---------------------------------------------------------
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    if level not in LEVEL_CONFIGS:
        logger.error(f"Invalid level: {level}. Must be 1, 2, or 3.")
        sys.exit(1)

    lvl_cfg = LEVEL_CONFIGS[level]
    out_dir = make_output_dir(OUTPUT_DIR, spectra, level, weighted)

    # ---- Load & prepare ---------------------------------------------------
    df = load_spectra(data_path)
    df = remap_labels(df, dataset=args.labelset)
    df = subsample_turf_rois(df, random_seed=42)  # optional — remove for full dataset

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
    save_split_metadata(train_df, val_df, test_df, le, level, out_dir)

    # ---- DMatrix ----------------------------------------------------------
    sample_weight = compute_sample_weights(y_train) if weighted else None
    dtrain = make_dmatrix(train_df, feature_cols, y_train, sample_weight, ref=None)
    dval = make_dmatrix(val_df, feature_cols, y_val, ref=dtrain)

    # ---- Train ------------------------------------------------------------
    params = build_params(XGB, lvl_cfg)
    params = patch_num_class(params, le)  # Ensure num_class matches actual encoded classes
    booster, evals_result = train_model(dtrain, dval, params, XGB, run_id)
    save_model(booster, out_dir)
    save_training_metadata(booster, evals_result, params, le, out_dir, weighted)

    logger.info(f"Training complete — outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
