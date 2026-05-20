"""
train_cv.py
PBS-ready CV training script. Trains one fold of a StratifiedGroupKFold
cross-validation, grouped by roi_ID so all pixels from one ROI go entirely
to train or validation.

Output goes to: outputs/spectra_{X}/level_{N}_cv/fold_{K}/

Usage:
    python3 train_cv.py <data_path> <level> <weighted> --fold N [--labelset reefcompare]

Examples:
    python3 train_cv.py data/spectra_A.parquet 3 true --fold 0
    python3 train_cv.py data/spectra_A.parquet 1 true --fold 2 --labelset reefcompare

PBS array job usage:
    #PBS -J 0-4
    python3 scripts/train_cv.py data/spectra_A.parquet 3 true --fold $PBS_ARRAY_INDEX
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one CV fold for the algal turf pipeline."
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("weighted", type=str)
    parser.add_argument("--fold", type=int, required=True,
                        help="Fold index (0 to n_splits - 1)")
    parser.add_argument("--labelset", type=str, default="reefcompare")
    parser.add_argument("--held-out-frac", type=float, default=0.20,
                        help="Fraction of ROIs per Level 2 class to hold out (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for held-out sampling (default: 42)")
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

    from utils.logger import get_logger
    from config.config import OUTPUT_DIR, LOG_DIR, CV
    run_id = f"cv_spectra_{spectra}_level{level}_fold{fold}_{'weighted' if weighted else 'unweighted'}"
    logger = get_logger(f"train_{run_id}", LOG_DIR)

    logger.info("=" * 60)
    logger.info(f"TRAIN CV  {run_id}")
    logger.info(f"  data_path     : {data_path}")
    logger.info(f"  level         : {level}")
    logger.info(f"  fold          : {fold} / {CV.n_splits - 1}")
    logger.info(f"  weighted      : {weighted}")
    logger.info(f"  labelset      : {args.labelset}")
    logger.info(f"  held_out_frac : {args.held_out_frac:.0%}")
    logger.info(f"  seed          : {args.seed}")
    logger.info("=" * 60)

    from config.config import LEVEL_CONFIGS, XGB, OUTPUT_DIR, LABEL_COLUMNS
    from data.loader import (load_spectra, remap_labels, sample_held_out_rois,
                             split_data_cv, get_feature_columns, encode_labels,
                             compute_sample_weights, make_dmatrix,
                             save_split_metadata)
    from models.trainer import build_params, patch_num_class, train_model, save_model, save_training_metadata
    from utils.io import save_json

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    lvl_cfg = LEVEL_CONFIGS[level]

    # CV output directory: outputs/spectra_A/level_3_cv/fold_0/
    suffix = "" if weighted else "_unweighted"
    out_dir = Path(OUTPUT_DIR) / f"spectra_{spectra}" / f"level_{level}_cv{suffix}" / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load & prepare ---------------------------------------------------
    df = load_spectra(data_path)
    df = remap_labels(df, dataset=args.labelset)
    df = sample_held_out_rois(df, held_out_frac=args.held_out_frac,
                              random_seed=args.seed, spectra=spectra)

    if level == 4:
        n_rois = df[LABEL_COLUMNS[4]].nunique()
        LEVEL_CONFIGS[4].n_classes = n_rois
        logger.info(f"Level 4 n_classes set dynamically: {n_rois}")

    feature_cols = get_feature_columns(df)

    # ---- CV split ---------------------------------------------------------
    train_df, val_df = split_data_cv(df, level, fold)

    # Fit encoder on train only — apply to val
    from sklearn.preprocessing import LabelEncoder
    from config.config import LABEL_COLUMNS
    label_col = LABEL_COLUMNS[level]
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col])
    y_val = le.transform(val_df[label_col])

    # Save fold metadata
    save_json({
        "fold": fold,
        "n_splits": CV.n_splits,
        "level": level,
        "label_column": label_col,
        "class_mapping": {str(i): cls for i, cls in enumerate(le.classes_)},
        "split_sizes": {"train": len(train_df), "val": len(val_df)},
        "train_roi_count": int(train_df[CV.group_col].nunique()),
        "val_roi_count": int(val_df[CV.group_col].nunique()),
    }, out_dir / "fold_metadata.json")

    # ---- DMatrix ----------------------------------------------------------
    sample_weight = compute_sample_weights(y_train) if weighted else None
    dtrain = make_dmatrix(train_df, feature_cols, y_train, sample_weight, ref=None)
    dval = make_dmatrix(val_df, feature_cols, y_val, ref=dtrain)

    # ---- Train ------------------------------------------------------------
    params = build_params(XGB, lvl_cfg)
    params = patch_num_class(params, le)
    booster, evals_result = train_model(dtrain, dval, params, XGB, run_id)
    save_model(booster, out_dir)
    save_training_metadata(booster, evals_result, params, le, out_dir, weighted)

    logger.info(f"CV fold {fold} training complete — outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
