"""
evaluate_cv.py
PBS-ready CV evaluation script. Evaluates one trained CV fold model.

Usage:
    python3 evaluate_cv.py <data_path> <level> <weighted> --fold N [--labelset reefcompare]

PBS array job usage:
    #PBS -J 0-4
    python3 scripts/evaluate_cv.py data/spectra_A.parquet 3 true --fold $PBS_ARRAY_INDEX
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one CV fold model."
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("weighted", type=str)
    parser.add_argument("--fold", type=int, required=True)
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
    logger = get_logger(f"evaluate_{run_id}", LOG_DIR)

    logger.info("=" * 60)
    logger.info(f"EVALUATE CV  {run_id}")
    logger.info(f"  fold          : {fold} / {CV.n_splits - 1}")
    logger.info(f"  held_out_frac : {args.held_out_frac:.0%}")
    logger.info(f"  seed          : {args.seed}")
    logger.info("=" * 60)

    from config.config import LEVEL_CONFIGS, TURF_ALGAE_CLASS, LABEL_COLUMNS
    from data.loader import (load_spectra, remap_labels, sample_held_out_rois,
                             split_data_cv, get_feature_columns)
    from models.trainer import load_model
    from evaluation.evaluator import run_evaluation
    from utils.io import load_json
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
    df = sample_held_out_rois(df, held_out_frac=args.held_out_frac,
                              random_seed=args.seed, spectra=spectra)

    if level == 4:
        n_rois = df[LABEL_COLUMNS[4]].nunique()
        LEVEL_CONFIGS[4].n_classes = n_rois

    feature_cols = get_feature_columns(df)
    _, val_df = split_data_cv(df, level, fold)

    label_col = LABEL_COLUMNS[level]
    le = LabelEncoder()
    le.fit(df[label_col])  # Fit on full data for consistent class mapping
    y_val = le.transform(val_df[label_col])

    from data.loader import make_dmatrix
    dval = make_dmatrix(val_df, feature_cols, y_val, ref=False)

    booster = load_model(model_path)
    evals_result = load_json(out_dir / "evals_result.json")
    n_classes = len(le.classes_)

    run_evaluation(
        booster=booster,
        dtest=dval,
        y_test=y_val,
        evals_result=evals_result,
        le=le,
        n_classes=n_classes,
        eval_metric=lvl_cfg.eval_metric,
        turf_algae_class=TURF_ALGAE_CLASS,
        out_dir=out_dir,
    )

    logger.info(f"CV fold {fold} evaluation complete — outputs: {out_dir}")


if __name__ == "__main__":
    main()
