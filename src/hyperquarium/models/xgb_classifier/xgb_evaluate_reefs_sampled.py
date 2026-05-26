"""
evaluate.py
PBS-ready evaluation script. Loads data and a trained model, then runs
the full evaluation suite: metrics, confusion matrix, PR curves,
learning curve, and boundary samples.

Usage:
    python3 evaluate.py <data_path> <level> <weighted> [--labelset pilot]

PBS usage:
    module load python3/3.14.4
    python3 evaluate.py "$DATA_PATH" "$LEVEL" "$WEIGHTED"
"""

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one trained XGBoost model for the algal turf pipeline."
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
    logger = get_logger(f"evaluate_{run_id}", LOG_DIR)

    logger.info("=" * 60)
    logger.info(f"EVALUATE  {run_id}")
    logger.info(f"  data_path : {data_path}")
    logger.info(f"  level     : {level}")
    logger.info(f"  weighted  : {weighted}")
    logger.info("=" * 60)

    from config.config import LEVEL_CONFIGS, SPLIT, TURF_ALGAE_CLASS, LABEL_COLUMNS
    from data.loader import (load_spectra, remap_labels, sample_held_out_rois, split_data,
                             get_feature_columns, encode_labels, make_dmatrix,
                             patch_level_configs)
    from models.trainer import load_model
    from evaluation.evaluator import run_evaluation
    from utils.io import make_output_dir, load_json

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

    dtrain = make_dmatrix(train_df, feature_cols, y_train, ref=None)
    dtest = make_dmatrix(test_df, feature_cols, y_test, ref=dtrain)

    booster = load_model(model_path)
    evals_result = load_json(out_dir / "evals_result.json")

    run_evaluation(
        booster=booster,
        dtest=dtest,
        y_test=y_test,
        evals_result=evals_result,
        le=le,
        n_classes=len(le.classes_),
        eval_metric=lvl_cfg.eval_metric,
        turf_algae_class=TURF_ALGAE_CLASS,
        out_dir=out_dir,
    )

    logger.info(f"Evaluation complete — outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
