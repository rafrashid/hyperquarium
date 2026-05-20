"""
xgb_cv_summary.py
Aggregates cross-validation results across all folds for one model.
Computes mean/std of validation metrics and averaged SHAP feature importance.

Usage:
    python3 xgb_cv_summary.py <spectra> <level> [--weighted true]

Examples:
    python3 xgb_cv_summary.py A 3
    python3 xgb_cv_summary.py A 1 --weighted false
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate CV fold results into summary statistics."
    )
    parser.add_argument("spectra", type=str, help="Spectra type label e.g. A")
    parser.add_argument("level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--weighted", type=str, default="true")
    return parser.parse_args()


def parse_weighted(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def main() -> None:
    args = parse_args()
    spectra = args.spectra.upper()
    level = args.level
    weighted = parse_weighted(args.weighted)

    from utils.logger import get_logger
    from config.config import OUTPUT_DIR, LOG_DIR, CV
    logger = get_logger(f"cv_summary_{spectra}_L{level}", LOG_DIR)

    logger.info("=" * 60)
    logger.info(f"CV SUMMARY  spectra={spectra}  level={level}  weighted={weighted}")
    logger.info("=" * 60)

    suffix = "" if weighted else "_unweighted"
    cv_dir = Path(OUTPUT_DIR) / f"spectra_{spectra}" / f"level_{level}_cv{suffix}"
    out_dir = cv_dir / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_splits = CV.n_splits

    # ── 1. Validation metrics per fold ────────────────────────────────────
    metrics_records = []
    best_scores = []
    best_iters = []

    for fold in range(n_splits):
        fold_dir = cv_dir / f"fold_{fold}"
        metrics_path = fold_dir / "metrics.json"
        meta_path = fold_dir / "training_metadata.json"

        if not metrics_path.exists():
            logger.warning(f"fold {fold}: metrics.json not found — skipping.")
            continue

        with open(metrics_path) as f:
            m = json.load(f)
        with open(meta_path) as f:
            meta = json.load(f)

        row = {"fold": fold, "macro_f1": m.get("macro_f1")}
        for cls, f1 in m.get("per_class_f1", {}).items():
            row[f"f1_{cls}"] = f1
        for cls, ap in m.get("average_precision_per_class", {}).items():
            row[f"ap_{cls}"] = ap
        row["best_iteration"] = meta.get("best_iteration")
        row["best_score"] = meta.get("best_score")
        metrics_records.append(row)
        best_scores.append(meta.get("best_score", np.nan))
        best_iters.append(meta.get("best_iteration", np.nan))

    if not metrics_records:
        logger.error("No fold metrics found — run evaluate_cv.py first.")
        sys.exit(1)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(out_dir / "metrics_per_fold.csv", index=False)

    # Summary statistics
    numeric_cols = [c for c in metrics_df.columns if c != "fold"]
    summary_stats = metrics_df[numeric_cols].agg(["mean", "std", "min", "max"])
    summary_stats.to_csv(out_dir / "metrics_summary.csv")

    logger.info(
        f"Validation metrics across {len(metrics_records)} folds:\n"
        f"  macro_f1  mean={metrics_df['macro_f1'].mean():.4f}  "
        f"std={metrics_df['macro_f1'].std():.4f}  "
        f"min={metrics_df['macro_f1'].min():.4f}  "
        f"max={metrics_df['macro_f1'].max():.4f}"
    )
    logger.info(
        f"  best_score mean={np.nanmean(best_scores):.5f}  "
        f"best_iter  mean={np.nanmean(best_iters):.1f}"
    )

    # ── 2. Averaged SHAP feature importance ───────────────────────────────
    shap_col = "mean_abs_shap_global"
    imp_frames = []

    for fold in range(n_splits):
        imp_path = cv_dir / f"fold_{fold}" / "feature_importance_shap.csv"
        if imp_path.exists():
            imp = pd.read_csv(imp_path, index_col=0)
            if shap_col in imp.columns:
                imp_frames.append(imp[shap_col].rename(f"fold_{fold}"))
        else:
            logger.warning(f"fold {fold}: feature_importance_shap.csv not found.")

    if imp_frames:
        imp_combined = pd.concat(imp_frames, axis=1)
        imp_combined["mean_shap"] = imp_combined.mean(axis=1)
        imp_combined["std_shap"] = imp_combined.std(axis=1)
        imp_combined["cv_stability"] = imp_combined["std_shap"] / (imp_combined["mean_shap"] + 1e-10)
        imp_combined = imp_combined.sort_values("mean_shap", ascending=False)
        imp_combined.to_csv(out_dir / "shap_importance_averaged.csv")

        top10 = imp_combined["mean_shap"].head(10)
        logger.info(f"Top 10 features (mean |SHAP| across {len(imp_frames)} folds):")
        for feat, val in top10.items():
            stability = imp_combined.loc[feat, "cv_stability"]
            logger.info(f"  {feat:<40} {val:.5f}  (cv_stability={stability:.3f})")
    else:
        logger.warning("No SHAP importance files found — run shap_cv.py first.")

    # ── 3. Print summary table ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"CV Summary — Spectra {spectra}, Level {level}")
    print(f"{'=' * 60}")
    print(f"{'Fold':<6} {'Macro F1':>10} {'Best iter':>10} {'Best score':>12}")
    print("-" * 42)
    for _, row in metrics_df.iterrows():
        print(f"  {int(row['fold']):<4} {row['macro_f1']:>10.4f} "
              f"{int(row['best_iteration']):>10} {row['best_score']:>12.5f}")
    print("-" * 42)
    print(f"  {'MEAN':<4} {metrics_df['macro_f1'].mean():>10.4f} "
          f"{metrics_df['best_iteration'].mean():>10.1f} "
          f"{metrics_df['best_score'].mean():>12.5f}")
    print(f"  {'STD':<4} {metrics_df['macro_f1'].std():>10.4f} "
          f"{metrics_df['best_iteration'].std():>10.1f} "
          f"{metrics_df['best_score'].std():>12.5f}")
    print(f"{'=' * 60}\n")

    logger.info(f"CV summary saved to: {out_dir}")


if __name__ == "__main__":
    main()
