"""
hac.py — HAC pipeline entry point
-----------------------------------
Unsupervised Ward linkage clustering of turf_algae pixels.
All valid feature columns used — no SHAP pre-selection.

Usage:
    python3 hac.py <parquet_path> --spectra A --labelset pilot
    python3 hac.py <parquet_path> --spectra A --overwrite false

PBS array:
    #PBS -J 0-3
    #PBS -l ncpus=4,mem=128gb,walltime=04:00:00
    SPECTRA=(A B C D)
    S=${SPECTRA[$PBS_ARRAY_INDEX]}
    python3 -u hac.py data/compiled.parquet --spectra $S
"""

from __future__ import annotations

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import numpy as np
import pandas as pd

from hac_pipeline.config.config import HACConfig
from hac_pipeline.data.loader import load_turf_sample
from hac_pipeline.evaluation.evaluator import (
    compute_all_roi_metrics,
    compute_feature_separation,
)
from hac_pipeline.models.clusterer import (
    assign_all_clusters,
    fit_pca,
    fit_ward_linkage,
    load_pca_from_checkpoint,
    silhouette_sweep,
)
from hac_pipeline.utils.io import load_json
from hac_pipeline.utils.logger import setup_logger
from hac_pipeline.visualisations.plots import (
    plot_cluster_accuracy,
    plot_dendrogram,
    plot_majority_vote_heatmap,
    plot_roi_assignment_summary,
    plot_roi_metrics_combined,
    plot_spectral_separation,
    plot_spatial_separation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HAC pipeline: unsupervised clustering of turf algae pixels."
    )
    p.add_argument("parquet_path", type=Path)
    p.add_argument("--spectra", required=True, choices=["A", "B", "C", "D"])
    p.add_argument("--labelset", default="pilot")
    p.add_argument("--xgb_output", type=Path, default=Path("outputs"))
    p.add_argument("--hac_output", type=Path, default=Path("outputs/hac"))
    p.add_argument("--mapping", type=Path, default=Path("data/labelset_mapping.csv"))
    p.add_argument(
        "--overwrite", default="true", choices=["true", "false"],
        help="true (default): rerun all steps. "
             "false: skip steps whose output files already exist.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def should_run(output_path: Path, overwrite: bool) -> bool:
    if overwrite:
        return True
    if output_path.exists():
        logger.info(f"Checkpoint: skipping (exists) — {output_path.name}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    overwrite = args.overwrite.lower() == "true"

    cfg = HACConfig()
    cfg.xgb_shap_dir = args.xgb_output / f"spectra_{args.spectra}" / "level_4"
    cfg.output_dir = args.hac_output / f"spectra_{args.spectra}"
    cfg.held_out_summary_path = (
            args.xgb_output / f"spectra_{args.spectra}" / "held_out_accuracy_summary.csv"
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir

    setup_logger(out, spectra=args.spectra)
    logger.info(f"=== HAC Pipeline — Spectra {args.spectra} ===")
    logger.info(f"Parquet:   {args.parquet_path}")
    logger.info(f"Output:    {out}")
    logger.info(f"Overwrite: {overwrite}")

    # ------------------------------------------------------------------
    # Step 1 — Load, filter, clean, sample
    # Always runs (fast; needed in-memory regardless of checkpoints)
    # ------------------------------------------------------------------
    logger.info("--- Step 1: Data loading ---")
    df_sample, feat_cols, nan_fractions = load_turf_sample(
        parquet_path=args.parquet_path,
        mapping_path=args.mapping,
        pixels_per_roi=cfg.pixels_per_roi,
        random_seed=cfg.random_seed,
        nan_col_threshold=cfg.nan_col_threshold,
        labelset=args.labelset,
    )
    roi_ids = df_sample["roi_ID"].reset_index(drop=True)
    n_rois = roi_ids.nunique()

    # Save feature column audit record
    if should_run(out / "feature_columns.json", overwrite):
        feat_record = {
            "n_features": len(feat_cols),
            "features": [
                {"name": f, "nan_fraction_before_drop": nan_fractions.get(f, 0.0)}
                for f in feat_cols
            ],
        }
        with open(out / "feature_columns.json", "w") as fh:
            json.dump(feat_record, fh, indent=2)
        logger.info(f"Feature columns saved: {out / 'feature_columns.json'} "
                    f"({len(feat_cols)} features).")

    # ------------------------------------------------------------------
    # Step 3 — PCA
    # ------------------------------------------------------------------
    logger.info("--- Step 3: PCA pre-reduction ---")
    if should_run(out / "X_pca.npy", overwrite):
        X_raw = df_sample[feat_cols].values
        X_pca, pca, _ = fit_pca(
            X=X_raw,
            feature_cols=feat_cols,
            variance_threshold=cfg.pca_variance_threshold,
            output_dir=out,
        )
    else:
        X_pca, pca = load_pca_from_checkpoint(out)

    # ------------------------------------------------------------------
    # Step 4 — Ward linkage
    # ------------------------------------------------------------------
    logger.info("--- Step 4: Ward linkage ---")
    if should_run(out / "linkage_matrix.npy", overwrite):
        Z = fit_ward_linkage(X_pca=X_pca, output_dir=out)
    else:
        Z = np.load(out / "linkage_matrix.npy")
        logger.info(f"Loaded linkage matrix from checkpoint: shape {Z.shape}.")

    # ------------------------------------------------------------------
    # Step 5 — Silhouette sweep + cluster assignment
    # ------------------------------------------------------------------
    logger.info("--- Step 5: Silhouette sweep and cluster assignment ---")
    if should_run(out / "silhouette_scores.json", overwrite):
        best_k = silhouette_sweep(
            X_pca=X_pca, Z=Z,
            k_min=cfg.silhouette_k_min,
            k_max=cfg.silhouette_k_max,
            output_dir=out,
        )
    else:
        best_k = load_json(out / "silhouette_scores.json")["best_k"]
        logger.info(f"Loaded silhouette scores from checkpoint. best_k={best_k}.")

    k_rois = n_rois  # K matching number of unique ROIs
    k_values = sorted(set(cfg.k_values + [best_k, k_rois]))
    logger.info(f"K values: {k_values} (best_k={best_k}, k_rois={k_rois})")

    if should_run(out / "pixel_clusters.parquet", overwrite):
        pixel_df = assign_all_clusters(
            Z=Z, k_values=k_values, roi_ids=roi_ids, output_dir=out,
        )
    else:
        pixel_df = pd.read_parquet(out / "pixel_clusters.parquet")
        logger.info(f"Loaded pixel clusters from checkpoint: {len(pixel_df):,} rows.")

    silhouette_scores = {
        int(k): v
        for k, v in load_json(out / "silhouette_scores.json")["scores"].items()
    }

    # ------------------------------------------------------------------
    # Step 6 — ROI-level validation
    # ------------------------------------------------------------------
    logger.info("--- Step 6: ROI-level validation ---")
    if should_run(out / "roi_clusters.csv", overwrite):
        roi_clusters = compute_all_roi_metrics(
            pixel_df=pixel_df,
            k_values=k_values,
            min_pixels=cfg.majority_vote_min_pixels,
            silhouette_scores=silhouette_scores,
            output_dir=out,
        )
    else:
        roi_clusters = pd.read_csv(out / "roi_clusters.csv")
        logger.info(f"Loaded ROI clusters from checkpoint: {len(roi_clusters):,} rows.")

    # ------------------------------------------------------------------
    # Step 6a — Feature separation
    # ------------------------------------------------------------------
    logger.info("--- Step 6a: Feature separation ---")
    for k in k_values:
        if should_run(out / f"feature_separation_k{k}.csv", overwrite):
            compute_feature_separation(
                df_sample=df_sample,
                feature_cols=feat_cols,
                pixel_df=pixel_df,
                pca=pca,
                k=k,
                n_top=20,
                output_dir=out,
                xgb_shap_dir=cfg.xgb_shap_dir,
            )

    # ------------------------------------------------------------------
    # Step 7 — Visualisations
    # ------------------------------------------------------------------
    logger.info("--- Step 7: Visualisations ---")

    # 7a — Dendrogram (K-independent)
    if should_run(out / "dendrogram.png", overwrite):
        plot_dendrogram(
            Z=Z,
            roi_ids=roi_ids.unique().tolist(),
            k_values=k_values,
            output_dir=out,
        )

    # Per-K figures
    for k in k_values:
        logger.info(f"  Visualisations for K={k}...")

        if should_run(out / f"cluster_accuracy_k{k}.png", overwrite):
            plot_cluster_accuracy(
                roi_clusters=roi_clusters,
                held_out_path=cfg.held_out_summary_path,
                k=k, output_dir=out,
            )
        if should_run(out / f"majority_vote_heatmap_k{k}.png", overwrite):
            plot_majority_vote_heatmap(pixel_df=pixel_df, k=k, output_dir=out)

        # Summary histogram for all K values
        if should_run(out / f"roi_assignment_summary_k{k}.png", overwrite):
            metrics = load_json(out / f"metrics_k{k}.json")
            plot_roi_assignment_summary(
                roi_clusters=roi_clusters,
                k=k,
                metrics=metrics,
                output_dir=out,
            )

        if should_run(out / f"feature_separation_spectral_k{k}.png", overwrite):
            sep_df = pd.read_csv(out / f"feature_separation_k{k}.csv")
            plot_spectral_separation(sep_df=sep_df, k=k, output_dir=out)

        if should_run(out / f"feature_separation_spatial_k{k}.png", overwrite):
            if not (out / f"feature_separation_k{k}.csv").exists():
                sep_df = pd.read_csv(out / f"feature_separation_k{k}.csv")
            plot_spatial_separation(sep_df=sep_df, k=k, output_dir=out)

    # Combined metrics table across all K values (one PNG + one CSV)
    if should_run(out / "roi_metrics_combined.png", overwrite):
        metrics_by_k = {k: load_json(out / f"metrics_k{k}.json") for k in k_values}
        plot_roi_metrics_combined(
            roi_clusters=roi_clusters,
            k_values=k_values,
            metrics_by_k=metrics_by_k,
            output_dir=out,
        )

    logger.info(f"=== HAC Pipeline complete --- Spectra {args.spectra} ===")


if __name__ == "__main__":
    main()