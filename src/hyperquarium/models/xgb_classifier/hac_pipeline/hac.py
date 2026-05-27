"""
hac.py — HAC pipeline entry point
-----------------------------------
Run hierarchical agglomerative clustering on turf_algae pixels for one spectra type.

Usage:
    python3 hac.py <parquet_path> --spectra A --labelset pilot

Arguments:
    parquet_path    Path to compiled pixel-level parquet (same as xgb_pipeline input).
    --spectra       Spectra type: A, B, C, or D.
    --labelset      Labelset name for remap_labels() (default: pilot).
    --xgb_output    Root directory of xgb_pipeline outputs (default: outputs/).
    --hac_output    Root directory for HAC outputs (default: outputs/hac/).
    --mapping       Path to labelset_mapping.csv (default: data/labelset_mapping.csv).

PBS array usage:
    #PBS -J 0-3
    SPECTRA=(A B C D)
    S=${SPECTRA[$PBS_ARRAY_INDEX]}
    python3 -u hac.py data/compiled.parquet --spectra $S
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")  # Must be set before any other matplotlib import

from pathlib import Path

from hac_pipeline.config.config import HACConfig
from hac_pipeline.data.loader import load_turf_sample
from hac_pipeline.evaluation.evaluator import (
    compute_all_roi_metrics,
    compute_feature_separation,
)
from hac_pipeline.features.feature_selector import (
    select_features,
    validate_features_in_dataframe,
)
from hac_pipeline.models.clusterer import (
    assign_all_clusters,
    fit_pca,
    fit_ward_linkage,
    silhouette_sweep,
)
from hac_pipeline.utils.io import load_json, save_json
from hac_pipeline.utils.logger import setup_logger
from hac_pipeline.visualisations.plots import (
    plot_cluster_accuracy,
    plot_dendrogram,
    plot_feature_separation,
    plot_majority_vote_heatmap,
    plot_umap,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HAC pipeline: unsupervised clustering of turf algae pixels."
    )
    parser.add_argument("parquet_path", type=Path,
                        help="Path to compiled pixel-level parquet.")
    parser.add_argument("--spectra", required=True, choices=["A", "B", "C", "D"],
                        help="Spectra type (A, B, C, or D).")
    parser.add_argument("--labelset", default="pilot",
                        help="Labelset name for remap_labels() (default: pilot).")
    parser.add_argument("--xgb_output", type=Path, default=Path("outputs"),
                        help="Root directory of xgb_pipeline outputs (default: outputs/).")
    parser.add_argument("--hac_output", type=Path, default=Path("outputs/hac"),
                        help="Root directory for HAC outputs (default: outputs/hac/).")
    parser.add_argument("--mapping", type=Path, default=Path("data/labelset_mapping.csv"),
                        help="Path to labelset_mapping.csv.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path derivation
# ---------------------------------------------------------------------------


def derive_paths(
        args: argparse.Namespace,
        cfg: HACConfig,
) -> HACConfig:
    """Populate runtime paths in HACConfig from CLI args."""
    cfg.xgb_shap_dir = args.xgb_output / f"spectra_{args.spectra}" / "level_4"
    cfg.output_dir = args.hac_output / f"spectra_{args.spectra}"
    cfg.held_out_summary_path = (
            args.xgb_output / f"spectra_{args.spectra}" / "held_out_accuracy_summary.csv"
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# Poor ROI loading
# ---------------------------------------------------------------------------


def load_poor_rois(
        held_out_path: Path,
        threshold: float,
) -> set[str]:
    """Load ROIs with prop_correct below threshold from held_out_accuracy_summary.csv.

    Returns empty set with a warning if the file doesn't exist.
    """
    import pandas as pd

    if not held_out_path.exists():
        logger.warning(
            f"held_out_accuracy_summary.csv not found at {held_out_path}. "
            f"Dendrogram will not annotate poorly-classified ROIs."
        )
        return set()

    df = pd.read_csv(held_out_path)
    poor = df[df["prop_correct"] < threshold]["roi_ID"].unique().tolist()
    logger.info(
        f"Poor ROIs (prop_correct < {threshold}): {len(poor)} identified."
    )
    return set(poor)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = derive_paths(args, HACConfig())

    # Logging
    setup_logger(cfg.output_dir, spectra=args.spectra)
    logger.info(f"=== HAC Pipeline — Spectra {args.spectra} ===")
    logger.info(f"Parquet: {args.parquet_path}")
    logger.info(f"Output:  {cfg.output_dir}")
    logger.info(f"Config:  {cfg}")

    # ------------------------------------------------------------------
    # Step 1 — Load and sample
    # ------------------------------------------------------------------
    logger.info("--- Step 1: Data loading ---")
    df_sample = load_turf_sample(
        parquet_path=args.parquet_path,
        mapping_path=args.mapping,
        pixels_per_roi=cfg.pixels_per_roi,
        random_seed=cfg.random_seed,
        labelset=args.labelset,
        nan_col_threshold=cfg.nan_col_threshold,
    )

    # ------------------------------------------------------------------
    # Step 2 — Feature selection
    # ------------------------------------------------------------------
    logger.info("--- Step 2: Feature selection ---")
    selected_features = select_features(
        shap_dir=cfg.xgb_shap_dir,
        n_top=cfg.n_top_features,
        turf_substring=cfg.turf_label_substring,
        output_path=cfg.output_dir / "selected_features.json",
    )
    selected_features = validate_features_in_dataframe(selected_features, df_sample)

    X_raw = df_sample[selected_features].values
    roi_ids = df_sample["roi_ID"].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 3 — PCA
    # ------------------------------------------------------------------
    logger.info("--- Step 3: PCA pre-reduction ---")
    X_pca, pca, scaler = fit_pca(
        X=X_raw,
        variance_threshold=cfg.pca_variance_threshold,
        output_dir=cfg.output_dir,
    )

    # ------------------------------------------------------------------
    # Step 4 — Ward linkage
    # ------------------------------------------------------------------
    logger.info("--- Step 4: Ward linkage ---")
    Z = fit_ward_linkage(X_pca=X_pca, output_dir=cfg.output_dir)

    # ------------------------------------------------------------------
    # Step 5 — Silhouette sweep + dendrogram cuts
    # ------------------------------------------------------------------
    logger.info("--- Step 5: Silhouette sweep and cluster assignment ---")
    best_k = silhouette_sweep(
        X_pca=X_pca,
        Z=Z,
        k_min=cfg.silhouette_k_min,
        k_max=cfg.silhouette_k_max,
        output_dir=cfg.output_dir,
    )

    k_values = sorted(set(cfg.k_values + [best_k]))
    logger.info(f"K values to evaluate: {k_values} (best_k={best_k})")

    pixel_df = assign_all_clusters(
        X_pca=X_pca,
        Z=Z,
        k_values=k_values,
        roi_ids=roi_ids,
        output_dir=cfg.output_dir,
    )

    silhouette_data = load_json(cfg.output_dir / "silhouette_scores.json")
    silhouette_scores = {int(k): v for k, v in silhouette_data["scores"].items()}

    # ------------------------------------------------------------------
    # Step 6 — ROI-level validation
    # ------------------------------------------------------------------
    logger.info("--- Step 6: ROI-level validation ---")
    roi_clusters = compute_all_roi_metrics(
        pixel_df=pixel_df,
        k_values=k_values,
        min_pixels=cfg.majority_vote_min_pixels,
        silhouette_scores=silhouette_scores,
        output_dir=cfg.output_dir,
    )

    # ------------------------------------------------------------------
    # Step 6a — Feature separation
    # ------------------------------------------------------------------
    logger.info("--- Step 6a: Feature separation analysis ---")
    import pandas as pd

    for k in k_values:
        df_k_pixels = df_sample.copy()
        df_k_pixels["cluster_label"] = (
            pixel_df[pixel_df["k"] == k]
            .sort_values("pixel_idx")["cluster_label"]
            .values
        )
        sep_df = compute_feature_separation(
            pixel_df_full=df_k_pixels,
            feature_cols=selected_features,
            pca=pca,
            k=k,
            n_top=cfg.n_top_separation_features,
            output_dir=cfg.output_dir,
        )

    # ------------------------------------------------------------------
    # Step 7 — Visualisations
    # ------------------------------------------------------------------
    logger.info("--- Step 7: Visualisations ---")

    # 7a — Dendrogram (K-independent; drawn once)
    poor_rois = load_poor_rois(
        cfg.held_out_summary_path,
        threshold=cfg.poor_roi_prop_correct_threshold,
    )
    unique_rois = roi_ids.unique().tolist()
    plot_dendrogram(
        Z=Z,
        roi_ids=unique_rois,
        poor_rois=poor_rois,
        k_values=k_values,
        output_dir=cfg.output_dir,
    )

    # Per-K visualisations
    for k in k_values:
        logger.info(f"  Visualisations for K={k}...")

        # 7b — UMAP
        plot_umap(
            X_pca=X_pca,
            roi_ids=roi_ids,
            pixel_df=pixel_df,
            k=k,
            output_dir=cfg.output_dir,
        )

        # 7c — Cluster accuracy strip plot
        plot_cluster_accuracy(
            roi_clusters=roi_clusters,
            held_out_path=cfg.held_out_summary_path,
            k=k,
            output_dir=cfg.output_dir,
        )

        # 7d — Majority vote heatmap
        plot_majority_vote_heatmap(
            pixel_df=pixel_df,
            k=k,
            output_dir=cfg.output_dir,
        )

        # 7e — Feature separation dot plot
        sep_df = pd.read_csv(cfg.output_dir / f"feature_separation_k{k}.csv")
        plot_feature_separation(
            sep_df=sep_df,
            k=k,
            n_top=cfg.n_top_separation_features,
            output_dir=cfg.output_dir,
        )

    logger.info(f"=== HAC Pipeline complete — Spectra {args.spectra} ===")


if __name__ == "__main__":
    main()