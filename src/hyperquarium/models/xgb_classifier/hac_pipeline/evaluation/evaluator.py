"""
hac_pipeline/evaluation/evaluator.py
--------------------------------------
Steps 6 and 6a of the HAC pipeline design.

Step 6  — ROI-level majority vote and external validation metrics
           (ARI, NMI, V-measure).
Step 6a — Feature separation analysis: ANOVA F-statistic + variance-weighted
           PCA loading magnitude, combined into feature_separation_k{K}.csv.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 6 — ROI-level majority vote and metrics
# ---------------------------------------------------------------------------


def compute_roi_majority_vote(
        pixel_df: pd.DataFrame,
        k: int,
        min_pixels: int,
) -> pd.DataFrame:
    """Compute majority cluster label per ROI for a given K.

    Parameters
    ----------
    pixel_df : pd.DataFrame
        Long-format pixel assignments with columns roi_ID, k, cluster_label.
    k : int
        The K value to evaluate.
    min_pixels : int
        Minimum pixel count per ROI required to report a result.

    Returns
    -------
    pd.DataFrame
        One row per ROI: roi_ID | k | cluster_label | pct_majority | n_pixels_sampled
    """
    df_k = pixel_df[pixel_df["k"] == k].copy()

    results = []
    for roi_id, group in df_k.groupby("roi_ID"):
        n_pixels = len(group)
        if n_pixels < min_pixels:
            logger.warning(
                f"ROI '{roi_id}' has only {n_pixels} pixels for K={k} "
                f"(< min={min_pixels}); skipping majority vote."
            )
            continue

        cluster_counts = group["cluster_label"].value_counts()
        majority_cluster = cluster_counts.idxmax()
        pct_majority = cluster_counts.max() / n_pixels * 100

        results.append(
            {
                "roi_ID": roi_id,
                "k": k,
                "cluster_label": int(majority_cluster),
                "pct_majority": round(pct_majority, 2),
                "n_pixels_sampled": n_pixels,
            }
        )

    return pd.DataFrame(results)


def compute_validation_metrics(
        roi_df: pd.DataFrame,
        k: int,
        silhouette_scores: dict[int, float],
        output_dir: Path,
) -> dict:
    """Compute ARI, NMI, V-measure for ROI-level cluster assignments.

    Parameters
    ----------
    roi_df : pd.DataFrame
        ROI majority vote results for this K (from compute_roi_majority_vote).
    k : int
        The K value being evaluated.
    silhouette_scores : dict
        Silhouette score per K from the silhouette sweep.
    output_dir : Path
        Where to save metrics_k{K}.json.

    Returns
    -------
    dict
        Metrics dictionary (also saved to JSON).
    """
    le = LabelEncoder()
    roi_true = le.fit_transform(roi_df["roi_ID"])  # integer-encoded roi_ID
    roi_pred = roi_df["cluster_label"].values

    ari = adjusted_rand_score(roi_true, roi_pred)
    nmi = normalized_mutual_info_score(roi_true, roi_pred)
    v_meas = v_measure_score(roi_true, roi_pred)
    sil = silhouette_scores.get(k, None)

    metrics = {
        "k": k,
        "n_rois": len(roi_df),
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "v_measure": round(v_meas, 4),
        "silhouette": round(sil, 4) if sil is not None else None,
        "roi_majority_vote": roi_df.to_dict(orient="records"),
    }

    metrics_path = output_dir / f"metrics_k{k}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(
        f"K={k}: ARI={ari:.4f}, NMI={nmi:.4f}, V={v_meas:.4f}, "
        f"sil={sil:.4f if sil else 'N/A'}. Saved: {metrics_path}"
    )
    return metrics


def compute_all_roi_metrics(
        pixel_df: pd.DataFrame,
        k_values: list[int],
        min_pixels: int,
        silhouette_scores: dict[int, float],
        output_dir: Path,
) -> pd.DataFrame:
    """Run majority vote and validation for all K values.

    Returns
    -------
    pd.DataFrame
        All ROI-level results concatenated (long format, all K).
    """
    all_roi_dfs = []
    for k in k_values:
        roi_df = compute_roi_majority_vote(pixel_df, k=k, min_pixels=min_pixels)
        compute_validation_metrics(
            roi_df, k=k, silhouette_scores=silhouette_scores, output_dir=output_dir
        )
        all_roi_dfs.append(roi_df)

    combined = pd.concat(all_roi_dfs, ignore_index=True)
    roi_path = output_dir / "roi_clusters.csv"
    combined.to_csv(roi_path, index=False)
    logger.info(f"ROI cluster assignments saved: {roi_path} ({len(combined):,} rows).")
    return combined


# ---------------------------------------------------------------------------
# Step 6a — Feature separation analysis
# ---------------------------------------------------------------------------


def compute_feature_separation(
        pixel_df_full: pd.DataFrame,
        feature_cols: list[str],
        pca: object,  # sklearn PCA
        k: int,
        n_top: int,
        output_dir: Path,
) -> pd.DataFrame:
    """Compute ANOVA F-statistic and variance-weighted PCA loading magnitude
    for each selected feature, then rank and combine into feature_separation_k{K}.csv.

    Parameters
    ----------
    pixel_df_full : pd.DataFrame
        Full sampled pixel DataFrame (pre-PCA) with feature columns and
        a 'cluster_label_k{k}' column or a pixel_df filtered to this K.
    feature_cols : list[str]
        The top-N selected feature names.
    pca : sklearn PCA
        Fitted PCA from Step 3 (for loadings and explained_variance_ratio_).
    k : int
        K value — used only for naming output files.
    n_top : int
        Number of top features to include in the dot plot output.
    output_dir : Path
        Where to save feature_separation_k{K}.csv.

    Returns
    -------
    pd.DataFrame
        Feature separation table, sorted by combined_rank ascending.
    """
    from hac_pipeline.features.feature_selector import _classify_feature_family

    # Cluster labels for this K (from pixel_df_full which must include 'cluster_label')
    cluster_labels = pixel_df_full["cluster_label"].values
    unique_clusters = np.unique(cluster_labels)

    # ---- Measure 1: ANOVA F-statistic per feature ----
    f_stats = {}
    p_values = {}
    for feat in feature_cols:
        groups = [
            pixel_df_full.loc[pixel_df_full["cluster_label"] == c, feat].values
            for c in unique_clusters
        ]
        # Drop empty groups (shouldn't happen but guard anyway)
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            f_stats[feat] = np.nan
            p_values[feat] = np.nan
        else:
            f_val, p_val = f_oneway(*groups)
            f_stats[feat] = float(f_val)
            p_values[feat] = float(p_val)

    # ---- Measure 2: Variance-weighted PCA loading magnitude ----
    # pca.components_ shape: (n_components, n_features_total)
    # We need to align feature_cols with the PCA feature order
    # pca was fitted on X_scaled which had columns = feature_cols in order
    ev_ratio = pca.explained_variance_ratio_  # (n_components,)
    loadings = pca.components_  # (n_components, n_features)

    # Weighted loading magnitude per feature
    # sqrt( sum_over_pcs( loading^2 * explained_variance_ratio ) )
    weighted_loadings = np.sqrt(
        (loadings ** 2 * ev_ratio[:, np.newaxis]).sum(axis=0)
    )  # shape: (n_features,)

    # ---- Assemble results ----
    records = []
    for i, feat in enumerate(feature_cols):
        records.append(
            {
                "feature": feat,
                "family": _classify_feature_family(feat),
                "f_statistic": f_stats.get(feat, np.nan),
                "p_value": p_values.get(feat, np.nan),
                "weighted_loading": float(weighted_loadings[i]),
            }
        )

    sep_df = pd.DataFrame(records)

    # Rank (1 = most separating; ascending F-statistic rank = DESCENDING f value)
    sep_df["f_rank"] = sep_df["f_statistic"].rank(
        ascending=False, method="min", na_option="bottom"
    ).astype(int)
    sep_df["loading_rank"] = sep_df["weighted_loading"].rank(
        ascending=False, method="min"
    ).astype(int)
    sep_df["combined_rank"] = (
            (sep_df["f_rank"] + sep_df["loading_rank"]) / 2
    ).round(1)

    sep_df = sep_df.sort_values("combined_rank").reset_index(drop=True)

    out_path = output_dir / f"feature_separation_k{k}.csv"
    sep_df.to_csv(out_path, index=False)
    logger.info(
        f"Feature separation saved: {out_path}. "
        f"Top feature: {sep_df.iloc[0]['feature']} "
        f"(combined_rank={sep_df.iloc[0]['combined_rank']})."
    )
    return sep_df


def _classify_feature_family(feature_name: str) -> str:
    """Classify a feature column name into spectral / glcm / sdiv.

    Mirrors classify_column() from xgb_pipeline/features/feature_family_classification.py.
    Adjust regexes to match actual column naming conventions in your compiled parquet.
    """
    import re

    if re.match(r"^\d+_nm$", feature_name):
        return "spectral"
    if re.match(r"^(contrast|energy|entropy|homogeneity)_window_\d+", feature_name):
        return "glcm"
    if re.match(r"^sdiv_", feature_name):
        return "sdiv"
    return "unknown"
