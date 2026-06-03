"""
hac_pipeline/evaluation/evaluator.py
--------------------------------------
Step 6  — ROI-level majority vote and external validation metrics.
Step 6a — Feature separation: ANOVA F-statistic + variance-weighted PCA loading.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from hac_pipeline.utils.io import classify_feature_family
from scipy.stats import f_oneway
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 6 — ROI majority vote and validation metrics
# ---------------------------------------------------------------------------


def compute_roi_majority_vote(
        pixel_df: pd.DataFrame,
        k: int,
        min_pixels: int,
) -> pd.DataFrame:
    df_k = pixel_df[pixel_df["k"] == k]
    results = []
    for roi_id, group in df_k.groupby("roi_ID"):
        n = len(group)
        if n < min_pixels:
            logger.warning(
                f"ROI '{roi_id}' has only {n} pixels for K={k} "
                f"(< min={min_pixels}); skipping."
            )
            continue
        counts = group["cluster_label"].value_counts()
        majority = counts.idxmax()
        pct = counts.max() / n * 100
        results.append({
            "roi_ID": roi_id,
            "k": k,
            "cluster_label": int(majority),
            "pct_majority": round(pct, 2),
            "n_pixels_sampled": n,
        })
    return pd.DataFrame(results)


def compute_validation_metrics(
        roi_df: pd.DataFrame,
        k: int,
        silhouette_scores: dict[int, float],
        output_dir: Path,
) -> dict:
    le = LabelEncoder()
    roi_true = le.fit_transform(roi_df["roi_ID"])
    roi_pred = roi_df["cluster_label"].values

    ari = adjusted_rand_score(roi_true, roi_pred)
    ami = adjusted_mutual_info_score(roi_true, roi_pred)
    nmi = normalized_mutual_info_score(roi_true, roi_pred)
    hom = homogeneity_score(roi_true, roi_pred)
    com = completeness_score(roi_true, roi_pred)
    v = v_measure_score(roi_true, roi_pred)
    sil = silhouette_scores.get(k)

    metrics = {
        "k": k,
        "n_rois": len(roi_df),
        "ari": round(ari, 4),
        "ami": round(ami, 4),
        "nmi": round(nmi, 4),
        "homogeneity": round(hom, 4),
        "completeness": round(com, 4),
        "v_measure": round(v, 4),
        "silhouette": round(sil, 4) if sil is not None else None,
        "roi_majority_vote": roi_df.to_dict(orient="records"),
    }

    with open(output_dir / f"metrics_k{k}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(
        f"K={k}: ARI={ari:.4f}, AMI={ami:.4f}, NMI={nmi:.4f}, "
        f"Hom={hom:.4f}, Com={com:.4f}, V={v:.4f}, "
        f"sil={f'{sil:.4f}' if sil is not None else 'N/A'}. "
        f"Saved: metrics_k{k}.json"
    )
    return metrics


def compute_all_roi_metrics(
        pixel_df: pd.DataFrame,
        k_values: list[int],
        min_pixels: int,
        silhouette_scores: dict[int, float],
        output_dir: Path,
) -> pd.DataFrame:
    all_roi_dfs = []
    for k in k_values:
        roi_df = compute_roi_majority_vote(pixel_df, k=k, min_pixels=min_pixels)
        compute_validation_metrics(roi_df, k=k,
                                   silhouette_scores=silhouette_scores,
                                   output_dir=output_dir)
        all_roi_dfs.append(roi_df)

    combined = pd.concat(all_roi_dfs, ignore_index=True)
    combined.to_csv(output_dir / "roi_clusters.csv", index=False)
    logger.info(f"ROI clusters saved: {output_dir / 'roi_clusters.csv'} "
                f"({len(combined):,} rows).")
    return combined


# ---------------------------------------------------------------------------
# Step 6a — Feature separation
# ---------------------------------------------------------------------------


def compute_feature_separation(
        df_sample: pd.DataFrame,
        feature_cols: list[str],
        pixel_df: pd.DataFrame,
        k: int,
        n_top: int,
        output_dir: Path,
        xgb_shap_dir: Path = None,
) -> pd.DataFrame:
    """ANOVA F-statistic per feature, sorted by f_rank.

    Parameters
    ----------
    df_sample : pd.DataFrame
        Full sampled pixel DataFrame with feature columns.
    feature_cols : list[str]
        All feature columns used in clustering.
    pixel_df : pd.DataFrame
        Long-format pixel cluster assignments.
    k : int
        K value for cluster labels.
    n_top : int
        Number of top features shown in dot plot (unused here, kept for signature compat).
    output_dir : Path
    xgb_shap_dir : Path, optional
        If provided, cross-reference SHAP rankings as a post-hoc column.
    """
    # Attach cluster labels for this K
    df_k = df_sample.copy()
    df_k["cluster_label"] = (
        pixel_df[pixel_df["k"] == k]
        .sort_values("pixel_idx")["cluster_label"]
        .values
    )
    unique_clusters = df_k["cluster_label"].unique()

    # Measure 1: ANOVA F-statistic
    f_stats, p_values = {}, {}
    for feat in feature_cols:
        groups = [
            df_k.loc[df_k["cluster_label"] == c, feat].values
            for c in unique_clusters
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            f_stats[feat] = np.nan
            p_values[feat] = np.nan
        else:
            f_val, p_val = f_oneway(*groups)
            f_stats[feat] = float(f_val)
            p_values[feat] = float(p_val)

    # Assemble — F-statistic only, no PCA loading rank
    records = []
    for feat in feature_cols:
        records.append({
            "feature": feat,
            "family": classify_feature_family(feat),
            "f_statistic": f_stats.get(feat, np.nan),
            "p_value": p_values.get(feat, np.nan),
        })

    sep_df = pd.DataFrame(records)
    sep_df["f_rank"] = sep_df["f_statistic"].rank(
        ascending=False, method="min", na_option="bottom").astype(int)

    # Optional post-hoc SHAP cross-reference
    if xgb_shap_dir is not None:
        shap_path = xgb_shap_dir / "feature_importance_shap.csv"
        if shap_path.exists():
            shap_df = pd.read_csv(shap_path, index_col=0)
            turf_cols = [c for c in shap_df.columns if "turf_algae" in c]
            if turf_cols:
                shap_importance = shap_df[turf_cols].mean(axis=1).rename("shap_mean_abs")
                sep_df = sep_df.merge(
                    shap_importance.reset_index().rename(columns={"index": "feature"}),
                    on="feature", how="left"
                )
                sep_df["shap_rank"] = sep_df["shap_mean_abs"].rank(
                    ascending=False, method="min", na_option="bottom").astype(int)
                logger.info("SHAP cross-reference added to feature separation.")
        else:
            logger.warning(f"SHAP file not found at {shap_path}; skipping cross-reference.")

    sep_df = sep_df.sort_values("f_rank").reset_index(drop=True)
    out_path = output_dir / f"feature_separation_k{k}.csv"
    sep_df.to_csv(out_path, index=False)
    logger.info(
        f"Feature separation saved: {out_path}. "
        f"Top feature: {sep_df.iloc[0]['feature']} "
        f"(f_rank=1, f_statistic={sep_df.iloc[0]['f_statistic']:.1f})."
    )
    return sep_df