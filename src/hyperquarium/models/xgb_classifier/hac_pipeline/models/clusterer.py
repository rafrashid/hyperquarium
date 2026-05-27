"""
hac_pipeline/models/clusterer.py
---------------------------------
PCA pre-reduction, Ward linkage, dendrogram cuts, and silhouette sweep.

Steps 3–5 of the HAC pipeline design.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 3 — PCA pre-reduction
# ---------------------------------------------------------------------------


def fit_pca(
        X: np.ndarray,
        variance_threshold: float,
        output_dir: Path,
) -> tuple[np.ndarray, PCA, StandardScaler]:
    """Scale features and apply PCA.

    Parameters
    ----------
    X : np.ndarray
        Pixel × feature matrix (selected features only).
    variance_threshold : float
        Fraction of variance to retain (e.g. 0.95).
    output_dir : Path
        Where to save pca_loadings.csv.

    Returns
    -------
    X_pca : np.ndarray
        PCA-transformed pixel matrix.
    pca : PCA
        Fitted PCA object (retained for feature separation step).
    scaler : StandardScaler
        Fitted scaler (retained for any future transform calls).
    """
    logger.info(f"Scaling {X.shape[1]} features (StandardScaler).")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Fitting PCA (variance threshold={variance_threshold}).")
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)

    n_components = pca.n_components_
    variance_explained = pca.explained_variance_ratio_.sum()
    logger.info(
        f"PCA: {n_components} components retained, "
        f"{variance_explained * 100:.2f}% variance explained."
    )

    # Save loadings: (n_components × n_features)
    loadings_df = pd.DataFrame(
        pca.components_,
        index=[f"PC{i + 1}" for i in range(n_components)],
    )
    loadings_path = output_dir / "pca_loadings.csv"
    loadings_df.to_csv(loadings_path)
    logger.info(f"PCA loadings saved: {loadings_path}")

    return X_pca, pca, scaler


# ---------------------------------------------------------------------------
# Step 4 — Ward linkage
# ---------------------------------------------------------------------------


def fit_ward_linkage(
        X_pca: np.ndarray,
        output_dir: Path,
) -> np.ndarray:
    """Compute condensed distance matrix and Ward linkage.

    The linkage matrix Z is saved to disk so dendrogram cuts can be re-run
    without recomputing the expensive linkage step.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced pixel matrix.
    output_dir : Path
        Where to save linkage_matrix.npy.

    Returns
    -------
    Z : np.ndarray
        Linkage matrix of shape (N-1, 4).
    """
    N = X_pca.shape[0]
    n_pairs = N * (N - 1) // 2
    mem_gb = n_pairs * 4 / 1e9  # float32
    logger.info(
        f"Computing condensed distance matrix: N={N:,}, "
        f"{n_pairs:,} pairs, ~{mem_gb:.1f} GB (float32)."
    )

    # Compute as float32 to halve memory vs float64
    dist_condensed = pdist(X_pca.astype(np.float32), metric="euclidean")

    logger.info("Computing Ward linkage.")
    Z = linkage(dist_condensed, method="ward")

    linkage_path = output_dir / "linkage_matrix.npy"
    np.save(linkage_path, Z)
    logger.info(f"Linkage matrix saved: {linkage_path} (shape {Z.shape}).")

    # Free condensed matrix — no longer needed
    del dist_condensed

    return Z


# ---------------------------------------------------------------------------
# Step 5 — Dendrogram cuts and silhouette sweep
# ---------------------------------------------------------------------------


def cut_dendrogram(Z: np.ndarray, k: int) -> np.ndarray:
    """Cut linkage matrix Z to produce K clusters.

    Returns
    -------
    np.ndarray
        Integer cluster labels, 1-indexed, shape (N,).
    """
    return fcluster(Z, k, criterion="maxclust")


def silhouette_sweep(
        X_pca: np.ndarray,
        Z: np.ndarray,
        k_min: int,
        k_max: int,
        output_dir: Path,
) -> int:
    """Sweep K from k_min to k_max and select silhouette-best K.

    Silhouette is computed at pixel level on the full sample (no subsampling).
    Scores are saved to silhouette_scores.json.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced pixel matrix.
    Z : np.ndarray
        Ward linkage matrix.
    k_min, k_max : int
        Range of K values to sweep (inclusive).
    output_dir : Path
        Where to save silhouette_scores.json.

    Returns
    -------
    int
        K value with highest silhouette score.
    """
    import json

    logger.info(f"Silhouette sweep K={k_min}..{k_max} (pixel-level, full sample).")
    scores: dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        labels = cut_dendrogram(Z, k)
        score = silhouette_score(X_pca, labels, metric="euclidean")
        scores[k] = float(score)
        if k % 10 == 0:
            logger.info(f"  K={k}: silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    logger.info(
        f"Silhouette sweep complete. Best K={best_k} "
        f"(score={scores[best_k]:.4f})."
    )

    scores_path = output_dir / "silhouette_scores.json"
    with open(scores_path, "w") as f:
        json.dump({"scores": scores, "best_k": best_k}, f, indent=2)
    logger.info(f"Silhouette scores saved: {scores_path}")

    return best_k


def assign_all_clusters(
        X_pca: np.ndarray,
        Z: np.ndarray,
        k_values: list[int],
        roi_ids: pd.Series,
        output_dir: Path,
) -> pd.DataFrame:
    """Assign pixel-level cluster labels for all K values.

    Returns
    -------
    pd.DataFrame
        Long-format pixel cluster assignments:
        roi_ID | pixel_idx | k | cluster_label
    """
    records = []
    for k in k_values:
        labels = cut_dendrogram(Z, k)
        for idx, (roi_id, cluster) in enumerate(zip(roi_ids, labels)):
            records.append(
                {"roi_ID": roi_id, "pixel_idx": idx, "k": k, "cluster_label": int(cluster)}
            )

    pixel_df = pd.DataFrame(records)

    pixel_path = output_dir / "pixel_clusters.parquet"
    pixel_df.to_parquet(pixel_path, index=False)
    logger.info(
        f"Pixel cluster assignments saved: {pixel_path} "
        f"({len(pixel_df):,} rows, {len(k_values)} K values)."
    )
    return pixel_df