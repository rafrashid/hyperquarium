"""
hac_pipeline/models/clusterer.py
---------------------------------
PCA pre-reduction, Ward linkage, silhouette sweep, dendrogram cuts.
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


def fit_pca(
        X: np.ndarray,
        feature_cols: list[str],
        variance_threshold: float,
        output_dir: Path,
) -> tuple[np.ndarray, PCA, StandardScaler]:
    """Scale and PCA-reduce the feature matrix. Saves X_pca.npy and pca_loadings.csv."""
    logger.info(f"Scaling {X.shape[1]} features (StandardScaler).")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Fitting PCA (variance threshold={variance_threshold}).")
    pca = PCA(n_components=variance_threshold, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)
    logger.info(
        f"PCA: {pca.n_components_} components retained, "
        f"{pca.explained_variance_ratio_.sum() * 100:.2f}% variance explained."
    )

    # Save loadings: rows = PCs, cols = original features
    loadings_df = pd.DataFrame(
        pca.components_,
        index=[f"PC{i + 1}" for i in range(pca.n_components_)],
        columns=feature_cols,
    )
    loadings_df.to_csv(output_dir / "pca_loadings.csv")
    logger.info(f"PCA loadings saved: {output_dir / 'pca_loadings.csv'}")

    np.save(output_dir / "X_pca.npy", X_pca.astype(np.float32))
    logger.info(f"X_pca saved: {output_dir / 'X_pca.npy'} (shape {X_pca.shape})")

    return X_pca, pca, scaler


def load_pca_from_checkpoint(output_dir: Path) -> tuple[np.ndarray, PCA]:
    """Load X_pca.npy and reconstruct PCA object from pca_loadings.csv."""
    X_pca = np.load(output_dir / "X_pca.npy")
    loadings_df = pd.read_csv(output_dir / "pca_loadings.csv", index_col=0)
    pca = PCA()
    pca.components_ = loadings_df.values
    pca.n_components_ = loadings_df.shape[0]
    # Approximate explained_variance_ratio_ from X_pca column variances
    var = np.var(X_pca, axis=0)
    pca.explained_variance_ratio_ = var / var.sum()
    logger.info(f"PCA loaded from checkpoint: shape {X_pca.shape}, "
                f"{pca.n_components_} components.")
    return X_pca, pca


def fit_linkage(X_pca: np.ndarray, output_dir: Path) -> np.ndarray:
    """Compute condensed distance matrix and UPGMA (average) linkage.

    Saves linkage_matrix.npy for reuse without recomputation.
    UPGMA makes no assumption about cluster shape or size — appropriate
    for assemblage discovery where cluster sizes and shapes are unknown.
    """
    N = X_pca.shape[0]
    n_pairs = N * (N - 1) // 2
    mem_gb = n_pairs * 8 / 1e9  # float64
    logger.info(
        f"Computing condensed distance matrix: N={N:,}, "
        f"{n_pairs:,} pairs, ~{mem_gb:.1f} GB (float64)."
    )
    dist_condensed = pdist(X_pca, metric="euclidean")

    logger.info("Computing UPGMA linkage (average).")
    Z = linkage(dist_condensed, method="average")
    del dist_condensed

    np.save(output_dir / "linkage_matrix.npy", Z)
    logger.info(f"Linkage matrix saved: {output_dir / 'linkage_matrix.npy'} "
                f"(shape {Z.shape}).")
    return Z


def silhouette_sweep(
        X_pca: np.ndarray,
        Z: np.ndarray,
        k_min: int,
        k_max: int,
        output_dir: Path,
) -> int:
    """Sweep K and return silhouette-best K. Saves silhouette_scores.json."""
    import json
    logger.info(f"Silhouette sweep K={k_min}..{k_max} (pixel-level, full sample).")
    scores: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, k, criterion="maxclust")
        scores[k] = float(silhouette_score(X_pca, labels, metric="euclidean"))
        if k % 10 == 0:
            logger.info(f"  K={k}: silhouette={scores[k]:.4f}")

    best_k = max(scores, key=scores.get)
    logger.info(f"Silhouette sweep complete. Best K={best_k} (score={scores[best_k]:.4f}).")

    with open(output_dir / "silhouette_scores.json", "w") as f:
        json.dump({"scores": scores, "best_k": best_k}, f, indent=2)
    logger.info(f"Silhouette scores saved: {output_dir / 'silhouette_scores.json'}")
    return best_k


def assign_all_clusters(
        Z: np.ndarray,
        k_values: list[int],
        roi_ids: pd.Series,
        output_dir: Path,
) -> pd.DataFrame:
    """Cut dendrogram at each K and save pixel-level assignments (long format)."""
    records = []
    for k in k_values:
        labels = fcluster(Z, k, criterion="maxclust")
        for idx, (roi_id, cluster) in enumerate(zip(roi_ids, labels)):
            records.append({
                "roi_ID": roi_id,
                "pixel_idx": idx,
                "k": k,
                "cluster_label": int(cluster),
            })

    pixel_df = pd.DataFrame(records)
    pixel_df.to_parquet(output_dir / "pixel_clusters.parquet", index=False)
    logger.info(
        f"Pixel clusters saved: {output_dir / 'pixel_clusters.parquet'} "
        f"({len(pixel_df):,} rows, {len(k_values)} K values)."
    )
    return pixel_df