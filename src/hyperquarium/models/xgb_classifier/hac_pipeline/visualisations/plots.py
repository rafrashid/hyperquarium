"""
hac_pipeline/visualisations/plots.py
--------------------------------------
Steps 7a–7e: dendrogram, UMAP, strip plot, heatmap, feature separation dot plot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hac_pipeline.utils.io import classify_feature_family
from scipy.cluster.hierarchy import dendrogram

logger = logging.getLogger(__name__)

FAMILY_COLOURS = {
    "spectral": "#4C72B0",
    "glcm": "#DD8452",
    "sdiv": "#55A868",
    "unknown": "#8C8C8C",
}
DEFAULT_LEAF_COLOUR = "#333333"

# ---------------------------------------------------------------------------
# 7a — Dendrogram
# ---------------------------------------------------------------------------

def plot_dendrogram(
        Z: np.ndarray,
        roi_ids: list[str],
        k_values: list[int],
        output_dir: Path,
) -> None:
    n_rois = len(set(roi_ids))
    fig, ax = plt.subplots(figsize=(max(18, n_rois * 0.15), 10))

    dendrogram(
        Z,
        truncate_mode="lastp",
        p=n_rois,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=0,
        above_threshold_color=DEFAULT_LEAF_COLOUR,
    )

    ax.set_xticklabels(roi_ids, rotation=90, fontsize=5)

    _draw_k_cutlines(ax, Z, k_values, n_rois)

    ax.set_title("UPGMA Dendrogram: Turf Algae ROIs", fontsize=13)
    ax.set_xlabel("ROI ID", fontsize=10)
    ax.set_ylabel("Linkage distance", fontsize=10)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="grey", linestyle="--", linewidth=1, label=f"K={k} cut")
        for k in sorted(k_values)
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "dendrogram.png", dpi=150)
    plt.close(fig)
    logger.info(f"Dendrogram saved: {output_dir / 'dendrogram.png'}")


def _draw_k_cutlines(ax, Z, k_values, n_rois):
    n = Z.shape[0] + 1
    for k in sorted(k_values, reverse=True):
        if k >= n:
            continue
        merge_idx = n - k - 1
        if 0 <= merge_idx < len(Z):
            height = Z[merge_idx, 2]
            ax.axhline(y=height, color="grey", linestyle="--",
                       linewidth=0.8, alpha=0.7)
            ax.text(ax.get_xlim()[1] * 0.98, height,
                    f" K={k}", va="bottom", ha="right",
                    fontsize=7, color="grey")


# ---------------------------------------------------------------------------
# 7b — UMAP (two-panel: roi_ID vs cluster)
# ---------------------------------------------------------------------------

def plot_umap(
        X_pca: np.ndarray,
        roi_ids: pd.Series,
        pixel_df: pd.DataFrame,
        k: int,
        output_dir: Path,
        umap_random_state: int = 42,
) -> None:
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP plots.")
        return

    logger.info(f"Fitting UMAP (K={k}).")
    reducer = UMAP(n_components=2, random_state=umap_random_state,
                   n_jobs=1)
    embedding = reducer.fit_transform(X_pca)

    from sklearn.preprocessing import LabelEncoder
    roi_encoded = LabelEncoder().fit_transform(roi_ids)
    cluster_labels = (pixel_df[pixel_df["k"] == k]
                      .sort_values("pixel_idx")["cluster_label"].values)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].scatter(embedding[:, 0], embedding[:, 1],
                    c=roi_encoded, cmap="tab20", s=1, alpha=0.5, linewidths=0)
    axes[0].set_title("Coloured by ROI ID", fontsize=11)
    axes[0].axis("off")

    sc = axes[1].scatter(embedding[:, 0], embedding[:, 1],
                         c=cluster_labels, cmap="tab10", s=1, alpha=0.5,
                         linewidths=0, vmin=1, vmax=k)
    plt.colorbar(sc, ax=axes[1], label="Cluster", shrink=0.7)
    axes[1].set_title(f"Coloured by HAC cluster (K={k})", fontsize=11)
    axes[1].axis("off")

    fig.suptitle(f"UMAP — Turf Algae Pixels (K={k})", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_dir / f"umap_k{k}.png", dpi=150)
    plt.close(fig)
    logger.info(f"UMAP saved: {output_dir / f'umap_k{k}.png'}")


# ---------------------------------------------------------------------------
# 7c — Cluster vs held-out accuracy strip plot
# ---------------------------------------------------------------------------

def plot_cluster_accuracy(
        roi_clusters: pd.DataFrame,
        held_out_path: Path,
        k: int,
        output_dir: Path,
) -> None:
    if not held_out_path.exists():
        logger.warning(f"held_out_accuracy_summary.csv not found; skipping K={k} strip plot.")
        return

    held_out = pd.read_csv(held_out_path)
    df_k = roi_clusters[roi_clusters["k"] == k].copy()
    merged = df_k.merge(held_out[["roi_ID", "prop_correct"]], on="roi_ID", how="left")

    fig, ax = plt.subplots(figsize=(max(8, k * 1.2), 5))
    rng = np.random.default_rng(seed=42)
    for cluster_id, group in merged.groupby("cluster_label"):
        x = rng.uniform(cluster_id - 0.3, cluster_id + 0.3, size=len(group))
        ax.scatter(x, group["prop_correct"], alpha=0.6, s=30, color="#4C72B0")
        ax.hlines(group["prop_correct"].median(),
                  cluster_id - 0.4, cluster_id + 0.4,
                  colors="black", linewidths=1.5)

    ax.set_xticks(range(1, k + 1))
    ax.set_xticklabels([f"C{c}" for c in range(1, k + 1)])
    ax.set_xlabel("HAC Cluster", fontsize=11)
    ax.set_ylabel("Held-out prop_correct", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Held-out accuracy by cluster (K={k})", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / f"cluster_accuracy_k{k}.png", dpi=150)
    plt.close(fig)
    logger.info(f"Cluster accuracy plot saved: {output_dir / f'cluster_accuracy_k{k}.png'}")


# ---------------------------------------------------------------------------
# 7d_summary — ROI assignment confidence summary (for k=n_rois)
# ---------------------------------------------------------------------------

def plot_roi_assignment_summary(
        roi_clusters: pd.DataFrame,
        k: int,
        metrics: dict,
        output_dir: Path,
) -> None:
    """Histogram of pct_majority for all K values.
    Metrics table saved as a separate PNG via plot_roi_metrics_table().
    """
    df_k = roi_clusters[roi_clusters["k"] == k].copy()
    pct = df_k["pct_majority"].values

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pct, bins=20, range=(0, 100), color="#4C72B0", edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(x=float(np.median(pct)), color="black", linestyle="--",
               linewidth=1.2, label=f"Median: {np.median(pct):.1f}%")
    random_threshold = 100.0 / k
    ax.axvline(x=random_threshold, color="grey", linestyle=":",
               linewidth=1.0, label=f"Chance-level threshold: {random_threshold:.1f}%")
    ax.set_xlabel("% pixels assigned to majority cluster", fontsize=11)
    ax.set_ylabel("Number of ROIs", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_title(
        f"ROI assignment confidence (K={k})",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = output_dir / f"roi_assignment_summary_k{k}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROI assignment summary saved: {out_path}")


def plot_roi_metrics_combined(
        roi_clusters: pd.DataFrame,
        k_values: list[int],
        metrics_by_k: dict[int, dict],
        output_dir: Path,
) -> None:
    """Combined metrics table across all K values.

    PNG: metrics as rows, K values as columns.
    CSV: K values as rows, metrics as columns (transposed).

    Parameters
    ----------
    roi_clusters : pd.DataFrame
        Long-format ROI cluster assignments (all K).
    k_values : list[int]
        All K values evaluated.
    metrics_by_k : dict[int, dict]
        Loaded metrics_k{K}.json dicts keyed by K.
    output_dir : Path
    """
    metric_labels = [
        "Number of turf algae ROIs",
        "Median pct_majority (%)",
        "Mean pct_majority (%)",
        "IQR pct_majority (%)",
        "Chance-level threshold (1/K) (%)",
        "ROIs >= 50% (%)",
        "Adjusted Rand Index (ARI)",
        "Adjusted Mutual Information (AMI)",
        "Homogeneity",
        "Completeness",
        "V-measure",
    ]

    # Build data: rows = metrics, columns = K values
    table_cols = {k: [] for k in k_values}
    csv_rows = []

    for k in k_values:
        df_k = roi_clusters[roi_clusters["k"] == k]
        pct = df_k["pct_majority"].values
        m = metrics_by_k.get(k, {})
        chance = 100.0 / k
        iqr = float(np.percentile(pct, 75) - np.percentile(pct, 25))
        pct_above_50 = float((pct >= 50).mean() * 100)

        values = [
            str(len(df_k)),
            f"{np.median(pct):.1f}",
            f"{np.mean(pct):.1f}",
            f"{iqr:.1f}",
            f"{chance:.1f}",
            f"{pct_above_50:.1f}",
            str(m.get("ari", "N/A")),
            str(m.get("ami", "N/A")),
            str(m.get("homogeneity", "N/A")),
            str(m.get("completeness", "N/A")),
            str(m.get("v_measure", "N/A")),
        ]
        table_cols[k] = values
        csv_rows.append({
            "k": k,
            "n_rois": len(df_k),
            "median_pct_majority": round(float(np.median(pct)), 1),
            "mean_pct_majority": round(float(np.mean(pct)), 1),
            "iqr_pct_majority": round(iqr, 1),
            "chance_level_threshold": round(chance, 1),
            "pct_rois_above_50": round(pct_above_50, 1),
            "ari": m.get("ari", None),
            "ami": m.get("ami", None),
            "homogeneity": m.get("homogeneity", None),
            "completeness": m.get("completeness", None),
            "v_measure": m.get("v_measure", None),
        })

    # Save CSV (K as rows, metrics as columns)
    csv_path = output_dir / "roi_metrics_combined.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info(f"Combined ROI metrics CSV saved: {csv_path}")

    # Build PNG (metrics as rows, K as columns)
    col_labels = [f"K={k}" for k in k_values]
    cell_text = [
        [table_cols[k][i] for k in k_values]
        for i in range(len(metric_labels))
    ]

    n_rows = len(metric_labels)
    n_cols = len(k_values)
    fig_w = max(6, 1.5 * n_cols + 3)
    fig_h = max(3, 0.45 * n_rows + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=metric_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.5)
    ax.set_title("ROI assignment metrics across K values", fontsize=11, pad=12)
    plt.tight_layout()
    out_path = output_dir / "roi_metrics_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Combined ROI metrics table saved: {out_path}")


def plot_majority_vote_heatmap(
        pixel_df: pd.DataFrame,
        k: int,
        output_dir: Path,
) -> None:
    df_k = pixel_df[pixel_df["k"] == k]
    counts = (df_k.groupby(["roi_ID", "cluster_label"])
              .size().unstack(fill_value=0))
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    # Sort rows by majority cluster first, then by pct_majority descending
    # within each cluster — confident ROIs appear at top of each block
    majority_cluster = pct.idxmax(axis=1)
    pct_majority = pct.max(axis=1)
    sort_index = (pd.DataFrame({"majority_cluster": majority_cluster,
                                "pct_majority": pct_majority})
                  .sort_values(["majority_cluster", "pct_majority"],
                               ascending=[True, False])
                  .index)
    pct = pct.loc[sort_index]

    fig, ax = plt.subplots(figsize=(max(8, k * 1.0), max(10, len(pct) * 0.18)))
    im = ax.imshow(pct.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% pixels", shrink=0.5)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"C{c}" for c in range(1, k + 1)], fontsize=8)
    ax.set_yticks(range(len(pct)))
    ax.set_yticklabels(pct.index, fontsize=5)
    ax.set_title(f"Majority vote heatmap (K={k})", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / f"majority_vote_heatmap_k{k}.png", dpi=150)
    plt.close(fig)
    logger.info(f"Majority vote heatmap saved: {output_dir / f'majority_vote_heatmap_k{k}.png'}")


# ---------------------------------------------------------------------------
# 7e — Feature separation dot plot (Cleveland-style)
# ---------------------------------------------------------------------------

def plot_spectral_separation(
        sep_df: pd.DataFrame,
        k: int,
        output_dir: Path,
) -> None:
    """Figure 1: F-statistic vs wavelength (nm) for spectral features only.

    Line plot sorted from 475 nm to 705 nm. One continuous line connecting
    all wavelength bands, showing where in the spectrum clusters are best
    separated.
    """

    spectral = sep_df[sep_df["family"] == "spectral"].copy()

    # Extract wavelength integer from column names like "475_nm"
    spectral["wavelength"] = spectral["feature"].str.extract(r"(\d+)_nm").astype(int)
    spectral = spectral.sort_values("wavelength")

    if spectral.empty:
        logger.warning(f"No spectral features found in sep_df, skipping spectral separation plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(spectral["wavelength"], spectral["f_statistic"],
            color=FAMILY_COLOURS["spectral"], linewidth=1.5, zorder=2)

    ax.set_xlabel("Wavelength (nm)", fontsize=11)
    ax.set_ylabel("F-statistic (ANOVA)", fontsize=11)
    ax.set_xlim(spectral["wavelength"].min(), spectral["wavelength"].max())
    ax.set_ylim(bottom=0)
    ax.set_title(
        f"Spectral feature separation by wavelength (K={k})\n"
        f"Higher F-statistic = greater between-cluster vs within-cluster variance",
        fontsize=11,
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    out_path = output_dir / f"feature_separation_spectral_k{k}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Spectral separation plot saved: {out_path}")


def plot_spatial_separation(
        sep_df: pd.DataFrame,
        k: int,
        output_dir: Path,
) -> None:
    """Figure 2: F-statistic vs window/plot size for GLCM and specdiv features.

    Two-panel figure. Left panel: GLCM — four lines, one per metric
    (contrast, energy, entropy, homogeneity). Right panel: specdiv — two
    lines (alpha_local, beta_local). X-axis sorted ascending by window/plot
    size. All line plots with markers.
    """

    # --- GLCM ---
    glcm = sep_df[sep_df["family"] == "glcm"].copy()
    glcm["metric"] = glcm["feature"].str.extract(
        r"^(contrast|energy|entropy|homogeneity)"
    )
    glcm["window_size"] = glcm["feature"].str.extract(r"_window_(\d+)").astype(int)
    glcm = glcm.sort_values("window_size")

    # --- Specdiv ---
    sdiv = sep_df[sep_df["family"] == "sdiv"].copy()
    sdiv["metric"] = sdiv["feature"].str.extract(r"sdiv_(alpha_local|beta_local)")
    sdiv["plot_size"] = sdiv["feature"].str.extract(r"_plot_(\d+)").astype(int)
    sdiv = sdiv.sort_values("plot_size")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Consistent markers per metric
    glcm_markers = {
        "contrast": ("o", "#4C72B0"),
        "energy": ("s", "#DD8452"),
        "entropy": ("^", "#55A868"),
        "homogeneity": ("D", "#C44E52"),
    }
    sdiv_markers = {
        "alpha_local": ("o", "#4C72B0"),
        "beta_local": ("s", "#DD8452"),
    }

    # Left panel - GLCM
    ax = axes[0]
    for metric, (marker, colour) in glcm_markers.items():
        subset = glcm[glcm["metric"] == metric].sort_values("window_size")
        if subset.empty:
            continue
        sizes = subset["window_size"].tolist()
        x_pos = range(len(sizes))
        ax.plot(x_pos, subset["f_statistic"],
                marker=marker, color=colour, linewidth=1.5,
                markersize=6, label=metric, zorder=2)
    glcm_sizes = sorted(glcm["window_size"].unique())
    ax.set_xticks(range(len(glcm_sizes)))
    ax.set_xticklabels([str(s) for s in glcm_sizes], fontsize=9)
    ax.set_xlabel("Window size (px)", fontsize=11)
    ax.set_ylabel("F-statistic (ANOVA)", fontsize=11)
    ax.set_title(f"GLCM feature separation by window size (K={k})", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, title="GLCM metric", title_fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # Right panel - Specdiv
    ax2 = axes[1]
    for metric, (marker, colour) in sdiv_markers.items():
        subset = sdiv[sdiv["metric"] == metric].sort_values("plot_size")
        if subset.empty:
            continue
        ax2.plot(range(len(subset)), subset["f_statistic"],
                 marker=marker, color=colour, linewidth=1.5,
                 markersize=6, label=metric, zorder=2)
    sdiv_sizes = sorted(sdiv["plot_size"].unique())
    ax2.set_xticks(range(len(sdiv_sizes)))
    ax2.set_xticklabels([str(s) for s in sdiv_sizes], fontsize=9)
    ax2.set_xlabel("Plot size (px)", fontsize=11)
    ax2.set_ylabel("F-statistic (ANOVA)", fontsize=11)
    ax2.set_title(f"Specdiv feature separation by plot size (K={k})", fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.legend(fontsize=9, title="Specdiv metric", title_fontsize=9)
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.suptitle(
        f"Spatial feature separation by scale (K={k})\n"
        f"Higher F-statistic = greater between-cluster vs within-cluster variance",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = output_dir / f"feature_separation_spatial_k{k}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Spatial separation plot saved: {out_path}")