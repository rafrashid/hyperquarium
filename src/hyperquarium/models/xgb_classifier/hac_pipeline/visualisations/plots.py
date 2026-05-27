"""
hac_pipeline/visualisations/plots.py
--------------------------------------
All visualisation steps (7a–7e) for the HAC pipeline.

Figures are saved as PNG. matplotlib.use("Agg") is set in hac.py — no display needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour conventions — mirror xgb_pipeline family colours where possible
# ---------------------------------------------------------------------------

FAMILY_COLOURS = {
    "spectral": "#4C72B0",
    "glcm": "#DD8452",
    "sdiv": "#55A868",
    "unknown": "#8C8C8C",
}

POOR_ROI_COLOUR = "#D62728"  # red — 14 consistently misclassified ROIs
DEFAULT_LEAF_COLOUR = "#333333"


# ---------------------------------------------------------------------------
# 7a — Dendrogram
# ---------------------------------------------------------------------------


def plot_dendrogram(
        Z: np.ndarray,
        roi_ids: list[str],
        poor_rois: set[str],
        k_values: list[int],
        output_dir: Path,
) -> None:
    """Plot Ward linkage dendrogram truncated to one leaf per ROI.

    Parameters
    ----------
    Z : np.ndarray
        Ward linkage matrix.
    roi_ids : list[str]
        Ordered ROI IDs corresponding to leaves (one per ROI after truncation).
    poor_rois : set[str]
        ROI IDs to highlight in red (consistently misclassified).
    k_values : list[int]
        K values at which to draw horizontal cut lines.
    output_dir : Path
        Where to save dendrogram.png.
    """
    n_rois = len(set(roi_ids))

    fig, ax = plt.subplots(figsize=(max(18, n_rois * 0.15), 10))

    ddata = dendrogram(
        Z,
        truncate_mode="lastp",
        p=n_rois,
        ax=ax,
        no_plot=True,  # compute positions without plotting
    )

    # Re-draw with leaf label colours
    ddata_plot = dendrogram(
        Z,
        truncate_mode="lastp",
        p=n_rois,
        ax=ax,
        labels=roi_ids,
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=0,  # suppress scipy's automatic colour threshold
        above_threshold_color=DEFAULT_LEAF_COLOUR,
    )

    # Colour leaf labels for poor ROIs
    xlabels = ax.get_xticklabels()
    for label in xlabels:
        if label.get_text() in poor_rois:
            label.set_color(POOR_ROI_COLOUR)
            label.set_fontweight("bold")

    # Draw horizontal cut lines for each K
    _draw_k_cutlines(ax, Z, k_values)

    ax.set_title("Ward Linkage Dendrogram — Turf Algae ROIs", fontsize=13)
    ax.set_xlabel("ROI ID", fontsize=10)
    ax.set_ylabel("Linkage distance", fontsize=10)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=POOR_ROI_COLOUR, linewidth=2,
               label=f"Consistently misclassified ROIs (n={len(poor_rois)})"),
    ]
    for k in k_values:
        legend_elements.append(
            Line2D([0], [0], color="grey", linestyle="--", linewidth=1,
                   label=f"K={k} cut")
        )
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = output_dir / "dendrogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Dendrogram saved: {out_path}")


def _draw_k_cutlines(ax, Z: np.ndarray, k_values: list[int]) -> None:
    """Draw dashed horizontal lines at the linkage heights corresponding to K clusters."""

    n = Z.shape[0] + 1  # number of original observations
    for k in sorted(k_values, reverse=True):
        if k >= n:
            continue
        # Height at which the dendrogram has exactly K clusters
        # = the (N-K)-th merge height (0-indexed from the bottom)
        merge_idx = n - k - 1
        if 0 <= merge_idx < len(Z):
            height = Z[merge_idx, 2]
            ax.axhline(
                y=height, color="grey", linestyle="--", linewidth=0.8, alpha=0.7
            )
            ax.text(
                ax.get_xlim()[1] * 0.98, height,
                f" K={k}", va="bottom", ha="right", fontsize=7, color="grey"
            )


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
        umap_n_jobs: int = 4,
) -> None:
    """UMAP embedding of turf pixels: left = roi_ID, right = HAC cluster (K).

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-reduced pixel matrix.
    roi_ids : pd.Series
        ROI ID per pixel (for left panel colouring).
    pixel_df : pd.DataFrame
        Long-format cluster assignments; filtered to k internally.
    k : int
        K value for right panel.
    output_dir : Path
    umap_random_state : int
    umap_n_jobs : int
        Cores for UMAP parallelism.
    """
    try:
        import umap
    except ImportError:
        logger.warning(
            "umap-learn not installed. Skipping UMAP plots. "
            "Install with: pip install umap-learn"
        )
        return

    logger.info(f"Fitting UMAP embedding (K={k}).")
    reducer = umap.UMAP(
        n_components=2,
        random_state=umap_random_state,
        n_jobs=umap_n_jobs,
    )
    embedding = reducer.fit_transform(X_pca)

    # Left: roi_ID (integer-encoded for colour)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    roi_encoded = le.fit_transform(roi_ids)

    # Right: cluster labels for this K
    df_k = pixel_df[pixel_df["k"] == k].sort_values("pixel_idx")
    cluster_labels = df_k["cluster_label"].values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sc0 = axes[0].scatter(
        embedding[:, 0], embedding[:, 1],
        c=roi_encoded, cmap="tab20", s=1, alpha=0.5, linewidths=0,
    )
    axes[0].set_title("Coloured by ROI ID", fontsize=11)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].axis("off")

    sc1 = axes[1].scatter(
        embedding[:, 0], embedding[:, 1],
        c=cluster_labels, cmap="tab10", s=1, alpha=0.5, linewidths=0,
        vmin=1, vmax=k,
    )
    plt.colorbar(sc1, ax=axes[1], label="Cluster", shrink=0.7)
    axes[1].set_title(f"Coloured by HAC cluster (K={k})", fontsize=11)
    axes[1].set_xlabel("UMAP 1")
    axes[1].axis("off")

    fig.suptitle(
        f"UMAP — Turf Algae Pixels (K={k})\n"
        f"Left: ROI identity  |  Right: HAC cluster assignment",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = output_dir / f"umap_k{k}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"UMAP saved: {out_path}")


# ---------------------------------------------------------------------------
# 7c — Cluster vs held-out accuracy strip plot
# ---------------------------------------------------------------------------


def plot_cluster_accuracy(
        roi_clusters: pd.DataFrame,
        held_out_path: Path,
        k: int,
        output_dir: Path,
) -> None:
    """Strip plot: cluster label (x) vs held-out prop_correct (y).

    Parameters
    ----------
    roi_clusters : pd.DataFrame
        ROI majority vote results (from evaluator).
    held_out_path : Path
        Path to held_out_accuracy_summary.csv from xgb_pipeline.
    k : int
    output_dir : Path
    """
    if not held_out_path.exists():
        logger.warning(
            f"held_out_accuracy_summary.csv not found at {held_out_path}. "
            f"Skipping cluster accuracy plot for K={k}."
        )
        return

    held_out = pd.read_csv(held_out_path)
    df_k = roi_clusters[roi_clusters["k"] == k].copy()

    merged = df_k.merge(held_out[["roi_ID", "prop_correct"]], on="roi_ID", how="left")
    missing = merged["prop_correct"].isna().sum()
    if missing > 0:
        logger.warning(
            f"{missing} ROIs in cluster assignments not found in held_out summary."
        )

    fig, ax = plt.subplots(figsize=(max(8, k * 1.2), 5))

    rng = np.random.default_rng(seed=42)
    for cluster_id, group in merged.groupby("cluster_label"):
        x = rng.uniform(cluster_id - 0.3, cluster_id + 0.3, size=len(group))
        ax.scatter(x, group["prop_correct"], alpha=0.6, s=30, color="#4C72B0")
        median = group["prop_correct"].median()
        ax.hlines(median, cluster_id - 0.4, cluster_id + 0.4,
                  colors="black", linewidths=1.5)

    ax.set_xticks(range(1, k + 1))
    ax.set_xticklabels([f"C{c}" for c in range(1, k + 1)])
    ax.set_xlabel("HAC Cluster", fontsize=11)
    ax.set_ylabel("Held-out prop_correct", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Held-out classification accuracy by HAC cluster (K={k})\n"
        f"Each dot = one ROI; horizontal bar = median",
        fontsize=11,
    )
    plt.tight_layout()
    out_path = output_dir / f"cluster_accuracy_k{k}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Cluster accuracy strip plot saved: {out_path}")


# ---------------------------------------------------------------------------
# 7d — Majority vote heatmap
# ---------------------------------------------------------------------------


def plot_majority_vote_heatmap(
        pixel_df: pd.DataFrame,
        k: int,
        output_dir: Path,
) -> None:
    """Heatmap: ROIs (rows) × clusters (cols), values = % pixels per cluster.

    Rows sorted by majority cluster — block-diagonal structure = clean mapping.
    """
    df_k = pixel_df[pixel_df["k"] == k].copy()

    # Compute percentage of pixels per ROI assigned to each cluster
    counts = (
        df_k.groupby(["roi_ID", "cluster_label"])
        .size()
        .unstack(fill_value=0)
    )
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    # Sort rows by majority cluster
    majority_cluster = pct.idxmax(axis=1)
    pct = pct.loc[majority_cluster.sort_values().index]

    fig, ax = plt.subplots(figsize=(max(8, k * 1.0), max(10, len(pct) * 0.18)))

    im = ax.imshow(pct.values, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="% pixels assigned to cluster", shrink=0.5)

    ax.set_xticks(range(k))
    ax.set_xticklabels([f"C{c}" for c in range(1, k + 1)], fontsize=8)
    ax.set_yticks(range(len(pct)))
    ax.set_yticklabels(pct.index, fontsize=5)
    ax.set_xlabel("HAC Cluster", fontsize=10)
    ax.set_ylabel("ROI ID", fontsize=10)
    ax.set_title(
        f"Majority vote heatmap (K={k})\n"
        f"Block-diagonal structure = ROI → cluster alignment",
        fontsize=11,
    )

    plt.tight_layout()
    out_path = output_dir / f"majority_vote_heatmap_k{k}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Majority vote heatmap saved: {out_path}")


# ---------------------------------------------------------------------------
# 7e — Feature separation dot plot (Cleveland-style)
# ---------------------------------------------------------------------------


def plot_feature_separation(
        sep_df: pd.DataFrame,
        k: int,
        n_top: int,
        output_dir: Path,
) -> None:
    """Cleveland-style dot plot: F-rank vs loading-rank per feature.

    - Y-axis: top-N features sorted by combined_rank (most important at top)
    - X-axis: rank value (inverted — lower rank = more important = plotted further right)
    - Filled circle: F-statistic rank
    - Open circle: PCA loading rank
    - Connector line: gap shows agreement/disagreement between the two measures
    - Colour: feature family (spectral / glcm / sdiv)

    Parameters
    ----------
    sep_df : pd.DataFrame
        Output of compute_feature_separation(), sorted by combined_rank.
    k : int
    n_top : int
        Number of top features to display.
    output_dir : Path
    """
    plot_df = sep_df.head(n_top).copy()
    n_features = len(plot_df)
    n_all = len(sep_df)

    fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.38)))

    y_positions = np.arange(n_features)[::-1]  # top feature at top of plot

    for i, (_, row) in enumerate(plot_df.iterrows()):
        y = y_positions[i]
        colour = FAMILY_COLOURS.get(row["family"], FAMILY_COLOURS["unknown"])
        f_rank = row["f_rank"]
        l_rank = row["loading_rank"]

        # Connector line
        ax.hlines(y, min(f_rank, l_rank), max(f_rank, l_rank),
                  colors=colour, linewidths=0.8, alpha=0.5)

        # F-statistic rank — filled circle
        ax.scatter(f_rank, y, color=colour, s=55, zorder=3,
                   marker="o", label=None)

        # Loading rank — open circle
        ax.scatter(l_rank, y, color=colour, s=55, zorder=3,
                   marker="o", facecolors="none", linewidths=1.5, label=None)

    # Y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["feature"].tolist(), fontsize=8)

    # X-axis: inverted so rank 1 (most important) is on the right
    ax.set_xlim(n_all + 1, 0)
    ax.set_xlabel("Rank (lower = more important)", fontsize=10)
    ax.axvline(x=n_all / 2, color="grey", linestyle="--",
               linewidth=0.8, alpha=0.6, label="Top 50% threshold")

    ax.set_title(
        f"Feature separation — top {n_top} features (K={k})\n"
        f"● F-statistic rank   ○ PCA loading rank   "
        f"Gap = disagreement between measures",
        fontsize=10,
    )

    # Family legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=col, markersize=8, label=fam.upper())
        for fam, col in FAMILY_COLOURS.items()
        if fam != "unknown"
    ]
    legend_handles.append(
        Line2D([0], [0], color="grey", linestyle="--",
               linewidth=0.8, label="Top 50% threshold")
    )
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    out_path = output_dir / f"feature_separation_k{k}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Feature separation dot plot saved: {out_path}")
