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
POOR_ROI_COLOUR = "#D62728"
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

    # Annotate leaves: scipy uses integer leaf labels with lastp truncation.
    # Re-label x-ticks with roi_ids in dendrogram order.
    ax.set_xticklabels(roi_ids, rotation=90, fontsize=5)

    # Colour poor ROI labels
    for lbl in ax.get_xticklabels():
        if lbl.get_text() in poor_rois:
            lbl.set_color(POOR_ROI_COLOUR)
            lbl.set_fontweight("bold")

    _draw_k_cutlines(ax, Z, k_values, n_rois)

    ax.set_title("Ward Linkage Dendrogram — Turf Algae ROIs", fontsize=13)
    ax.set_xlabel("ROI ID", fontsize=10)
    ax.set_ylabel("Linkage distance", fontsize=10)

    from matplotlib.lines import Line2D
    legend_elements = [
                          Line2D([0], [0], color=POOR_ROI_COLOUR, linewidth=2,
                                 label=f"Consistently misclassified ROIs (n={len(poor_rois)})"),
                      ] + [
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
        umap_n_jobs: int = 4,
) -> None:
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP plots.")
        return

    logger.info(f"Fitting UMAP (K={k}).")
    reducer = umap.UMAP(n_components=2, random_state=umap_random_state,
                        n_jobs=umap_n_jobs)
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
# 7d — Majority vote heatmap
# ---------------------------------------------------------------------------

def plot_majority_vote_heatmap(
        pixel_df: pd.DataFrame,
        k: int,
        output_dir: Path,
) -> None:
    df_k = pixel_df[pixel_df["k"] == k]
    counts = (df_k.groupby(["roi_ID", "cluster_label"])
              .size().unstack(fill_value=0))
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pct = pct.loc[pct.idxmax(axis=1).sort_values().index]

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

def plot_feature_separation(
        sep_df: pd.DataFrame,
        k: int,
        n_top: int,
        output_dir: Path,
) -> None:
    plot_df = sep_df.head(n_top).copy()
    n_all = len(sep_df)
    y_positions = np.arange(len(plot_df))[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.38)))

    for i, (_, row) in enumerate(plot_df.iterrows()):
        y = y_positions[i]
        colour = FAMILY_COLOURS.get(row["family"], FAMILY_COLOURS["unknown"])
        f_rank = row["f_rank"]
        l_rank = row["loading_rank"]
        ax.hlines(y, min(f_rank, l_rank), max(f_rank, l_rank),
                  colors=colour, linewidths=0.8, alpha=0.5)
        ax.scatter(f_rank, y, color=colour, s=55, zorder=3)
        ax.scatter(l_rank, y, color=colour, s=55, zorder=3,
                   facecolors="none", linewidths=1.5)

        # If SHAP cross-reference present, add triangle marker
        if "shap_rank" in row and not np.isnan(row["shap_rank"]):
            ax.scatter(row["shap_rank"], y, color=colour, s=40, zorder=3,
                       marker="^", alpha=0.7)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["feature"].tolist(), fontsize=8)
    ax.set_xlim(n_all + 1, 0)
    ax.set_xlabel("Rank (lower = more important)", fontsize=10)
    ax.axvline(x=n_all / 2, color="grey", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.set_title(
        f"Feature separation — top {n_top} features (K={k})\n"
        f"● F-statistic rank   ○ PCA loading rank"
        + ("   ▲ SHAP rank (post-hoc)" if "shap_rank" in sep_df.columns else ""),
        fontsize=10,
    )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=col, markersize=8, label=fam.upper())
        for fam, col in FAMILY_COLOURS.items() if fam != "unknown"
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_dir / f"feature_separation_k{k}.png", dpi=150)
    plt.close(fig)
    logger.info(f"Feature separation dot plot saved: "
                f"{output_dir / f'feature_separation_k{k}.png'}")
