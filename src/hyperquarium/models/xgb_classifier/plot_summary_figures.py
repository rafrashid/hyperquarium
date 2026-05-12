"""
plot_summary_figures.py
====================
Generates summary visualisations from a compiled dataset parquet file.

Usage:
    python plot_summary_figures.py path/to/compiled_dataset.parquet path/to/label_mapping.csv

Outputs (saved to same folder as input parquet), one set per label level:
    Level_1_fig1_label_summary.png
    Level_1_fig2_specdiv_swarm.png
    Level_1_fig3_glcm_strip.png
    Level_2_fig1_label_summary.png
    ... etc
"""

import re
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUBSAMPLE_N = 50  # max points per (label × roi_ID) for swarm/strip plots
RANDOM_SEED = 42
FIGURE_DPI = 150
SUBPLOT_WIDTH = 18  # inches — wide enough for all labels
SUBPLOT_HEIGHT = 3.5  # inches per subplot row

SPECDIV_VARS = ["alpha_sdiv", "beta_lcsd"]
GLCM_FEATURES = ["contrast", "energy", "entropy", "homogeneity"]

COLORBLIND_PALETTE = sns.color_palette("colorblind")

sns.set_theme(style="ticks", palette="colorblind")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "font.size": 10,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    print(f"Loading {path} ...")
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape}")
    print(f"  Labels: {sorted(df['label'].unique())}")
    return df


def load_mapping(path: Path) -> pd.DataFrame:
    """Load label mapping CSV. Returns DataFrame with columns Level_0..Level_N."""
    mapping = pd.read_csv(path)
    # Strip BOM from column names if present
    mapping.columns = [c.lstrip("\ufeff").strip() for c in mapping.columns]
    print(f"Loaded mapping: {path.name}")
    print(f"  Levels: {[c for c in mapping.columns if c.startswith('Level_')]}")
    return mapping


def apply_label_level(df: pd.DataFrame, mapping: pd.DataFrame, level: str) -> pd.DataFrame:
    """
    Map df['label'] (which corresponds to Level_0) to the target level column.
    Rows whose Level_0 label is not in the mapping are dropped with a warning.
    """
    lut = mapping.set_index("Level_0")[level].to_dict()
    missing = set(df["label"].unique()) - set(lut.keys())
    if missing:
        print(f"  [WARN] {level}: labels not in mapping (dropped): {sorted(missing)}")
    df = df[df["label"].isin(lut)].copy()
    df["label"] = df["label"].map(lut)
    return df


def stratified_subsample(df: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Subsample up to n rows per (label, roi_ID) group, preserving all columns."""
    groups = []
    for _, grp in df.groupby(["label", "roi_ID"]):
        groups.append(grp.sample(min(len(grp), n), random_state=seed))
    return pd.concat(groups, ignore_index=True)


def parse_size(col: str, keyword: str) -> int:
    """Extract integer size from column name, e.g. 'contrast_window_7' → 7."""
    m = re.search(rf"{keyword}_(\d+)$", col)
    return int(m.group(1)) if m else -1


def valid_labels_for_col(df: pd.DataFrame, col: str) -> list[str]:
    """Return labels that have at least one non-NaN value in col, sorted by label."""
    valid = df.dropna(subset=[col]).groupby("label")[col].count()
    return valid[valid > 0].index.tolist()


def label_order(df: pd.DataFrame) -> list[str]:
    """Return labels sorted by descending row count."""
    return df["label"].value_counts().index.tolist()


# ---------------------------------------------------------------------------
# Figure 1 — ROI counts + row counts per label
# ---------------------------------------------------------------------------

def figure1(df: pd.DataFrame, out_path: Path):
    print("Generating Figure 1 ...")

    roi_counts = (df.groupby("label")["roi_ID"]
                  .nunique()
                  .sort_values(ascending=False))
    row_counts = df["label"].value_counts().reindex(roi_counts.index)
    labels = roi_counts.index.tolist()

    top1_val = roi_counts.iloc[0]
    top2_val = roi_counts.iloc[1]
    diff = top1_val - top2_val
    top_label = labels[0]

    fig, ax1 = plt.subplots(figsize=(max(10, len(labels) * 0.9), 6))
    ax2 = ax1.twinx()

    x = np.arange(len(labels))
    width = 0.55
    bars = ax1.bar(x, roi_counts.values, width=width,
                   color=COLORBLIND_PALETTE[0], alpha=0.85, zorder=3,
                   label="Unique ROIs")

    # Difference section — darker shade of the same bar colour
    darker_blue = tuple(c * 0.6 for c in COLORBLIND_PALETTE[0])
    ax1.bar(x[0], diff, width=width, bottom=top2_val,
            color=darker_blue, zorder=4, label=f"Δ vs 2nd ({diff} ROIs)")

    bars.set_label("_nolegend_")

    # Value labels above each bar
    for bar, val in zip(bars, roi_counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + top1_val * 0.01,
                 str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Annotation for the difference
    ax1.annotate(
        f"Δ = {diff}",
        xy=(x[0], top2_val + diff / 2),
        xytext=(x[0] + 0.6, top2_val + diff / 2),
        arrowprops=dict(arrowstyle="->", color="dimgrey"),
        fontsize=9, color="dimgrey", va="center"
    )

    # Line — row counts
    ax2.plot(x, row_counts.values, color=COLORBLIND_PALETTE[1],
             marker="o", linewidth=2, markersize=5, label="Total rows", zorder=5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Unique ROI count", color=COLORBLIND_PALETTE[0])
    ax2.set_ylabel("Total pixel rows", color=COLORBLIND_PALETTE[1])
    ax1.set_xlabel("Label")
    ax1.set_ylim(0, top1_val * 1.15)
    ax2.set_ylim(0, row_counts.max() * 1.15)
    ax1.tick_params(axis="y", labelcolor=COLORBLIND_PALETTE[0])
    ax2.tick_params(axis="y", labelcolor=COLORBLIND_PALETTE[1])
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    ax1.set_title("ROI and pixel row counts per label", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Specdiv swarm plots, one subplot per plot size
# ---------------------------------------------------------------------------

def figure2(df: pd.DataFrame, out_path: Path):
    print("Generating Figure 2 ...")

    sdiv_cols = sorted([c for c in df.columns if c.startswith("sdiv_")],
                       key=lambda c: parse_size(c, "plot"))
    plot_sizes = sorted(set(parse_size(c, "plot") for c in sdiv_cols))
    n_rows = len(plot_sizes)

    # Subsample for display
    df_sub = stratified_subsample(df, SUBSAMPLE_N)

    var_palette = {v: COLORBLIND_PALETTE[i] for i, v in enumerate(SPECDIV_VARS)}
    var_offsets = {SPECDIV_VARS[0]: -0.18, SPECDIV_VARS[1]: 0.18}

    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(SUBPLOT_WIDTH, SUBPLOT_HEIGHT * n_rows),
                             sharex=False)
    if n_rows == 1:
        axes = [axes]

    for ax, psize in zip(axes, plot_sizes):
        # Only labels with valid data for at least one var at this size
        valid_lbls = set()
        for var in SPECDIV_VARS:
            col = f"sdiv_{var}_plot_{psize}"
            if col in df.columns:
                valid_lbls |= set(valid_labels_for_col(df, col))
        valid_lbls = [l for l in label_order(df) if l in valid_lbls]

        if not valid_lbls:
            ax.set_visible(False)
            continue

        lbl_pos = {lbl: i for i, lbl in enumerate(valid_lbls)}

        for var in SPECDIV_VARS:
            col = f"sdiv_{var}_plot_{psize}"
            if col not in df.columns:
                continue
            sub = df_sub[df_sub["label"].isin(valid_lbls)].dropna(subset=[col])
            offset = var_offsets[var]
            color = var_palette[var]

            for lbl in valid_lbls:
                grp = sub[sub["label"] == lbl][col].values
                if len(grp) == 0:
                    continue
                xpos = lbl_pos[lbl] + offset
                # Jitter within the offset band
                jitter = np.random.default_rng(RANDOM_SEED).uniform(
                    -0.08, 0.08, size=len(grp))
                ax.scatter(xpos + jitter, grp, color=color,
                           alpha=0.5, s=12, zorder=3, linewidths=0)
                # Median line — solid black
                med = np.nanmedian(grp)
                ax.hlines(med, xpos - 0.12, xpos + 0.12,
                          colors="black", linewidths=2, zorder=4)

        ax.set_xticks(range(len(valid_lbls)))
        ax.set_xticklabels(valid_lbls, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.set_title(f"Plot size = {psize}", fontsize=9, loc="left", pad=4)
        ax.set_xlim(-0.6, len(valid_lbls) - 0.4)

    # Legend
    legend_handles = [
        mpatches.Patch(color=var_palette[v], label=v) for v in SPECDIV_VARS
    ]
    fig.legend(handles=legend_handles, title="Specdiv variable",
               loc="upper right", fontsize=9, title_fontsize=9)
    fig.suptitle("Spectral diversity by plot size and label", fontsize=14, y=1.002)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — GLCM strip plots, one subplot per feature
# ---------------------------------------------------------------------------

def figure3(df: pd.DataFrame, out_path: Path):
    print("Generating Figure 3 ...")

    glcm_cols = [c for c in df.columns
                 if any(c.startswith(f + "_window_") for f in GLCM_FEATURES)]
    window_sizes = sorted(set(parse_size(c, "window") for c in glcm_cols))
    n_features = len(GLCM_FEATURES)

    df_sub = stratified_subsample(df, SUBSAMPLE_N)

    # One colour per window size
    size_palette = {s: COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
                    for i, s in enumerate(window_sizes)}

    fig, axes = plt.subplots(n_features, 1,
                             figsize=(SUBPLOT_WIDTH, SUBPLOT_HEIGHT * n_features),
                             sharex=False)
    if n_features == 1:
        axes = [axes]

    rng = np.random.default_rng(RANDOM_SEED)

    for ax, feature in zip(axes, GLCM_FEATURES):
        # Collect all labels with at least one valid window for this feature
        valid_lbls = set()
        for ws in window_sizes:
            col = f"{feature}_window_{ws}"
            if col in df.columns:
                valid_lbls |= set(valid_labels_for_col(df, col))
        valid_lbls = [l for l in label_order(df) if l in valid_lbls]

        n_lbls = len(valid_lbls)
        n_ws = len(window_sizes)
        grp_w = 0.8
        ws_w = grp_w / n_ws

        lbl_pos = {lbl: i for i, lbl in enumerate(valid_lbls)}

        for ws_i, ws in enumerate(window_sizes):
            col = f"{feature}_window_{ws}"
            if col not in df.columns:
                continue
            color = size_palette[ws]
            sub = df_sub[df_sub["label"].isin(valid_lbls)].dropna(subset=[col])

            for lbl in valid_lbls:
                grp = sub[sub["label"] == lbl][col].values
                if len(grp) == 0:
                    continue
                # Centre of this window size within the label group
                x_centre = lbl_pos[lbl] - grp_w / 2 + ws_w * (ws_i + 0.5)
                jitter = rng.uniform(-ws_w * 0.35, ws_w * 0.35, size=len(grp))
                ax.scatter(x_centre + jitter, grp, color=color,
                           alpha=0.5, s=8, zorder=3, linewidths=0)
                # Median line — solid black
                med = np.nanmedian(grp)
                ax.hlines(med, x_centre - ws_w * 0.4, x_centre + ws_w * 0.4,
                          colors="black", linewidths=1.5, zorder=4)

        ax.set_xticks(range(n_lbls))
        ax.set_xticklabels(valid_lbls, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.set_title(feature.capitalize(), fontsize=10, loc="left", pad=4)
        ax.set_xlim(-0.6, n_lbls - 0.4)

    # Shared legend for window sizes
    legend_handles = [
        mpatches.Patch(color=size_palette[ws], label=f"window {ws}")
        for ws in window_sizes
    ]
    fig.legend(handles=legend_handles, title="Window size",
               loc="upper right", fontsize=8, title_fontsize=9,
               ncol=2)
    fig.suptitle("GLCM texture features by label", fontsize=14, y=1.002)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_summary_figures.py path/to/compiled_dataset.parquet [path/to/label_mapping.csv]")
        sys.exit(1)

    input_path = Path(sys.argv[1]).resolve()
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    out_dir = input_path.parent / f"{input_path.stem}_figs"
    out_dir.mkdir(exist_ok=True)
    df_raw = load_data(input_path)

    # Load mapping if provided
    if len(sys.argv) >= 3:
        mapping_path = Path(sys.argv[2]).resolve()
        mapping = load_mapping(mapping_path)
        level_cols = [c for c in mapping.columns if c.startswith("Level_") and c != "Level_0"]
    else:
        mapping = None
        level_cols = []

    if mapping is not None:
        for level in level_cols:
            print(f"\n{'=' * 50}")
            print(f"Generating figures for {level} ...")
            df_level = apply_label_level(df_raw, mapping, level)
            prefix = f"{level}_"
            figure1(df_level, out_dir / f"{prefix}fig1_label_summary.png")
            figure2(df_level, out_dir / f"{prefix}fig2_specdiv_swarm.png")
            figure3(df_level, out_dir / f"{prefix}fig3_glcm_strip.png")
    else:
        # No mapping — produce figures with raw labels
        figure1(df_raw, out_dir / "fig1_label_summary.png")
        figure2(df_raw, out_dir / "fig2_specdiv_swarm.png")
        figure3(df_raw, out_dir / "fig3_glcm_strip.png")

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
