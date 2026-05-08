"""
visualisations.py
Standalone visualisation module for cross-model feature rank comparison.
Not integrated into the main pipeline — run independently after SHAP outputs exist.

Functions:
    bump_chart()  — rank trajectories across 4 spectra types (top N features)
    rank_heatmap() — full rank matrix across all models (spectra × level)

Usage:
    python3 visualisations.py --type bump --level 3 --top-n 20
    python3 visualisations.py --type heatmap --level 3
    python3 visualisations.py --type both --level 3 --top-n 20

Expects feature_importance_shap.csv to exist in each model output directory.
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_importance(output_dir: Path, spectra: str, level: int, weighted: bool = True) -> pd.DataFrame | None:
    """Loads feature_importance_shap.csv for one model. Returns None if missing."""
    suffix = "" if weighted else "_unweighted"
    path = output_dir / f"spectra_{spectra}" / f"level_{level}{suffix}" / "feature_importance_shap.csv"
    if not path.exists():
        logger.warning(f"Missing: {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    logger.info(f"Loaded: {path}  ({len(df)} features)")
    return df


def get_family(col: str) -> str:
    """Classifies a feature column into spectral / glcm / sdiv."""
    import re
    if re.match(r"^\d+_nm$", col):
        return "spectral"
    if re.match(r"^(energy|entropy|homogeneity|contrast)_window_\d+$", col):
        return "glcm"
    if re.match(r"^sdiv_.+_plot_\d+$", col):
        return "sdiv"
    return "other"


FAMILY_COLOURS = {
    "spectral": "#185FA5",  # blue
    "glcm": "#0F6E56",  # teal
    "sdiv": "#854F0B",  # amber
    "other": "#888780",  # gray
}

FAMILY_LABELS = {
    "spectral": "Spectral",
    "glcm": "GLCM",
    "sdiv": "Spectral diversity",
    "other": "Other",
}


def build_rank_matrix(
        output_dir: Path,
        spectra_types: list[str],
        level: int,
        weighted: bool,
        shap_col: str = "mean_abs_shap_global",
) -> pd.DataFrame:
    """
    Loads importance CSVs for each spectra type and returns a DataFrame of ranks.
    Rows = features, columns = spectra types.
    Rank 1 = most important.
    """
    frames = {}
    for s in spectra_types:
        imp = load_importance(output_dir, s, level, weighted)
        if imp is not None and shap_col in imp.columns:
            frames[s] = imp[shap_col].rank(ascending=False).astype(int)

    if not frames:
        raise FileNotFoundError("No importance files found. Run shap.py first.")

    rank_df = pd.DataFrame(frames)
    rank_df.index.name = "feature"
    return rank_df


def build_cross_model_rank_matrix(
        output_dir: Path,
        spectra_types: list[str],
        levels: list[int],
        weighted: bool = True,
        shap_col: str = "mean_abs_shap_global",
) -> pd.DataFrame:
    """
    Builds a rank matrix across all spectra × level combinations.
    Columns labelled as e.g. 'A_L3', 'B_L3', 'A_L2', etc.
    """
    frames = {}
    for level in levels:
        for s in spectra_types:
            imp = load_importance(output_dir, s, level, weighted)
            if imp is not None and shap_col in imp.columns:
                col_label = f"{s}_L{level}"
                frames[col_label] = imp[shap_col].rank(ascending=False).astype(int)

    if not frames:
        raise FileNotFoundError("No importance files found.")

    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Bump chart
# ---------------------------------------------------------------------------

def bump_chart(
        output_dir: Path,
        level: int,
        spectra_types: list[str] = None,
        top_n: int = 20,
        weighted: bool = True,
        shap_col: str = "mean_abs_shap_global",
        out_path: Path | None = None,
        **kwargs,
) -> None:
    """
    Bump chart of feature rank trajectories across spectra types A→B→C→D.
    Shows top_n features by mean rank across all spectra.
    Lines coloured by feature family (spectral / glcm / sdiv).
    Rank axis is inverted so rank 1 is at the top.

    Args:
        output_dir:    Root output directory (e.g. Path('outputs')).
        level:         Hierarchy level to plot.
        spectra_types: List of spectra labels. Defaults to ['A','B','C','D'].
        top_n:         Number of top features to include (by mean rank).
        weighted:      Whether to use weighted model outputs.
        shap_col:      SHAP column to rank by.
        out_path:      Output PNG path. Auto-generated if None.
    """
    spectra_types = spectra_types or ["A", "B", "C", "D"]
    rank_df = build_rank_matrix(output_dir, spectra_types, level, weighted, shap_col)

    # Select top N features by mean rank across spectra
    rank_df["mean_rank"] = rank_df.mean(axis=1)
    top_features = rank_df.nsmallest(top_n, "mean_rank").index.tolist()
    rank_df = rank_df.loc[top_features, spectra_types]

    fig, ax = plt.subplots(figsize=(max(8, len(spectra_types) * 2.5), top_n * 0.45 + 2))

    x_pos = np.arange(len(spectra_types))

    for feat in top_features:
        family = get_family(feat)
        colour = FAMILY_COLOURS[family]
        ranks = rank_df.loc[feat, spectra_types].values

        ax.plot(x_pos, ranks, color=colour, linewidth=1.5, alpha=0.75,
                marker="o", markersize=5, zorder=2)

        # Label at start and end
        ax.text(-0.15, ranks[0], feat, ha="right", va="center",
                fontsize=8, color=colour)
        ax.text(len(spectra_types) - 0.85, ranks[-1], feat, ha="left", va="center",
                fontsize=8, color=colour)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Spectra {s}" for s in spectra_types], fontsize=10)
    ax.set(
        ylabel="Rank (1 = most important)",
        title=f"Feature rank trajectories across spectra — Level {level} (top {top_n})",
        **kwargs,
    )
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlim(-0.5, len(spectra_types) - 0.5)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    legend_handles = [
        mpatches.Patch(color=FAMILY_COLOURS[f], label=FAMILY_LABELS[f])
        for f in ["spectral", "glcm", "sdiv"]
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              framealpha=0.8, title="Feature family", title_fontsize=9)

    fig.tight_layout()

    if out_path is None:
        out_path = output_dir / f"bump_chart_level{level}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Bump chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Rank heatmap
# ---------------------------------------------------------------------------

def rank_heatmap(
        output_dir: Path,
        levels: list[int] = None,
        spectra_types: list[str] = None,
        weighted: bool = True,
        top_n: int | None = None,
        shap_col: str = "mean_abs_shap_global",
        feature_order: list[str] | None = None,
        out_path: Path | None = None,
        **kwargs,
) -> None:
    """
    Heatmap of feature ranks across all spectra × level combinations.
    Rows = features (sorted by mean rank), columns = model (spectra_level).
    Cell colour encodes rank — darker = higher rank (more important).
    Row colours on the left encode feature family.

    Args:
        output_dir:    Root output directory.
        levels:        List of levels to include. Defaults to [3, 2, 1].
        spectra_types: List of spectra labels. Defaults to ['A','B','C','D'].
        weighted:      Whether to use weighted model outputs.
        top_n:         Limit to top N features by mean rank. None = all features.
        shap_col:      SHAP column to rank by.
        out_path:      Output PNG path. Auto-generated if None.
    """
    levels = levels or [3, 2, 1]
    spectra_types = spectra_types or ["A", "B", "C", "D"]

    rank_df = build_cross_model_rank_matrix(output_dir, spectra_types, levels, weighted, shap_col)

    if feature_order is not None:
        # Reindex to original data column order, keeping only features present in rank_df
        ordered = [f for f in feature_order if f in rank_df.index]
        rank_df = rank_df.loc[ordered]
        if top_n is not None:
            rank_df = rank_df.head(top_n)
    else:
        # Default: sort by mean rank (most important at top)
        rank_df["_mean"] = rank_df.mean(axis=1)
        rank_df = rank_df.sort_values("_mean")
        rank_df = rank_df.drop(columns="_mean")
        if top_n is not None:
            rank_df = rank_df.head(top_n)

    n_feat, n_models = rank_df.shape
    families = [get_family(f) for f in rank_df.index]
    fam_colours = [FAMILY_COLOURS[f] for f in families]

    fig_h = max(4, n_feat * 0.28)
    fig_w = max(8, n_models * 0.75 + 2)

    # Two columns: narrow strip for family colours, wide heatmap
    fig, (strip_ax, ax) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [0.02, 1], "wspace": 0.01},
    )

    # Family colour strip + legend below it
    for i, fc in enumerate(fam_colours):
        strip_ax.barh(i, 1, color=fc, height=0.85)
    strip_ax.set_xlim(0, 1)
    strip_ax.set_ylim(-0.5, n_feat - 0.5)
    strip_ax.axis("off")

    # Family legend placed in the strip column, below the strips
    legend_handles = [
        mpatches.Patch(color=FAMILY_COLOURS[f], label=FAMILY_LABELS[f])
        for f in ["spectral", "glcm", "sdiv"]
    ]
    strip_ax.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, -0.02), fontsize=8,
        framealpha=0.85, title="Family", title_fontsize=8,
        ncol=1,
    )

    # Rank heatmap
    data = rank_df.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap="inferno_r",
                   vmin=1, vmax=data.max())

    # Annotate cells — only if not too many features
    if n_feat <= 40:
        for i in range(n_feat):
            for j in range(n_models):
                val = int(data[i, j])
                brightness = data[i, j] / data.max()
                txt_colour = "white" if brightness < 0.45 else "black"
                ax.text(j, i, str(val), ha="center", va="center",
                        fontsize=7, color=txt_colour)

    # Axes
    ax.set_xticks(np.arange(n_models))
    ax.set_xticklabels(rank_df.columns, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_feat))
    ax.set_yticklabels(rank_df.index, fontsize=8)
    ax.yaxis.set_tick_params(pad=8)  # Padding between strip and y-tick labels
    ax.set(**kwargs)
    ax.set_title(
        f"Feature rank heatmap — all models (rank 1 = most important)\n"
        f"Levels {levels}, Spectra {spectra_types}",
        pad=14,  # Padding between title and plot
    )

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Rank", fontsize=9)

    # (Family legend is placed in strip_ax above)

    # Column group separators between levels
    n_spectra = len(spectra_types)
    for lvl_idx in range(1, len(levels)):
        x_sep = lvl_idx * n_spectra - 0.5
        ax.axvline(x_sep, color="white", linewidth=1.2, alpha=0.8)

    # Column group labels (level headers) using axis fraction
    for lvl_idx, lvl in enumerate(levels):
        x_frac = (lvl_idx * n_spectra + (n_spectra - 1) / 2) / n_models
        ax.text(x_frac, 1.01, f"Level {lvl}", ha="center", va="bottom",
                fontsize=9, fontweight=500, transform=ax.transAxes)

    fig.tight_layout()

    if out_path is None:
        lvl_str = "_".join(str(l) for l in levels)
        out_path = output_dir / f"rank_heatmap_levels{lvl_str}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Rank heatmap saved: {out_path}")


# ---------------------------------------------------------------------------
# Directional bar chart biplot
# ---------------------------------------------------------------------------

def directional_bar_biplot(
        output_dir: Path,
        model_a: tuple[str, int],
        model_b: tuple[str, int],
        weighted: bool = True,
        top_n: int = 30,
        shap_col: str = "mean_abs_shap_global",
        out_path: Path | None = None,
) -> None:
    """
    Directional bar chart biplot comparing feature ranks between two models.
    Each feature is a horizontal bar. Bars extend left for Model A and right
    for Model B — length encodes rank (longer = higher rank = more important).
    Features are sorted by rank difference, so the most divergent features
    appear at the top and bottom.

    Useful for direct pairwise comparison e.g.:
        Level 2 vs Level 4 (same spectra)
        Spectra A vs Spectra B (same level)
        Weighted vs unweighted (same spectra + level)

    Args:
        output_dir: Root output directory.
        model_a:    (spectra, level) tuple for the left model e.g. ('A', 2)
        model_b:    (spectra, level) tuple for the right model e.g. ('A', 4)
        weighted:   Use weighted model outputs.
        top_n:      Total features to show (most divergent by rank difference).
        shap_col:   SHAP column to rank by.
        out_path:   Output PNG path. Auto-generated if None.
    """
    spec_a, lvl_a = model_a
    spec_b, lvl_b = model_b

    imp_a = load_importance(output_dir, spec_a, lvl_a, weighted)
    imp_b = load_importance(output_dir, spec_b, lvl_b, weighted)

    if imp_a is None or imp_b is None:
        logger.error("Could not load both importance files — aborting biplot.")
        return

    if shap_col not in imp_a.columns or shap_col not in imp_b.columns:
        logger.error(f"Column '{shap_col}' not found in one or both importance files.")
        return

    # Align on common features and compute ranks
    common = imp_a.index.intersection(imp_b.index)
    rank_a = imp_a.loc[common, shap_col].rank(ascending=False).astype(int)
    rank_b = imp_b.loc[common, shap_col].rank(ascending=False).astype(int)

    compare = pd.DataFrame({"rank_a": rank_a, "rank_b": rank_b})
    compare["diff"] = compare["rank_b"] - compare["rank_a"]  # Positive = rose in B
    compare["family"] = [get_family(f) for f in compare.index]

    # Select top_n most divergent features by absolute rank difference
    compare = compare.reindex(compare["diff"].abs().nlargest(top_n).index)
    compare = compare.sort_values("diff")  # Most negative (fell in B) at top

    n = len(compare)
    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.38 + 2)))
    y_pos = np.arange(n)
    max_rank = max(compare["rank_a"].max(), compare["rank_b"].max())

    # Bar length = max_rank - rank + 1 so rank 1 -> longest bar
    len_a = (max_rank - compare["rank_a"] + 1).values
    len_b = (max_rank - compare["rank_b"] + 1).values

    label_a = f"Spectra {spec_a} — Level {lvl_a}"
    label_b = f"Spectra {spec_b} — Level {lvl_b}"

    ax.barh(y_pos, -len_a, color="#B5D4F4", edgecolor="#185FA5",
            linewidth=0.5, label=label_a, zorder=2)
    ax.barh(y_pos, len_b, color="#9FE1CB", edgecolor="#0F6E56",
            linewidth=0.5, label=label_b, zorder=2)

    # Family colour dots at centre spine
    for i, (_, row) in enumerate(compare.iterrows()):
        ax.plot(0, i, "o", color=FAMILY_COLOURS[row["family"]],
                markersize=6, zorder=4)

    # Rank annotations inside bars
    for i, (la, lb) in enumerate(zip(len_a, len_b)):
        ax.text(-la - 1, i, f"#{compare['rank_a'].iloc[i]}",
                ha="right", va="center", fontsize=7.5, color="#185FA5")
        ax.text(lb + 1, i, f"#{compare['rank_b'].iloc[i]}",
                ha="left", va="center", fontsize=7.5, color="#0F6E56")

    # Feature labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(compare.index, fontsize=8.5)
    ax.yaxis.set_tick_params(length=0)

    # Rank-shift annotation on right margin
    for i, diff in enumerate(compare["diff"].values):
        colour = "#0F6E56" if diff > 0 else "#993C1D" if diff < 0 else "#888780"
        arrow = "\u25b2" if diff > 0 else "\u25bc" if diff < 0 else "\u2014"
        ax.text(max_rank * 1.08, i, f"{arrow} {abs(diff)}",
                ha="left", va="center", fontsize=8, color=colour)

    ax.axvline(0, color="black", linewidth=0.8, zorder=3)
    ax.set_xticks([])
    ax.set_xlabel("Relative rank importance (bar length proportional to rank)", fontsize=9)
    ax.set_title(
        f"Directional rank biplot\n{label_a}  vs  {label_b}\n"
        f"Top {top_n} most divergent features — sorted by rank shift",
        fontsize=10,
    )

    # Direction labels below x-axis
    ax.text(-max_rank * 0.5, -1.3, f"\u2190 {label_a}", ha="center",
            fontsize=9, color="#185FA5", transform=ax.get_xaxis_transform())
    ax.text(max_rank * 0.5, -1.3, f"{label_b} \u2192", ha="center",
            fontsize=9, color="#0F6E56", transform=ax.get_xaxis_transform())

    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.15, linewidth=0.5)

    bar_handles = [
        mpatches.Patch(facecolor="#B5D4F4", edgecolor="#185FA5", label=label_a),
        mpatches.Patch(facecolor="#9FE1CB", edgecolor="#0F6E56", label=label_b),
    ]
    fam_handles = [
        mpatches.Patch(color=FAMILY_COLOURS[f], label=FAMILY_LABELS[f])
        for f in ["spectral", "glcm", "sdiv"]
    ]
    leg1 = ax.legend(handles=bar_handles, loc="lower right", fontsize=8.5,
                     framealpha=0.85, title="Model", title_fontsize=8.5)
    ax.legend(handles=fam_handles, loc="upper right", fontsize=8.5,
              framealpha=0.85, title="Feature family", title_fontsize=8.5)
    ax.add_artist(leg1)

    fig.tight_layout()

    if out_path is None:
        out_path = output_dir / f"biplot_{spec_a}L{lvl_a}_vs_{spec_b}L{lvl_b}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Directional biplot saved: {out_path}")


# ---------------------------------------------------------------------------
# Spectral wavelength importance heatmap
# ---------------------------------------------------------------------------

def wavelength_heatmap(
        output_dir: Path,
        level: int | list[int] = 3,
        spectra_types: list[str] = None,
        weighted: bool = True,
        shap_col: str = "mean_abs_shap_global",
        cmap: str = "inferno_r",
        out_path: Path | None = None,
        **kwargs,
) -> None:
    """
    Heatmap of feature rank for spectral features only.
    X-axis: wavelengths in ascending order (e.g. 425_nm ... 705_nm).
    Y-axis: models (one row per spectra type by default).

    Useful for comparing which wavelengths are most important across
    spectral transformations (A/B/C/D) at a given level, or across
    levels for a given spectra type.

    Args:
        output_dir:    Root output directory.
        level:         Hierarchy level (default: 3).
        spectra_types: List of spectra labels (default: ['A','B','C','D']).
                       Each becomes one row on the y-axis.
        weighted:      Use weighted model outputs.
        shap_col:      SHAP column to rank by (e.g. mean_abs_shap_global or
                       mean_abs_shap_turf_algae). Ranking is done across all
                       features; only spectral results are then displayed.
        cmap:          Matplotlib colourmap (default: cividis_r).
        out_path:      Output PNG path. Auto-generated if None.
    """
    import re
    spectra_types = spectra_types or ["A", "B", "C", "D"]
    levels = [level] if isinstance(level, int) else level

    # Load importance for each model row — one row per (spectra, level) combo
    rows = {}
    for lvl in levels:
        for s in spectra_types:
            imp = load_importance(output_dir, s, lvl, weighted)
            if imp is not None and shap_col in imp.columns:
                ranked = imp[shap_col].rank(ascending=False).astype(int)
                spectral = {
                    col: ranked[col]
                    for col in imp.index
                    if re.match(r"^\d+_nm$", col) and col in ranked
                }
                row_label = f"Spectra {s} L{lvl}" if len(levels) > 1 else f"Spectra {s}"
                rows[row_label] = spectral

    if not rows:
        logger.error("No spectral features found — check importance files and shap_col.")
        return

    df = pd.DataFrame(rows).T  # rows = models, cols = wavelengths

    # Sort columns by wavelength numerically
    def wav_int(col):
        return int(col.replace("_nm", ""))

    sorted_cols = sorted(df.columns, key=wav_int)
    df = df[sorted_cols]

    n_models, n_wav = df.shape
    fig_w = max(12, n_wav * 0.12 + 2)
    fig_h = max(3, n_models * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = df.values.astype(float)
    im = ax.imshow(data, aspect="auto", cmap=cmap,
                   vmin=1, vmax=np.nanmax(data))

    # X-axis: wavelength labels — show every Nth to avoid crowding
    step = max(1, n_wav // 20)
    ax.set_xticks(np.arange(0, n_wav, step))
    ax.set_xticklabels(
        [sorted_cols[i] for i in range(0, n_wav, step)],
        rotation=45, ha="right", fontsize=8,
    )

    # Y-axis: model labels
    ax.set_yticks(np.arange(n_models))
    ax.set_yticklabels(df.index, fontsize=9)

    ax.set(
        xlabel="Wavelength",
        title=(
            f"Spectral wavelength importance — "
            f"Level{'s' if isinstance(level, list) else ''} "
            f"{level}\n({shap_col})"
        ),
        **kwargs,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Rank (1 = most important)", fontsize=9)

    # Vertical separators every 50nm (approximate)
    wav_vals = [wav_int(c) for c in sorted_cols]
    boundaries = [425, 475, 525, 575, 625, 675]
    for b in boundaries:
        closest = min(range(n_wav), key=lambda i: abs(wav_vals[i] - b))
        ax.axvline(closest - 0.5, color="white", linewidth=0.6, alpha=0.5)

    fig.tight_layout()

    if out_path is None:
        out_path = output_dir / f"wavelength_heatmap_level{level}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wavelength heatmap saved: {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-model feature rank visualisations for the algal turf pipeline."
    )
    parser.add_argument("--type", choices=["bump", "heatmap", "biplot", "wavelength", "both"], default="both",
                        help="Which chart(s) to produce (default: both)")
    parser.add_argument("--model-a", nargs=2, metavar=("SPECTRA", "LEVEL"),
                        default=None,
                        help="Model A for biplot e.g. --model-a A 2")
    parser.add_argument("--model-b", nargs=2, metavar=("SPECTRA", "LEVEL"),
                        default=None,
                        help="Model B for biplot e.g. --model-b A 4")
    parser.add_argument("--biplot-top-n", type=int, default=30,
                        help="Top N divergent features for biplot (default: 30)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="Root output directory (default: outputs/)")
    parser.add_argument("--level", type=int, default=3,
                        help="Level for bump chart and wavelength heatmap (default: 3)")
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 2, 1],
                        help="Levels for heatmap (default: 3 2 1)")
    parser.add_argument("--spectra", nargs="+", default=["A", "B", "C", "D"],
                        help="Spectra types to include (default: A B C D)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top N features to show in bump chart (default: 20)")
    parser.add_argument("--heatmap-top-n", type=int, default=None,
                        help="Top N features for heatmap; None = all (default: None)")
    parser.add_argument("--feature-order", type=Path, default=None,
                        help="Path to a text file with one feature name per line "
                             "defining the y-axis order (default: sort by mean rank)")
    parser.add_argument("--unweighted", action="store_true",
                        help="Use unweighted model outputs (default: weighted)")
    parser.add_argument("--shap-col", type=str, default="mean_abs_shap_global",
                        help="SHAP column to rank by (default: mean_abs_shap_global)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weighted = not args.unweighted

    if args.type in ("bump", "both"):
        bump_chart(
            output_dir=args.output_dir,
            level=args.level,
            spectra_types=args.spectra,
            top_n=args.top_n,
            weighted=weighted,
            shap_col=args.shap_col,
        )

    if args.type in ("heatmap", "both"):
        feature_order = None
        if args.feature_order is not None:
            feature_order = Path(args.feature_order).read_text().splitlines()
        rank_heatmap(
            output_dir=args.output_dir,
            levels=args.levels,
            spectra_types=args.spectra,
            weighted=weighted,
            top_n=args.heatmap_top_n,
            shap_col=args.shap_col,
            feature_order=feature_order,
        )

    if args.type == "wavelength":
        wavelength_heatmap(
            output_dir=args.output_dir,
            level=args.level,
            spectra_types=args.spectra,
            weighted=weighted,
            shap_col=args.shap_col,
        )

    if args.type == "biplot":
        if args.model_a is None or args.model_b is None:
            raise ValueError("--model-a and --model-b are required for biplot.")
        directional_bar_biplot(
            output_dir=args.output_dir,
            model_a=(args.model_a[0], int(args.model_a[1])),
            model_b=(args.model_b[0], int(args.model_b[1])),
            weighted=weighted,
            top_n=args.biplot_top_n,
            shap_col=args.shap_col,
        )


if __name__ == "__main__":
    main()