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
# Internal helper
# ---------------------------------------------------------------------------

# Parameters used by visualisation functions that must NOT be forwarded to ax.set()
_PIPELINE_KWARGS = frozenset({
    "shap_col", "weighted", "top_n", "sample_size", "random_seed",
    "out_path", "feature_order", "spectra_types", "levels", "level",
    "spectra", "model_a", "model_b", "cmap", "output_dir",
})


def _ax_kwargs(kwargs: dict) -> dict:
    """Strips pipeline-specific keys from kwargs before passing to ax.set()."""
    return {k: v for k, v in kwargs.items() if k not in _PIPELINE_KWARGS}


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
        **_ax_kwargs(kwargs),
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
    ax.set(**_ax_kwargs(kwargs))
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
        **_ax_kwargs(kwargs),
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
# SHAP beeswarm plot
# ---------------------------------------------------------------------------

def _plot_single_beeswarm(
        shap_df: "pd.DataFrame",
        top_features: list,
        class_label: str,
        spectra: str,
        level: int,
        sample_size: int,
        random_seed: int,
        model_dir: "Path",
        kwargs: dict,
) -> None:
    """Internal — plots and saves one beeswarm for a single class."""
    rng = np.random.default_rng(random_seed)
    df = shap_df.copy()
    if len(df) > sample_size:
        idx = rng.choice(len(df), sample_size, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    n = len(top_features)
    fig, ax = plt.subplots(figsize=(9, max(5, n * 0.42 + 1.5)))

    for rank, feat in enumerate(reversed(top_features)):
        shap_vals = df[feat].values
        fv = df[feat].values
        fmin, fmax = fv.min(), fv.max()
        norm = (fv - fmin) / (fmax - fmin + 1e-9)
        colours = plt.cm.coolwarm(norm)
        y_jitter = rank + rng.uniform(-0.35, 0.35, size=len(shap_vals))
        ax.scatter(shap_vals, y_jitter, c=colours, s=6, alpha=0.6,
                   linewidths=0, rasterized=True, zorder=2)

    ax.axvline(0, color="black", linewidth=0.8, zorder=3)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(list(reversed(top_features)), fontsize=8.5)
    ax.yaxis.set_tick_params(length=0)
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(
        xlabel="SHAP value (impact on model output)",
        title=f"SHAP beeswarm — Spectra {spectra}, Level {level} | class: {class_label}",
        **_ax_kwargs(kwargs),
    )
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.02)
    cbar.set_label("Feature value", fontsize=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    fig.tight_layout()

    safe_label = class_label.replace(" ", "_").replace("/", "_")
    out_path = model_dir / f"shap_beeswarm_spectra{spectra}_L{level}_{safe_label}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP beeswarm saved: {out_path}")


def shap_beeswarm(
        output_dir: Path,
        spectra: str,
        level: int,
        weighted: bool = True,
        top_n: int = 20,
        sample_size: int = 5_000,
        random_seed: int = 42,
        shap_col: str = "mean_abs_shap_global",
        class_names: list[str] | None = None,
        out_path: Path | None = None,
        **kwargs,
) -> None:
    """
    SHAP beeswarm plot — one PNG per class, saved to the model output directory.

    For binary models: one plot for the positive class.
    For multiclass: one plot per class (columns named {feat}__class{c} in SHAP CSV).
    Raises ValueError if n_classes > 20.

    Output filenames: shap_beeswarm_spectra{spectra}_L{level}_{class_label}.png
    Saved to: outputs/spectra_{spectra}/level_{level}/

    Args:
        output_dir:   Root output directory.
        spectra:      Spectra type label e.g. 'A'.
        level:        Hierarchy level.
        weighted:     Use weighted model outputs.
        top_n:        Number of top features to show per plot.
        sample_size:  Max samples to plot (subsampled if larger).
        random_seed:  For reproducible subsampling.
        shap_col:     Importance column for feature ranking.
        class_names:  Class label strings. If None, loaded from training_metadata.json.
        out_path:     Ignored — output path is always auto-generated per class.
        **kwargs:     Passed to ax.set() for title/label overrides.
    """
    from utils.io import load_json

    suffix = "" if weighted else "_unweighted"
    model_dir = output_dir / f"spectra_{spectra}" / f"level_{level}{suffix}"

    # Load SHAP values
    shap_csv = model_dir / "shap_values.csv"
    shap_pq = model_dir / "shap_values.parquet"
    if shap_csv.exists():
        shap_df = pd.read_csv(shap_csv)
    elif shap_pq.exists():
        shap_df = pd.read_parquet(shap_pq)
    else:
        logger.error(f"No SHAP values found in {model_dir} — run shap.py first.")
        return

    # Load class names from training metadata if not provided
    if class_names is None:
        meta_path = model_dir / "training_metadata.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            class_map = meta.get("class_mapping", {})
            class_names = [class_map[str(i)] for i in range(len(class_map))]
        else:
            class_names = None

    # Detect binary vs multiclass from column names
    multiclass_cols = [c for c in shap_df.columns if "__class" in c]
    is_multiclass = len(multiclass_cols) > 0

    if is_multiclass:
        # Infer n_classes from column suffix
        class_indices = sorted(set(
            int(c.split("__class")[-1]) for c in multiclass_cols
        ))
        n_classes = len(class_indices)
        if n_classes > 20:
            raise ValueError(
                f"n_classes={n_classes} exceeds the maximum of 20 for beeswarm plots. "
                f"Specify a subset via class_names."
            )
    else:
        n_classes = 2
        class_indices = [1]  # Binary: only plot positive class

    # Load feature importance for ranking
    imp_path = model_dir / "feature_importance_shap.csv"
    if not imp_path.exists():
        logger.error(f"Missing: {imp_path}")
        return
    imp = pd.read_csv(imp_path, index_col=0)
    if shap_col not in imp.columns:
        logger.warning(f"'{shap_col}' not in importance — falling back to first column.")
        shap_col = imp.columns[0]

    logger.info(f"Generating {n_classes} beeswarm plot(s) for spectra {spectra} level {level}")

    for c_idx in class_indices:
        if is_multiclass:
            # Extract columns for this class: {feat}__class{c}
            class_cols = [col for col in multiclass_cols if col.endswith(f"__class{c_idx}")]
            feat_names = [col.replace(f"__class{c_idx}", "") for col in class_cols]
            class_shap_df = shap_df[class_cols].copy()
            class_shap_df.columns = feat_names
            # Rank by global importance for this class (use same shap_col if available)
            avail_feats = [f for f in imp.index if f in feat_names]
            top_features = (
                imp.loc[avail_feats, shap_col]
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
            label = class_names[c_idx] if class_names and c_idx < len(class_names) else f"class{c_idx}"
        else:
            # Binary: use columns directly (no __class suffix)
            feat_names = [c for c in shap_df.columns if "__class" not in c]
            class_shap_df = shap_df[feat_names].copy()
            avail_feats = [f for f in imp.index if f in feat_names]
            top_features = (
                imp.loc[avail_feats, shap_col]
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
            label = class_names[1] if class_names and len(class_names) > 1 else "positive"

        _plot_single_beeswarm(
            shap_df=class_shap_df,
            top_features=top_features,
            class_label=label,
            spectra=spectra,
            level=level,
            sample_size=sample_size,
            random_seed=random_seed,
            model_dir=model_dir,
            kwargs=kwargs,
        )


# ---------------------------------------------------------------------------
# SHAP waterfall plot
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Waterfall axis helper + interesting 2×2 plot
# ---------------------------------------------------------------------------

def _draw_waterfall_on_ax(
        ax: "plt.Axes",
        steps: list,
        base_value: float,
        final_value: float,
        title: str,
        kwargs: dict,
) -> None:
    """
    Internal — draws one waterfall chart onto an existing Axes object.
    Shared by shap_waterfall() (single figure) and waterfall_interesting() (2×2 grid).
    """
    n_bars = len(steps)
    running = base_value
    bar_colours, bar_starts, bar_widths, labels = [], [], [], []

    for feat, val in steps:
        bar_starts.append(running)
        bar_widths.append(val)
        bar_colours.append("#E8334A" if val >= 0 else "#3B82C4")
        labels.append(feat)
        running += val

    y_pos = np.arange(n_bars)
    spread = abs(final_value - base_value) or 1.0

    for i, (start, width, colour) in enumerate(zip(bar_starts, bar_widths, bar_colours)):
        ax.barh(i, width, left=start, color=colour, height=0.55,
                edgecolor="white", linewidth=0.5, zorder=2)
        sign = "+" if width >= 0 else ""
        ha = "left" if width >= 0 else "right"
        ax.text(start + width + (0.005 * spread * (1 if width >= 0 else -1)),
                i, f"{sign}{width:.3f}",
                va="center", ha=ha, fontsize=7, color=colour, fontweight=500)

    ax.axvline(base_value, color="gray", linewidth=0.9, linestyle="--", alpha=0.7, zorder=1)
    ax.axvline(final_value, color="black", linewidth=1.1, linestyle="-", alpha=0.9, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.yaxis.set_tick_params(length=0)

    running2 = base_value
    for i, (_, val) in enumerate(steps[:-1]):
        x_conn = running2 + val
        ax.plot([x_conn, x_conn], [i + 0.28, i + 0.72],
                color="gray", linewidth=0.6, linestyle=":", zorder=1)
        running2 += val

    y_top = n_bars - 0.5
    ax.text(base_value, y_top + 0.12, f"E[f(x)]={base_value:.3f}",
            ha="center", fontsize=7, color="gray")
    ax.text(final_value, y_top + 0.12, f"f(x)={final_value:.3f}",
            ha="center", fontsize=7, color="black", fontweight=500)

    ax.set(xlabel=f"Model output   E[f(x)] = {base_value:.3f}",
           title=title, **_ax_kwargs(kwargs))
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def waterfall_interesting(
        output_dir: Path,
        spectra: str,
        level: int,
        weighted: bool = True,
        top_n: int = 10,
        class_names: list[str] | None = None,
        **kwargs,
) -> None:
    """
    For each class, produces one figure with 4 waterfall subplots (2×2):
        top-left:     most_confident
        top-right:    least_confident
        bottom-left:  most_uncertain
        bottom-right: misclassified  (greyed out if unavailable)

    Calls find_interesting_samples() internally — no need to run it separately.
    One PNG per class saved to the model output directory.

    Output: shap_waterfall_interesting_spectra{spectra}_L{level}_{class_label}.png

    Args:
        output_dir:   Root output directory.
        spectra:      Spectra type label e.g. 'A'.
        level:        Hierarchy level.
        weighted:     Use weighted model outputs.
        top_n:        Features per subplot.
        class_names:  Override class names (loaded from metadata if None).
        **kwargs:     Passed to each ax.set() via _ax_kwargs.
    """
    from utils.io import load_json

    suffix = "" if weighted else "_unweighted"
    model_dir = output_dir / f"spectra_{spectra}" / f"level_{level}{suffix}"

    # Load SHAP values
    shap_csv = model_dir / "shap_values.csv"
    shap_pq = model_dir / "shap_values.parquet"
    if shap_csv.exists():
        shap_df = pd.read_csv(shap_csv)
    elif shap_pq.exists():
        shap_df = pd.read_parquet(shap_pq)
    else:
        logger.error(f"No SHAP values found in {model_dir} — run shap.py first.")
        return

    # Load class names
    if class_names is None:
        meta_path = model_dir / "training_metadata.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            class_map = meta["class_mapping"]
            class_names = [class_map[str(i)] for i in range(len(class_map))]
        else:
            logger.error("training_metadata.json not found.")
            return

    n_classes = len(class_names)
    if n_classes > 20:
        raise ValueError(f"n_classes={n_classes} exceeds maximum of 20.")

    # Load base values
    bias_path = model_dir / "shap_base_values.csv"
    bias_df = pd.read_csv(bias_path) if bias_path.exists() else None

    # Detect binary vs multiclass
    multiclass_cols = [c for c in shap_df.columns if "__class" in c]
    is_multiclass = len(multiclass_cols) > 0

    # Get interesting sample indices
    samples_df = find_interesting_samples(
        output_dir=output_dir, spectra=spectra,
        level=level, weighted=weighted, n_per_class=1,
    )

    categories = ["most_confident", "least_confident", "most_uncertain", "misclassified"]
    cat_titles = {
        "most_confident": "Most confident",
        "least_confident": "Least confident",
        "most_uncertain": "Most uncertain",
        "misclassified": "Misclassified",
    }

    for c_idx, cls_name in enumerate(class_names):
        cls_samples = samples_df[samples_df["class_idx"] == c_idx]

        fig, axes = plt.subplots(2, 2, figsize=(16, max(8, top_n * 0.6 + 3)))
        fig.suptitle(
            f"SHAP waterfall — Spectra {spectra}, Level {level} | class: {cls_name}",
            fontsize=12, fontweight=500, y=1.01,
        )

        for ax, category in zip(axes.flat, categories):
            row_match = cls_samples[cls_samples["category"] == category]

            if row_match.empty:
                # Grey out unavailable subplot
                ax.set_facecolor("#F5F5F5")
                ax.text(0.5, 0.5, f"{cat_titles[category]}\n(not available)",
                        ha="center", va="center", fontsize=10,
                        color="gray", transform=ax.transAxes)
                ax.axis("off")
                continue

            sample_idx = int(row_match.iloc[0]["sample_idx"])
            prob = float(row_match.iloc[0]["probability"])
            pred_cls = row_match.iloc[0]["predicted_class"]

            # Extract SHAP row for this class
            if is_multiclass:
                class_cols = [c for c in multiclass_cols if c.endswith(f"__class{c_idx}")]
                feat_names = [c.replace(f"__class{c_idx}", "") for c in class_cols]
                row_df = shap_df[class_cols].copy()
                row_df.columns = feat_names
                row = row_df.iloc[sample_idx]
            else:
                feat_cols = [c for c in shap_df.columns if "__class" not in c]
                row = shap_df[feat_cols].iloc[sample_idx]

            # Base and final values
            if bias_df is not None and sample_idx < len(bias_df):
                base_value = float(bias_df["base_value"].iloc[sample_idx])
            else:
                base_value = float(shap_df.sum(axis=1).mean() - shap_df.sum(axis=1).std())
            final_value = float(row.sum()) + base_value

            # Build steps
            row_sorted = row.abs().sort_values(ascending=False)
            top_feats = row_sorted.head(top_n).index.tolist()
            other_feats = row_sorted.index[top_n:].tolist()
            steps = [(f, float(row[f])) for f in top_feats]
            if other_feats:
                steps.append((f"{len(other_feats)} other features",
                              float(row[other_feats].sum())))

            # Subplot title
            if category == "misclassified":
                sub_title = f"{cat_titles[category]} (predicted: {pred_cls})\nP({cls_name})={prob:.3f}  sample {sample_idx}"
            else:
                sub_title = f"{cat_titles[category]}\nP({cls_name})={prob:.3f}  sample {sample_idx}"

            _draw_waterfall_on_ax(ax, steps, base_value, final_value, sub_title, kwargs)

        fig.tight_layout()
        safe_label = cls_name.replace(" ", "_").replace("/", "_")
        out_path = model_dir / f"shap_waterfall_interesting_spectra{spectra}_L{level}_{safe_label}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Interesting waterfall saved: {out_path}")


def shap_waterfall(
        output_dir: Path,
        spectra: str,
        level: int,
        sample_idx: int = 0,
        weighted: bool = True,
        top_n: int = 10,
        class_idx: int | None = None,
        class_names: list[str] | None = None,
        out_path: Path | None = None,
        **kwargs,
) -> None:
    """
    SHAP waterfall plot for a single sample.
    Shows how each feature pushes the prediction from E[f(x)] to f(x).
    For multiclass, specify class_idx to select which class to explain.
    Saved to: outputs/spectra_{spectra}/level_{level}/

    Args:
        output_dir:   Root output directory.
        spectra:      Spectra type label.
        level:        Hierarchy level.
        sample_idx:   Row index into the SHAP values file to explain.
        weighted:     Use weighted model outputs.
        top_n:        Number of individual features to show (rest collapsed).
        class_idx:    Class index for multiclass models. None = binary or class 0.
        class_names:  Class label strings. If None, loaded from training_metadata.json.
        out_path:     Output PNG path. Auto-generated if None.
        **kwargs:     Passed to ax.set() for title/label overrides.
    """
    from utils.io import load_json

    suffix = "" if weighted else "_unweighted"
    model_dir = output_dir / f"spectra_{spectra}" / f"level_{level}{suffix}"

    # Load SHAP values
    shap_csv = model_dir / "shap_values.csv"
    shap_pq = model_dir / "shap_values.parquet"
    if shap_csv.exists():
        shap_df = pd.read_csv(shap_csv)
    elif shap_pq.exists():
        shap_df = pd.read_parquet(shap_pq)
    else:
        logger.error(f"No SHAP values found in {model_dir}")
        return

    if sample_idx >= len(shap_df):
        logger.error(f"sample_idx {sample_idx} out of range ({len(shap_df)} rows)")
        return

    # Detect binary vs multiclass
    multiclass_cols = [c for c in shap_df.columns if "__class" in c]
    is_multiclass = len(multiclass_cols) > 0

    # Load class names from training metadata if not provided
    if class_names is None:
        meta_path = model_dir / "training_metadata.json"
        if meta_path.exists():
            meta = load_json(meta_path)
            class_map = meta.get("class_mapping", {})
            class_names = [class_map[str(i)] for i in range(len(class_map))]

    if is_multiclass:
        c_idx = class_idx if class_idx is not None else 0
        n_classes = len(set(int(c.split("__class")[-1]) for c in multiclass_cols))
        if n_classes > 20:
            raise ValueError(f"n_classes={n_classes} exceeds maximum of 20.")
        # Extract columns for chosen class
        class_cols = [c for c in multiclass_cols if c.endswith(f"__class{c_idx}")]
        feat_names = [c.replace(f"__class{c_idx}", "") for c in class_cols]
        row_df = shap_df[class_cols].copy()
        row_df.columns = feat_names
        row = row_df.iloc[sample_idx]
        class_label = class_names[c_idx] if class_names and c_idx < len(class_names) else f"class{c_idx}"
    else:
        feat_cols = [c for c in shap_df.columns if "__class" not in c]
        row = shap_df[feat_cols].iloc[sample_idx]
        class_label = class_names[1] if class_names and len(class_names) > 1 else "positive"

    # Load exact base value E[f(x)] saved by compute_shap_values()
    bias_path = model_dir / "shap_base_values.csv"
    if bias_path.exists():
        bias_df = pd.read_csv(bias_path)
        base_value = float(bias_df["base_value"].iloc[sample_idx])
        logger.info(f"Base value loaded from file: {base_value:.4f}")
    else:
        # Fallback approximation if shap.py was run before this fix
        logger.warning("shap_base_values.csv not found — re-run shap.py for exact base value. Using approximation.")
        base_value = float(shap_df.sum(axis=1).mean() - shap_df.sum(axis=1).std())
    final_value = float(row.sum()) + base_value

    # Sort features by |SHAP| for this sample
    row_sorted = row.abs().sort_values(ascending=False)
    top_feats = row_sorted.head(top_n).index.tolist()
    other_feats = row_sorted.index[top_n:].tolist()
    steps = [(f, float(row[f])) for f in top_feats]
    if other_feats:
        steps.append((f"{len(other_feats)} other features", float(row[other_feats].sum())))

    subtitle = (
        f"SHAP waterfall — Spectra {spectra}, Level {level} | class: {class_label}\n"
        f"sample {sample_idx}   f(x) = {final_value:.3f}"
    )
    fig, ax = plt.subplots(figsize=(8, max(5, len(steps) * 0.52 + 2)))
    _draw_waterfall_on_ax(ax, steps, base_value, final_value, subtitle, kwargs)
    fig.tight_layout()

    if out_path is None:
        safe_label = class_label.replace(" ", "_").replace("/", "_")
        out_path = model_dir / f"shap_waterfall_spectra{spectra}_L{level}_{safe_label}_s{sample_idx}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"SHAP waterfall saved: {out_path}")


# ---------------------------------------------------------------------------
# Interesting sample selector
# ---------------------------------------------------------------------------

def find_interesting_samples(
        output_dir: Path,
        spectra: str,
        level: int,
        weighted: bool = True,
        n_per_class: int = 1,
) -> "pd.DataFrame":
    """
    Identifies interesting sample indices from SHAP and prediction outputs
    for use with shap_waterfall(). For each class, finds:

        1. most_confident   — highest predicted probability for this class
        2. least_confident  — lowest predicted probability for this class
        3. most_uncertain   — predicted probability closest to 1/n_classes (decision boundary)
        4. misclassified    — true label is this class but model predicted another
                              (requires true labels in evaluation outputs)

    Results saved to: outputs/spectra_{spectra}/level_{level}/interesting_samples.csv
    Printed as a summary table for easy reference before calling shap_waterfall().

    Args:
        output_dir:   Root output directory.
        spectra:      Spectra type label e.g. 'A'.
        level:        Hierarchy level.
        weighted:     Use weighted model outputs.
        n_per_class:  Number of samples to return per category per class (default 1).

    Returns:
        DataFrame with columns:
            class_idx, class_label, category, sample_idx, probability
    """
    from utils.io import load_json

    suffix = "" if weighted else "_unweighted"
    model_dir = output_dir / f"spectra_{spectra}" / f"level_{level}{suffix}"

    # Load training metadata for class names
    meta_path = model_dir / "training_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"training_metadata.json not found in {model_dir}")
    meta = load_json(meta_path)
    class_map = meta["class_mapping"]
    class_names = [class_map[str(i)] for i in range(len(class_map))]
    n_classes = len(class_names)

    # Load SHAP values to get row count and index alignment
    shap_csv = model_dir / "shap_values.csv"
    shap_pq = model_dir / "shap_values.parquet"
    if shap_csv.exists():
        shap_df = pd.read_csv(shap_csv)
    elif shap_pq.exists():
        shap_df = pd.read_parquet(shap_pq)
    else:
        raise FileNotFoundError(f"No SHAP values found in {model_dir} — run shap.py first.")

    n_samples = len(shap_df)

    # Load boundary samples CSV for misclassified/uncertain flags
    bnd_path = model_dir / "boundary_samples.csv"
    bnd_df = pd.read_csv(bnd_path) if bnd_path.exists() else None

    # Load classification report for per-class probability reconstruction
    # We need predicted probabilities — reconstruct from SHAP sums as proxy
    # For exact probabilities, load from a saved pred_proba if available
    # otherwise approximate: P(class c) ~ softmax of sum of SHAP for class c

    # Build probability matrix (n_samples, n_classes)
    multiclass_cols = [c for c in shap_df.columns if "__class" in c]
    is_multiclass = len(multiclass_cols) > 0

    if is_multiclass:
        # Stack SHAP sums per class as raw scores, apply softmax
        scores = np.zeros((n_samples, n_classes))
        for c_idx in range(n_classes):
            cols = [c for c in multiclass_cols if c.endswith(f"__class{c_idx}")]
            if cols:
                scores[:, c_idx] = shap_df[cols].sum(axis=1).values
        # Softmax
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        proba = exp_s / exp_s.sum(axis=1, keepdims=True)
        pred_cls = proba.argmax(axis=1)
    else:
        # Binary: SHAP sum approximates log-odds; sigmoid to probability
        feat_cols = [c for c in shap_df.columns if "__class" not in c]
        log_odds = shap_df[feat_cols].sum(axis=1).values
        p1 = 1 / (1 + np.exp(-log_odds))
        proba = np.column_stack([1 - p1, p1])
        pred_cls = (p1 >= 0.5).astype(int)

    # Decision boundary = 1/n_classes
    boundary = 1.0 / n_classes

    # Try to get true labels from boundary_samples if available
    true_labels = None
    if bnd_df is not None and "true_label" in bnd_df.columns:
        # boundary_samples has a subset — build a true_label array aligned to shap rows
        # Note: boundary_samples may be a subset of the test set
        # We can only use misclassified info if index aligns
        if len(bnd_df) == n_samples:
            true_labels = bnd_df["true_label"].values

    records = []
    for c_idx, cls_name in enumerate(class_names):
        p_cls = proba[:, c_idx]  # Probability of this class for all samples

        # 1. Most confident — highest P(class)
        top_idx = np.argsort(p_cls)[::-1][:n_per_class]
        for idx in top_idx:
            records.append({
                "class_idx": c_idx,
                "class_label": cls_name,
                "category": "most_confident",
                "sample_idx": int(idx),
                "probability": round(float(p_cls[idx]), 4),
                "predicted_class": class_names[pred_cls[idx]],
            })

        # 2. Least confident — lowest P(class)
        bot_idx = np.argsort(p_cls)[:n_per_class]
        for idx in bot_idx:
            records.append({
                "class_idx": c_idx,
                "class_label": cls_name,
                "category": "least_confident",
                "sample_idx": int(idx),
                "probability": round(float(p_cls[idx]), 4),
                "predicted_class": class_names[pred_cls[idx]],
            })

        # 3. Most uncertain — P(class) closest to decision boundary
        dist_to_boundary = np.abs(p_cls - boundary)
        unc_idx = np.argsort(dist_to_boundary)[:n_per_class]
        for idx in unc_idx:
            records.append({
                "class_idx": c_idx,
                "class_label": cls_name,
                "category": "most_uncertain",
                "sample_idx": int(idx),
                "probability": round(float(p_cls[idx]), 4),
                "predicted_class": class_names[pred_cls[idx]],
            })

        # 4. Misclassified — true label is this class but predicted as another
        if true_labels is not None:
            # Encode true labels to integers
            true_int = np.array([
                class_names.index(lbl) if lbl in class_names else -1
                for lbl in true_labels
            ])
            mis_mask = (true_int == c_idx) & (pred_cls != c_idx)
            mis_idx = np.where(mis_mask)[0]
            if len(mis_idx) > 0:
                # Sort by how wrong the model was (lowest P(true class))
                mis_sorted = mis_idx[np.argsort(p_cls[mis_idx])][:n_per_class]
                for idx in mis_sorted:
                    records.append({
                        "class_idx": c_idx,
                        "class_label": cls_name,
                        "category": "misclassified",
                        "sample_idx": int(idx),
                        "probability": round(float(p_cls[idx]), 4),
                        "predicted_class": class_names[pred_cls[idx]],
                    })
            else:
                logger.info(f"No misclassified samples found for class '{cls_name}'.")
        else:
            logger.info(
                "True labels not available — skipping misclassified category. "
                "Re-run evaluate.py to generate boundary_samples.csv."
            )

    result_df = pd.DataFrame(records)

    # Save and print
    out_path = model_dir / "interesting_samples.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Interesting samples saved: {out_path}")

    # Pretty print summary
    print(f"\nInteresting samples — Spectra {spectra}, Level {level}")
    print(f"Decision boundary = 1/{n_classes} = {boundary:.3f}")
    print(f"{'Class':<30} {'Category':<20} {'sample_idx':>10} {'P(class)':>10} {'Predicted as'}")
    print("-" * 90)
    for _, row in result_df.iterrows():
        print(
            f"  {row['class_label']:<28} {row['category']:<20} "
            f"{row['sample_idx']:>10} {row['probability']:>10.4f} "
            f"  {row['predicted_class']}"
        )

    return result_df

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-model feature rank visualisations for the algal turf pipeline."
    )
    parser.add_argument("--type",
                        choices=["bump", "heatmap", "biplot", "wavelength", "beeswarm", "waterfall", "interesting",
                                 "both"], default="both",
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

    if args.type == "interesting":
        if args.model_a is None:
            raise ValueError("--model-a is required for interesting samples e.g. --model-a A 3")
        waterfall_interesting(
            output_dir=args.output_dir,
            spectra=args.model_a[0],
            level=int(args.model_a[1]),
            weighted=weighted,
        )

    if args.type == "beeswarm":
        if args.model_a is None:
            raise ValueError("--model-a is required for beeswarm e.g. --model-a A 3")
        shap_beeswarm(
            output_dir=args.output_dir,
            spectra=args.model_a[0],
            level=int(args.model_a[1]),
            weighted=weighted,
            shap_col=args.shap_col,
        )

    if args.type == "waterfall":
        if args.model_a is None:
            raise ValueError("--model-a is required for waterfall e.g. --model-a A 3")
        shap_waterfall(
            output_dir=args.output_dir,
            spectra=args.model_a[0],
            level=int(args.model_a[1]),
            weighted=weighted,
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