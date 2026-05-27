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
# UMAP
# ---------------------------------------------------------------------------

def _load_embedding_matrix(
        model_dir: "Path",
        source: str,
        booster: object | None,
        feature_cols: list[str] | None,
        df: "pd.DataFrame | None",
        y: "np.ndarray | None",
) -> "np.ndarray | None":
    """
    Internal — loads or computes the embedding matrix for UMAP.
    source: 'leaf' or 'shap'
    """
    if source == "shap":
        shap_csv = model_dir / "shap_values.csv"
        shap_pq = model_dir / "shap_values.parquet"
        if shap_csv.exists():
            return pd.read_csv(shap_csv).values.astype(float)
        elif shap_pq.exists():
            return pd.read_parquet(shap_pq).values.astype(float)
        else:
            logger.error(f"No SHAP values found in {model_dir} — run shap.py first.")
            return None
    elif source == "leaf":
        if booster is None or df is None or y is None or feature_cols is None:
            logger.error("booster, df, feature_cols and y are required for leaf embeddings.")
            return None
        import xgboost as xgb
        dm = xgb.DMatrix(df[feature_cols].values, label=y,
                         feature_names=feature_cols)
        return booster.predict(dm, pred_leaf=True).astype(float)
    else:
        raise ValueError(f"source must be 'leaf' or 'shap', got '{source}'")


def umap_plot(
        output_dir: "Path",
        spectra: str,
        level: int,
        source: str = "leaf",
        weighted: bool = True,
        sample_size: int = 10_000,
        random_seed: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        booster: object | None = None,
        feature_cols: list[str] | None = None,
        df: "pd.DataFrame | None" = None,
        y: "np.ndarray | None" = None,
        le: object | None = None,
        turf_algae_class: str = "turf_algae",
        out_path: "Path | None" = None,
        **kwargs,
) -> None:
    """
    UMAP dimensionality reduction plot for one model.
    Saved to the model output directory.

    Two input sources (controlled by `source`):
        'leaf'  — XGBoost leaf node indices (pred_leaf=True).
                  Captures the tree structure and decision boundaries.
                  Requires booster, df, feature_cols, y, le to be passed.
        'shap'  — SHAP values matrix loaded from shap_values.csv/.parquet.
                  Captures feature contribution space — closer to the
                  explanation space than the tree space.
                  Requires only output_dir/spectra/level (file already saved).

    For Level 4 models: ROI numbers plotted as text markers coloured by
    Level 2 class (same as plot_pca_tsne). For all other levels: scatter
    coloured by class with turf algae plotted on top.

    Args:
        output_dir:       Root output directory.
        spectra:          Spectra type label e.g. 'A'.
        level:            Hierarchy level.
        source:           'leaf' or 'shap' — which matrix to embed.
        weighted:         Use weighted model outputs.
        sample_size:      Max rows to embed (subsampled if larger).
        random_seed:      Reproducibility seed.
        n_neighbors:      UMAP n_neighbors (controls local vs global structure).
        min_dist:         UMAP min_dist (controls cluster tightness).
        booster:          Trained XGBoost Booster (required for source='leaf').
        feature_cols:     Feature column names (required for source='leaf').
        df:               Feature DataFrame (required for source='leaf').
        y:                Integer-encoded labels (required for source='leaf').
        le:               Fitted LabelEncoder (required for source='leaf').
        turf_algae_class: Class name for turf algae — plotted on top.
        out_path:         Output PNG path. Auto-generated if None.
        **kwargs:         Passed to ax.set() for title/label overrides.
    """
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError(
            "umap-learn is not installed. Install with:\n"
            "  pip install umap-learn\n"
            "umap-learn is an optional dependency not required by the main pipeline."
        )

    import re as _re
    from matplotlib.patheffects import withStroke

    suffix = "" if weighted else "_unweighted"
    model_dir = Path(output_dir) / f"spectra_{spectra}" / f"level_{level}{suffix}"

    # Load class names from training metadata
    from utils.io import load_json
    meta_path = model_dir / "training_metadata.json"
    class_names = None
    if meta_path.exists():
        meta = load_json(meta_path)
        class_map = meta["class_mapping"]
        class_names = [class_map[str(i)] for i in range(len(class_map))]

    # Load or compute embedding matrix
    X = _load_embedding_matrix(model_dir, source, booster, feature_cols, df, y)
    if X is None:
        return

    # Load labels — from le if provided, else from training metadata class_names
    if le is not None and y is not None:
        labels_arr = le.inverse_transform(y)
    elif class_names is not None and y is not None:
        labels_arr = np.array([class_names[i] if i < len(class_names) else str(i) for i in y])
    else:
        labels_arr = np.array(["unknown"] * len(X))

    # Subsample
    rng = np.random.default_rng(random_seed)
    n = len(X)
    if n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X = X[idx]
        labels_arr = labels_arr[idx]
        logger.info(f"UMAP subsample: {sample_size:,} rows from {n:,}")
    else:
        logger.info(f"UMAP using all {n:,} rows")

    # Fit UMAP
    logger.info(f"Fitting UMAP (source={source}, n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_seed,
    )
    X_2d = reducer.fit_transform(X)
    logger.info("UMAP complete")

    # Detect Level 4
    is_level4 = any("_ROI_" in c for c in (class_names or []))
    roi_pattern = _re.compile(r"^(.+)_ROI_(\d+)$")
    white_edge = [withStroke(linewidth=1.0, foreground="white")]

    fig, ax = plt.subplots(figsize=(9, 7))

    if is_level4:
        parents = sorted(set(
            roi_pattern.match(c).group(1) if roi_pattern.match(c) else c
            for c in (class_names or [])
        ))
        cmap_ = plt.cm.get_cmap("tab10", len(parents))
        colours_ = {p: cmap_(i) for i, p in enumerate(parents)}

        for lbl, x, y_ in zip(labels_arr, X_2d[:, 0], X_2d[:, 1]):
            m = roi_pattern.match(lbl)
            parent = m.group(1) if m else lbl
            roi_n = str(int(m.group(2))) if m else "?"
            colour = colours_.get(parent, "gray")
            ax.text(x, y_, roi_n, fontsize=5, color=colour,
                    ha="center", va="center", alpha=0.9,
                    fontweight="normal", clip_on=False,
                    path_effects=white_edge)

        pad = ((np.nanmax(X_2d[:, 0]) - np.nanmin(X_2d[:, 0])) +
               (np.nanmax(X_2d[:, 1]) - np.nanmin(X_2d[:, 1]))) * 0.05
        ax.set_xlim(np.nanmin(X_2d[:, 0]) - pad, np.nanmax(X_2d[:, 0]) + pad)
        ax.set_ylim(np.nanmin(X_2d[:, 1]) - pad, np.nanmax(X_2d[:, 1]) + pad)

        import matplotlib.patches as _patches
        legend_handles = [
            _patches.Patch(color=colours_[p], label=p) for p in parents
        ]
        ax.legend(handles=legend_handles, fontsize=8, framealpha=0.7,
                  loc="best", title="Level 2 class", title_fontsize=8)

    else:
        unique_cls = list(class_names) if class_names else sorted(set(labels_arr))
        other_cls = [c for c in unique_cls if c != turf_algae_class]
        plot_order = other_cls + ([turf_algae_class] if turf_algae_class in unique_cls else [])
        cmap_ = plt.cm.get_cmap("tab10", len(plot_order))
        colours_ = {cls: cmap_(i) for i, cls in enumerate(plot_order)}

        for cls in plot_order:
            mask = labels_arr == cls
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[colours_[cls]],
                label=f"{cls} (n={mask.sum():,})",
                s=8,
                alpha=0.7,
                linewidths=0,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
        ax.legend(fontsize=8, markerscale=1.5, framealpha=0.7,
                  loc="best", title="Class", title_fontsize=8)

    title = (
        f"UMAP ({source}) — Spectra {spectra}, Level {level}\n"
        f"n_neighbors={n_neighbors}, min_dist={min_dist}"
    )
    if is_level4:
        title += "  |  markers = ROI number, colour = Level 2 class"

    ax.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2", **_ax_kwargs(kwargs))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if out_path is None:
        out_path = model_dir / f"umap_{source}_spectra{spectra}_L{level}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"UMAP plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Pairwise class SHAP comparison
# ---------------------------------------------------------------------------

def pairwise_class_shap(
        output_dir: Path,
        spectra: str,
        level: int,
        class_a: str,
        class_b: str,
        weighted: bool = True,
        top_n: int = 30,
        use_raw: bool = True,
        shap_col_prefix: str = "mean_abs_shap_",
        out_path: Path | None = None,
        **kwargs,
) -> "pd.DataFrame":
    """
    Pairwise SHAP comparison between two classes within one multiclass model.
    Identifies which features best discriminate class_a from class_b.

    Two modes (controlled by use_raw):

        use_raw=True  (default):
            Uses raw SHAP values from shap_values.parquet/csv.
            Filters to pixels with true labels class_a or class_b only.
            For each feature, computes mean SHAP value for class_a pixels minus
            mean SHAP value for class_b pixels. This captures directionality —
            features that push toward class_a vs class_b.
            Requires true label information in the SHAP file (aligned rows).

        use_raw=False:
            Uses mean |SHAP| columns from feature_importance_shap.csv.
            Computes: mean_abs_shap_{class_a} - mean_abs_shap_{class_b}.
            Faster, no row filtering needed. Loses directionality but
            sufficient for ranking which features are more diagnostic for
            one class vs the other.

    Output: directional bar chart — positive bars = more diagnostic for class_a,
    negative bars = more diagnostic for class_b.

    Saved to: outputs/spectra_{spectra}/level_{level}/
              pairwise_shap_{class_a}_vs_{class_b}.png
              pairwise_shap_{class_a}_vs_{class_b}.csv

    Args:
        output_dir:       Root output directory.
        spectra:          Spectra type label e.g. 'A'.
        level:            Hierarchy level.
        class_a:          First class name (positive direction in plot).
        class_b:          Second class name (negative direction in plot).
        weighted:         Use weighted model outputs.
        top_n:            Number of most discriminating features to show.
        use_raw:          If True, use raw SHAP values; if False, use importance CSV.
        shap_col_prefix:  Prefix for importance columns (default: mean_abs_shap_).
        out_path:         Output PNG path. Auto-generated if None.
        **kwargs:         Passed to ax.set().

    Returns:
        DataFrame with feature discriminability scores sorted by absolute value.
    """
    from utils.io import load_json

    suffix = "" if weighted else "_unweighted"
    model_dir = Path(output_dir) / f"spectra_{spectra}" / f"level_{level}{suffix}"

    # Load class names
    meta_path = model_dir / "training_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"training_metadata.json not found: {meta_path}")
    meta = load_json(meta_path)
    class_map = meta["class_mapping"]
    class_names = [class_map[str(i)] for i in range(len(class_map))]

    for cls in [class_a, class_b]:
        if cls not in class_names:
            raise ValueError(
                f"Class '{cls}' not found in model. "
                f"Available classes: {class_names}"
            )

    c_idx_a = class_names.index(class_a)
    c_idx_b = class_names.index(class_b)

    if use_raw:
        # Load raw SHAP values
        shap_csv = model_dir / "shap_values.csv"
        shap_pq = model_dir / "shap_values.parquet"
        if shap_pq.exists():
            shap_df = pd.read_parquet(shap_pq)
        elif shap_csv.exists():
            shap_df = pd.read_csv(shap_csv)
        else:
            raise FileNotFoundError(
                f"No SHAP values found in {model_dir} — run shap.py first."
            )

        # Detect multiclass columns: {feat}__class{c}
        multiclass_cols = [c for c in shap_df.columns if "__class" in c]
        if not multiclass_cols:
            raise ValueError(
                "Raw SHAP file appears to be binary — use use_raw=False or "
                "ensure this is a multiclass model."
            )

        # Get SHAP columns for class_a and class_b
        cols_a = {c.replace(f"__class{c_idx_a}", ""): c
                  for c in multiclass_cols if c.endswith(f"__class{c_idx_a}")}
        cols_b = {c.replace(f"__class{c_idx_b}", ""): c
                  for c in multiclass_cols if c.endswith(f"__class{c_idx_b}")}
        features = sorted(set(cols_a.keys()) & set(cols_b.keys()))

        # Compute mean SHAP per feature for class_a and class_b pixels
        # Use all pixels — mean SHAP across all samples for each class index
        mean_a = shap_df[[cols_a[f] for f in features]].mean()
        mean_b = shap_df[[cols_b[f] for f in features]].mean()
        mean_a.index = features
        mean_b.index = features

        diff = mean_a - mean_b
        result_df = pd.DataFrame({
            "feature": features,
            f"mean_shap_{class_a}": mean_a.values,
            f"mean_shap_{class_b}": mean_b.values,
            "discriminability": diff.values,
        })
        method_label = "mean SHAP difference (raw)"

    else:
        # Use feature_importance_shap.csv
        imp_path = model_dir / "feature_importance_shap.csv"
        if not imp_path.exists():
            raise FileNotFoundError(f"feature_importance_shap.csv not found: {imp_path}")
        imp = pd.read_csv(imp_path, index_col=0)

        col_a = f"{shap_col_prefix}{class_a.replace(' ', '_')}"
        col_b = f"{shap_col_prefix}{class_b.replace(' ', '_')}"

        missing = [c for c in [col_a, col_b] if c not in imp.columns]
        if missing:
            raise ValueError(
                f"Columns not found in importance file: {missing}\n"
                f"Available: {list(imp.columns)}"
            )

        diff = imp[col_a] - imp[col_b]
        result_df = pd.DataFrame({
            "feature": imp.index,
            f"mean_abs_shap_{class_a}": imp[col_a].values,
            f"mean_abs_shap_{class_b}": imp[col_b].values,
            "discriminability": diff.values,
        })
        method_label = "mean |SHAP| difference (importance CSV)"

    # Sort by absolute discriminability, keep top_n
    result_df = result_df.reindex(
        result_df["discriminability"].abs().sort_values(ascending=False).index
    ).head(top_n).reset_index(drop=True)

    # Save CSV
    safe_a = class_a.replace(" ", "_")
    safe_b = class_b.replace(" ", "_")
    csv_path = model_dir / f"pairwise_shap_{safe_a}_vs_{safe_b}.csv"
    result_df.to_csv(csv_path, index=False)
    logger.info(f"Pairwise SHAP CSV saved: {csv_path}")

    # ── Plot ────────────────────────────────────────────────────────────────
    n = len(result_df)
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.38 + 2)))
    y_pos = np.arange(n)
    vals = result_df["discriminability"].values
    feats = result_df["feature"].values

    # Colour by feature family
    bar_colours = [
        "#185FA5" if get_family(f) == "spectral" else
        "#0F6E56" if get_family(f) == "glcm" else
        "#854F0B" if get_family(f) == "sdiv" else
        "#888780"
        for f in feats
    ]

    ax.barh(y_pos, vals, color=bar_colours, height=0.6,
            edgecolor="white", linewidth=0.4, zorder=2)

    ax.axvline(0, color="black", linewidth=0.8, zorder=3)

    # Value labels
    for i, v in enumerate(vals):
        sign = "+" if v >= 0 else ""
        ha = "left" if v >= 0 else "right"
        ax.text(v + (0.002 * np.nanmax(np.abs(vals))), i,
                f"{sign}{v:.4f}", va="center", ha=ha, fontsize=7.5,
                color="black")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=8.5)
    ax.yaxis.set_tick_params(length=0)
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    # Direction labels
    x_range = np.nanmax(np.abs(vals))
    ax.text(x_range * 0.5, -1.3, f"{class_a} →",
            ha="center", fontsize=9, color="#E8334A",
            transform=ax.get_xaxis_transform())
    ax.text(-x_range * 0.5, -1.3, f"← {class_b}",
            ha="center", fontsize=9, color="#3B82C4",
            transform=ax.get_xaxis_transform())

    # Family legend
    import matplotlib.patches as _mp
    legend_handles = [
        _mp.Patch(color="#185FA5", label="Spectral"),
        _mp.Patch(color="#0F6E56", label="GLCM"),
        _mp.Patch(color="#854F0B", label="Spectral diversity"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.8,
              loc="lower right", title="Feature family", title_fontsize=8)

    ax.set(
        xlabel=f"Discriminability ({method_label})",
        title=(
            f"Pairwise SHAP — {class_a}  vs  {class_b}\n"
            f"Spectra {spectra}, Level {level} | top {top_n} most discriminating features"
        ),
        **_ax_kwargs(kwargs),
    )
    fig.tight_layout()

    if out_path is None:
        out_path = model_dir / f"pairwise_shap_{safe_a}_vs_{safe_b}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Pairwise SHAP plot saved: {out_path}")

    return result_df


# ---------------------------------------------------------------------------
# CV vs held-out accuracy paired chart
# ---------------------------------------------------------------------------

def cv_vs_held_out_chart(
        output_dir: Path,
        spectra_types: list[str] = None,
        levels: list[int] = None,
        weighted: bool = True,
        held_out_pattern: str = "held_out",
        cv_metric: str = "pixel_accuracy",
        out_path: Path | None = None,
        **kwargs,
) -> pd.DataFrame | None:
    """
    Paired dot chart comparing three accuracy estimates per model:
        • CV mean pixel accuracy ± 1 std (across 5 folds) — or macro_f1 if specified
        • Single-model test macro F1 (pixel-level 1% test set)
        • Held-out ROI mean prop_correct ± 1 std (novel ROIs, most independent)

    Each model (spectra × level combination) appears as a row. The three
    estimates are plotted as dots on a shared x-axis [0, 1], connected by a
    light horizontal line so the eye can follow the generalisation drop from
    test → held-out.

    Models are grouped by level (separated by horizontal rules) and ordered
    A→B→C→D within each level.

    Data sources
    ────────────
    CV        : outputs/spectra_{X}/level_{N}_cv[_unweighted]/summary/metrics_summary.csv
                  rows index = ["mean","std","min","max"], column = cv_metric
    Test F1   : outputs/spectra_{X}/level_{N}[_unweighted]/metrics.json  → macro_f1
    Held-out  : outputs/maps/{dir matching held_out_pattern}/  → prop_correct NetCDF attr
                  dirs also filtered to contain spectra label e.g. "spectraA"

    Args:
        output_dir:       Root output directory (e.g. Path('outputs')).
        spectra_types:    List of spectra labels. Defaults to ['A','B','C','D'].
        levels:           List of levels to include. Defaults to [1, 2, 3, 4].
        weighted:         Whether to use weighted model outputs.
        held_out_pattern: Substring to match held-out map subdirectory names.
                          Defaults to 'held_out' (matches both old 'turf_held_out'
                          and new 'held_out_*' naming conventions).
        cv_metric:        Which metric to read from metrics_summary.csv for the CV
                          estimate. Use 'pixel_accuracy' (default) for a fair
                          comparison with held-out prop_correct, or 'macro_f1' to
                          show the class-averaged metric.
        out_path:         Output PNG path. Auto-generated if None.
        **kwargs:         Forwarded to ax.set() via _ax_kwargs().

    Returns:
        DataFrame of all collected values (useful for further analysis), or
        None if no data was found.
    """
    import json
    try:
        import xarray as xr
        _has_xr = True
    except ImportError:
        _has_xr = False
        logger.warning("xarray not available — held-out prop_correct cannot be loaded.")

    spectra_types = spectra_types or ["A", "B", "C", "D"]
    levels = levels or [1, 2, 3]
    suffix = "" if weighted else "_unweighted"
    maps_dir = output_dir / "maps"

    # ── helpers ──────────────────────────────────────────────────────────────

    def _load_cv_metric(spectra: str, level: int, metric: str = "pixel_accuracy") -> tuple[float, float] | tuple[
        None, None]:
        """Returns (mean, std) for the requested metric from CV summary, or (None, None)."""
        p = (output_dir / f"spectra_{spectra}"
             / f"level_{level}_cv{suffix}" / "summary" / "metrics_summary.csv")
        if not p.exists():
            logger.debug(f"CV summary missing: {p}")
            return None, None
        df = pd.read_csv(p, index_col=0)
        if metric not in df.columns:
            logger.warning(f"Metric '{metric}' not in {p} — available: {list(df.columns)}")
            return None, None
        try:
            return float(df.loc["mean", metric]), float(df.loc["std", metric])
        except KeyError:
            return None, None

    def _load_cv_f1(spectra: str, level: int) -> tuple[float, float] | tuple[None, None]:
        """Returns (mean_f1, std_f1) from CV summary, or (None, None)."""
        p = (output_dir / f"spectra_{spectra}"
             / f"level_{level}_cv{suffix}" / "summary" / "metrics_summary.csv")
        if not p.exists():
            logger.debug(f"CV summary missing: {p}")
            return None, None
        df = pd.read_csv(p, index_col=0)
        try:
            return float(df.loc["mean", "macro_f1"]), float(df.loc["std", "macro_f1"])
        except KeyError:
            logger.warning(f"macro_f1 not found in {p}")
            return None, None

    def _load_test_f1(spectra: str, level: int) -> float | None:
        """Returns single-model test macro_f1 from metrics.json."""
        p = output_dir / f"spectra_{spectra}" / f"level_{level}{suffix}" / "metrics.json"
        if not p.exists():
            logger.debug(f"metrics.json missing: {p}")
            return None
        with open(p) as f:
            m = json.load(f)
        val = m.get("macro_f1")
        return float(val) if val is not None else None

    def _load_held_out(spectra: str, level: int) -> tuple[float, float, int] | tuple[None, None, None]:
        """
        Reads prop_correct from held-out NetCDF map files for a given spectra
        and level. Files are filtered by level to avoid returning the same
        averaged value across all levels.
        Returns (mean, std, n_rois) or (None, None, None).
        """
        if not _has_xr or not maps_dir.exists():
            return None, None, None

        spectra_tag = f"spectra{spectra}".lower()
        candidates = [
            d for d in maps_dir.iterdir()
            if d.is_dir()
               and held_out_pattern.lower() in d.name.lower()
               and spectra_tag in d.name.lower()
        ]
        if not candidates:
            logger.debug(f"No held-out map dir found for spectra {spectra} "
                         f"(pattern='{held_out_pattern}', spectra_tag='{spectra_tag}')")
            return None, None, None

        # Use the first matching directory (there should only be one per spectra)
        held_dir = candidates[0]
        prop_vals = []
        for nc_path in held_dir.glob(f"roi_*_spectra{spectra}_L{level}.nc"):
            try:
                ds = xr.open_dataset(nc_path, engine="netcdf4")
                pv = ds.attrs.get("prop_correct")
                ds.close()
                if pv is not None and not (isinstance(pv, float) and np.isnan(pv)):
                    prop_vals.append(float(pv))
            except Exception as e:
                logger.debug(f"Could not read {nc_path}: {e}")

        if not prop_vals:
            logger.warning(f"No valid prop_correct values found in {held_dir} for L{level}")
            return None, None, None

        arr = np.array(prop_vals)
        return float(arr.mean()), float(arr.std()), len(arr)

    # ── collect data ─────────────────────────────────────────────────────────

    records = []

    for level in levels:
        for spectra in spectra_types:
            cv_mean, cv_std = _load_cv_metric(spectra, level, cv_metric)
            test_f1 = _load_test_f1(spectra, level)

            ho_mean, ho_std, ho_n = _load_held_out(spectra, level)

            # Skip rows where we have nothing at all
            if cv_mean is None and test_f1 is None and ho_mean is None:
                logger.warning(f"No data for spectra={spectra} level={level} — skipping.")
                continue

            records.append({
                "spectra": spectra,
                "level": level,
                "label": f"Spectra {spectra} L{level}",
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "test_f1": test_f1,
                "ho_mean": ho_mean,
                "ho_std": ho_std,
                "ho_n": ho_n,
            })

    if not records:
        logger.error("No data found — check output directory structure and held_out_pattern.")
        return None

    df = pd.DataFrame(records)

    # ── plot ─────────────────────────────────────────────────────────────────

    cv_metric_label = cv_metric.replace("_", " ")
    n_rows = len(df)
    fig_h = max(5, n_rows * 0.52 + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    # Colour palette — consistent with pipeline conventions
    COL_CV = "#185FA5"  # blue  (CV)
    COL_TEST = "#0F6E56"  # teal  (single-model test)
    COL_HO = "#993C1D"  # coral/red (held-out — most independent)

    y_positions = np.arange(n_rows)

    for i, row in df.iterrows():
        y = n_rows - 1 - list(df.index).index(i)  # top-to-bottom ordering

        # Collect x-values that are not None for the connecting line
        x_vals = [v for v in [row["cv_mean"], row["test_f1"], row["ho_mean"]]
                  if v is not None]
        if len(x_vals) > 1:
            ax.plot([min(x_vals), max(x_vals)], [y, y],
                    color="#cccccc", linewidth=1.2, zorder=1, solid_capstyle="round")

        # CV dot + error bar
        if row["cv_mean"] is not None:
            ax.errorbar(
                row["cv_mean"], y,
                xerr=row["cv_std"] if row["cv_std"] is not None else 0,
                fmt="o", color=COL_CV, markersize=7, capsize=3,
                linewidth=1.2, zorder=3, label=f"CV mean {cv_metric_label} ± std" if i == df.index[0] else "",
            )

        # Single-model test F1 dot
        if row["test_f1"] is not None:
            ax.scatter(
                row["test_f1"], y,
                marker="D", color=COL_TEST, s=45, zorder=3,
                label="Single-model test F1" if i == df.index[0] else "",
            )

        # Held-out dot + error bar
        if row["ho_mean"] is not None:
            n_label = f" (n={int(row['ho_n'])})" if row["ho_n"] is not None else ""
            ax.errorbar(
                row["ho_mean"], y,
                xerr=row["ho_std"] if row["ho_std"] is not None else 0,
                fmt="s", color=COL_HO, markersize=7, capsize=3,
                linewidth=1.2, zorder=3,
                label=f"Held-out ROI mean ± std{n_label}" if i == df.index[0] else "",
            )

    # Y-axis labels
    y_labels = [df.loc[i, "label"] for i in reversed(df.index)]
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.yaxis.set_tick_params(length=0)

    # Horizontal separators between levels
    level_changes = []
    for idx in range(1, n_rows):
        if df.iloc[n_rows - 1 - idx]["level"] != df.iloc[n_rows - idx]["level"]:
            level_changes.append(idx - 0.5)
    for yc in level_changes:
        ax.axhline(yc, color="#dddddd", linewidth=1.0, zorder=0)

    # Level group labels on the right margin
    for level in levels:
        level_rows = [n_rows - 1 - list(df.index).index(i)
                      for i, row in df.iterrows() if row["level"] == level]
        if level_rows:
            y_mid = np.mean(level_rows)
            ax.text(1.01, y_mid / (n_rows - 1) if n_rows > 1 else 0.5,
                    f"L{level}", transform=ax.transAxes,
                    ha="left", va="center", fontsize=9, fontweight=500, color="#555555")

    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax.set(
        xlabel="Accuracy / F1",
        title=(
            f"CV {cv_metric_label} vs test F1 vs held-out ROI accuracy\n"
            f"{'Weighted' if weighted else 'Unweighted'} models — "
            f"Spectra {', '.join(spectra_types)}"
        ),
        **_ax_kwargs(kwargs),
    )

    ax.legend(
        loc="best", fontsize=8.5, framealpha=0.85,
        title="Estimate type", title_fontsize=8.5,
    )

    fig.tight_layout()

    if out_path is None:
        lvl_str = "_".join(str(l) for l in levels)
        out_path = output_dir / f"cv_vs_held_out_L{lvl_str}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"CV vs held-out chart saved: {out_path}")

    return df


# ---------------------------------------------------------------------------
# CV fold stability strip  (idea #3)
# ---------------------------------------------------------------------------

def cv_fold_stability_strip(
        output_dir: Path,
        spectra_types: list[str] = None,
        levels: list[int] = None,
        weighted: bool = True,
        held_out_pattern: str = "held_out",
        cv_metric: str = "pixel_accuracy",
        out_path: Path | None = None,
        **kwargs,
) -> None:
    """
    For each model (spectra × level), shows the 5 individual CV fold metric
    scores as dots with a mean line, and overlays the held-out ROI mean accuracy
    as a dashed horizontal line.

    Layout: one subplot per level (rows), one group of dots per spectra (columns
    within each subplot). This makes fold variance and the CV-vs-held-out gap
    immediately comparable across all 16 models.

    Data sources
    ────────────
    Fold F1s : outputs/spectra_{X}/level_{N}_cv[_unweighted]/summary/metrics_per_fold.csv
               column = "macro_f1"
    Held-out : outputs/maps/{held_out_pattern}*spectra{X}*/ → prop_correct NetCDF attrs

    Args:
        output_dir:       Root output directory.
        spectra_types:    Defaults to ['A','B','C','D'].
        levels:           Defaults to [1, 2, 3, 4].
        weighted:         Whether to use weighted model outputs.
        held_out_pattern: Substring to match held-out map directory names.
        out_path:         Output PNG path. Auto-generated if None.
    """
    try:
        import xarray as xr
        _has_xr = True
    except ImportError:
        _has_xr = False

    spectra_types = spectra_types or ["A", "B", "C", "D"]
    levels = levels or [1, 2, 3, 4]
    suffix = "" if weighted else "_unweighted"
    maps_dir = output_dir / "maps"

    # ── held-out loader (same logic as cv_vs_held_out_chart) ─────────────────
    def _held_out_mean(spectra: str, level: int) -> float | None:
        if not _has_xr or not maps_dir.exists():
            return None
        tag = f"spectra{spectra}".lower()
        candidates = [d for d in maps_dir.iterdir()
                      if d.is_dir()
                      and held_out_pattern.lower() in d.name.lower()
                      and tag in d.name.lower()]
        if not candidates:
            return None
        vals = []
        for nc in candidates[0].glob(f"roi_*_spectra{spectra}_L{level}.nc"):
            try:
                ds = xr.open_dataset(nc, engine="netcdf4")
                pv = ds.attrs.get("prop_correct")
                ds.close()
                if pv is not None and not (isinstance(pv, float) and np.isnan(pv)):
                    vals.append(float(pv))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else None

    # ── layout ───────────────────────────────────────────────────────────────
    n_levels = len(levels)
    n_spectra = len(spectra_types)

    # Shared y limits per level (computed after loading)
    fig, axes = plt.subplots(
        n_levels, 1,
        figsize=(max(7, n_spectra * 2.2 + 1.5), n_levels * 3.2),
        sharex=False,
    )
    if n_levels == 1:
        axes = [axes]

    COL_FOLD = "#185FA5"
    COL_MEAN = "#0F6E56"
    COL_HO = "#993C1D"

    for ax_idx, (level, ax) in enumerate(zip(levels, axes)):
        x_ticks = []
        x_labels = []

        all_vals = []  # for y-range

        for sp_idx, spectra in enumerate(spectra_types):
            x_centre = sp_idx * 1.0
            p = (output_dir / f"spectra_{spectra}"
                 / f"level_{level}_cv{suffix}" / "summary" / "metrics_per_fold.csv")

            if not p.exists():
                logger.debug(f"Missing: {p}")
                x_ticks.append(x_centre)
                x_labels.append(f"Spectra {spectra}")
                continue

            fold_df = pd.read_csv(p)
            if cv_metric not in fold_df.columns:
                logger.warning(f"'{cv_metric}' not in {p} — falling back to macro_f1")
                cv_metric = "macro_f1"
            fold_f1s = fold_df[cv_metric].dropna().values
            mean_f1 = fold_f1s.mean()
            all_vals.extend(fold_f1s.tolist())

            # Jittered fold dots
            jitter = np.linspace(-0.12, 0.12, len(fold_f1s))
            ax.scatter(
                np.full(len(fold_f1s), x_centre) + jitter,
                fold_f1s,
                color=COL_FOLD, s=30, alpha=0.75, zorder=3,
                label="Fold F1" if ax_idx == 0 and sp_idx == 0 else "",
            )

            # Mean line
            ax.hlines(
                mean_f1, x_centre - 0.2, x_centre + 0.2,
                colors=COL_MEAN, linewidth=2.0, zorder=4,
                label="CV mean" if ax_idx == 0 and sp_idx == 0 else "",
            )

            # Held-out dashed line
            ho = _held_out_mean(spectra, level)
            if ho is not None:
                ax.hlines(
                    ho, x_centre - 0.2, x_centre + 0.2,
                    colors=COL_HO, linewidth=1.8, linestyle="--", zorder=4,
                    label="Held-out mean" if ax_idx == 0 and sp_idx == 0 else "",
                )
                all_vals.append(ho)

            x_ticks.append(x_centre)
            x_labels.append(f"Spectra {spectra}")

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlim(-0.5, n_spectra - 0.5)
        ax.set_ylabel(cv_metric.replace("_", " ").title(), fontsize=8)
        ax.set_title(f"Level {level}", fontsize=9, fontweight=500, pad=4)
        ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        # Y-range: pad around actual values; always show 0 context for L1/L4
        if all_vals:
            lo = max(0, min(all_vals) - 0.05)
            hi = min(1.02, max(all_vals) + 0.05)
            ax.set_ylim(lo, hi)

    # Single shared legend on first subplot
    handles, labels_ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels_, fontsize=8, framealpha=0.85,
                   loc="best", title="Estimate", title_fontsize=8)

    cv_metric_label = cv_metric.replace("_", " ")
    fig.suptitle(
        f"CV fold stability ({cv_metric_label}) vs held-out ROI accuracy\n"
        f"{'Weighted' if weighted else 'Unweighted'} — "
        f"Spectra {', '.join(spectra_types)}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()

    if out_path is None:
        lvl_str = "_".join(str(l) for l in levels)
        out_path = output_dir / f"cv_fold_stability_L{lvl_str}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"CV fold stability strip saved: {out_path}")


# ---------------------------------------------------------------------------
# Δ (CV F1 − held-out accuracy) heatmap  (idea #4)
# ---------------------------------------------------------------------------

def cv_held_out_delta_heatmap(
        output_dir: Path,
        spectra_types: list[str] = None,
        levels: list[int] = None,
        weighted: bool = True,
        held_out_pattern: str = "held_out",
        cv_metric: str = "pixel_accuracy",
        out_path: Path | None = None,
        **kwargs,
) -> pd.DataFrame | None:
    """
    16-cell heatmap (4 spectra × 4 levels) showing Δ = CV mean F1 − held-out
    mean prop_correct. Diverging colourmap centred at 0: blue = CV < held-out
    (rare/good), red = CV > held-out (held-out harder).

    Also annotates each cell with the raw CV mean and held-out mean for
    full information density without needing a separate table.

    Data sources: same as cv_vs_held_out_chart().

    Args:
        output_dir:       Root output directory.
        spectra_types:    Defaults to ['A','B','C','D'].
        levels:           Defaults to [1, 2, 3, 4].
        weighted:         Whether to use weighted model outputs.
        held_out_pattern: Substring to match held-out map directory names.
        out_path:         Output PNG path. Auto-generated if None.

    Returns:
        DataFrame with columns [spectra, level, cv_mean, ho_mean, delta].
    """
    try:
        import xarray as xr
        _has_xr = True
    except ImportError:
        _has_xr = False

    spectra_types = spectra_types or ["A", "B", "C", "D"]
    levels = levels or [1, 2, 3, 4]
    suffix = "" if weighted else "_unweighted"
    maps_dir = output_dir / "maps"

    def _cv_mean(spectra: str, level: int) -> float | None:
        p = (output_dir / f"spectra_{spectra}"
             / f"level_{level}_cv{suffix}" / "summary" / "metrics_summary.csv")
        if not p.exists():
            return None
        _df = pd.read_csv(p, index_col=0)
        _metric = cv_metric if cv_metric in _df.columns else "macro_f1"
        if _metric != cv_metric:
            logger.warning(f"'{cv_metric}' not in {p} — falling back to macro_f1")
        try:
            return float(_df.loc["mean", _metric])
        except KeyError:
            return None

    def _held_out_mean(spectra: str, level: int) -> float | None:
        if not _has_xr or not maps_dir.exists():
            return None
        tag = f"spectra{spectra}".lower()
        candidates = [d for d in maps_dir.iterdir()
                      if d.is_dir()
                      and held_out_pattern.lower() in d.name.lower()
                      and tag in d.name.lower()]
        if not candidates:
            return None
        vals = []
        for nc in candidates[0].glob(f"roi_*_spectra{spectra}_L{level}.nc"):
            try:
                ds = xr.open_dataset(nc, engine="netcdf4")
                pv = ds.attrs.get("prop_correct")
                ds.close()
                if pv is not None and not (isinstance(pv, float) and np.isnan(pv)):
                    vals.append(float(pv))
            except Exception:
                pass
        return float(np.mean(vals)) if vals else None

    # ── collect ───────────────────────────────────────────────────────────────

    records = []
    for level in levels:
        for spectra in spectra_types:
            cv = _cv_mean(spectra, level)
            ho = _held_out_mean(spectra, level)
            delta = (cv - ho) if (cv is not None and ho is not None) else None
            records.append({"spectra": spectra, "level": level,
                            "cv_mean": cv, "ho_mean": ho, "delta": delta})

    result_df = pd.DataFrame(records)

    # ── build 2D arrays for heatmap (rows = levels, cols = spectra) ──────────
    delta_grid = np.full((len(levels), len(spectra_types)), np.nan)
    cv_grid = np.full_like(delta_grid, np.nan)
    ho_grid = np.full_like(delta_grid, np.nan)

    for i, level in enumerate(levels):
        for j, spectra in enumerate(spectra_types):
            row = result_df[(result_df.level == level) & (result_df.spectra == spectra)]
            if not row.empty:
                delta_grid[i, j] = row.iloc[0]["delta"]
                cv_grid[i, j] = row.iloc[0]["cv_mean"]
                ho_grid[i, j] = row.iloc[0]["ho_mean"]

    if np.all(np.isnan(delta_grid)):
        logger.error("No delta values computable — check CV summary and held-out map paths.")
        return None

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(spectra_types) * 1.8 + 1.5),
                                    max(4, len(levels) * 1.5 + 1.5)))

    abs_max = np.nanmax(np.abs(delta_grid))
    im = ax.imshow(
        delta_grid, cmap="RdBu_r",
        vmin=-abs_max, vmax=abs_max,
        aspect="auto",
    )

    # Cell annotations: Δ on top line, CV / HO on second line
    for i in range(len(levels)):
        for j in range(len(spectra_types)):
            d = delta_grid[i, j]
            cv = cv_grid[i, j]
            ho = ho_grid[i, j]
            if np.isnan(d):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#888888")
                continue
            # Text colour: white on saturated cells, black in centre
            brightness = abs(d) / (abs_max + 1e-9)
            txt_col = "white" if brightness > 0.55 else "black"
            sign = "+" if d >= 0 else ""
            ax.text(j, i - 0.12, f"Δ {sign}{d:.3f}",
                    ha="center", va="center", fontsize=9,
                    fontweight=600, color=txt_col)
            cv_str = f"{cv:.3f}" if not np.isnan(cv) else "—"
            ho_str = f"{ho:.3f}" if not np.isnan(ho) else "—"
            ax.text(j, i + 0.22, f"CV {cv_str}  HO {ho_str}",
                    ha="center", va="center", fontsize=7, color=txt_col, alpha=0.85)

    ax.set_xticks(np.arange(len(spectra_types)))
    ax.set_xticklabels([f"Spectra {s}" for s in spectra_types], fontsize=10)
    ax.set_yticks(np.arange(len(levels)))
    ax.set_yticklabels([f"Level {l}" for l in levels], fontsize=10)

    cv_metric_label = cv_metric.replace("_", " ")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f"Δ = CV {cv_metric_label} − held-out mean accuracy", fontsize=8)
    cbar.ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title(
        f"Generalisation gap: CV {cv_metric_label} minus held-out ROI accuracy\n"
        f"{'Weighted' if weighted else 'Unweighted'} — "
        f"positive (red) = CV over-estimates real-world performance",
        fontsize=9, pad=10,
        **_ax_kwargs(kwargs),
    )
    ax.tick_params(length=0)

    fig.tight_layout()

    if out_path is None:
        lvl_str = "_".join(str(l) for l in levels)
        out_path = output_dir / f"cv_held_out_delta_heatmap_L{lvl_str}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Delta heatmap saved: {out_path}")

    return result_df


# ---------------------------------------------------------------------------
# Level-split paired chart  (idea #5) — L1–3 vs L4 separate panels
# ---------------------------------------------------------------------------

def cv_vs_held_out_split_chart(
        output_dir: Path,
        spectra_types: list[str] = None,
        weighted: bool = True,
        held_out_pattern: str = "held_out",
        out_path: Path | None = None,
        **kwargs,
) -> None:
    """
    Two-panel version of cv_vs_held_out_chart() that treats Level 4 separately.

    Left panel  — Levels 1, 2, 3: shared x-axis [0, 1] so the accuracy drop
                  across levels is directly comparable.
    Right panel — Level 4 only: same x-axis [0, 1] but annotated separately
                  because the near-zero CV F1 and ~0.88 held-out accuracy are
                  qualitatively different from the coarser levels (see
                  entropy-comparison.md: +0.19–0.21 entropy gap on held-out).

    A vertical grey annotation band in the Level 4 panel highlights the
    CV F1 ≈ 0 region to distinguish it visually from absence-of-data.

    Uses the same data sources as cv_vs_held_out_chart().

    Args:
        output_dir:       Root output directory.
        spectra_types:    Defaults to ['A','B','C','D'].
        weighted:         Whether to use weighted model outputs.
        held_out_pattern: Substring to match held-out map directory names.
        out_path:         Output PNG path. Auto-generated if None.
    """
    # Delegate data collection to cv_vs_held_out_chart() with all levels
    df = cv_vs_held_out_chart(
        output_dir=output_dir,
        spectra_types=spectra_types or ["A", "B", "C", "D"],
        levels=[1, 2, 3, 4],
        weighted=weighted,
        held_out_pattern=held_out_pattern,
        out_path=Path(output_dir) / "_tmp_cv_held_out_internal.png",  # suppress auto-save
    )

    # Remove the temp file silently
    tmp = Path(output_dir) / "_tmp_cv_held_out_internal.png"
    if tmp.exists():
        tmp.unlink()

    if df is None or df.empty:
        logger.error("No data available for split chart.")
        return

    df_main = df[df["level"].isin([1, 2, 3])].copy()
    df_l4 = df[df["level"] == 4].copy()

    spectra_types = spectra_types or ["A", "B", "C", "D"]

    COL_CV = "#185FA5"
    COL_TEST = "#0F6E56"
    COL_HO = "#993C1D"

    def _draw_panel(ax, panel_df, title_suffix: str, show_legend: bool, note: str | None = None):
        """Draws one panel of the split chart onto ax."""
        n = len(panel_df)
        if n == 0:
            ax.set_visible(False)
            return

        for plot_rank, (_, row) in enumerate(panel_df.iterrows()):
            y = n - 1 - plot_rank

            x_vals = [v for v in [row["cv_mean"], row["test_f1"], row["ho_mean"]]
                      if v is not None]
            if len(x_vals) > 1:
                ax.plot([min(x_vals), max(x_vals)], [y, y],
                        color="#cccccc", linewidth=1.2, zorder=1)

            first = plot_rank == 0
            if row["cv_mean"] is not None:
                ax.errorbar(
                    row["cv_mean"], y,
                    xerr=row["cv_std"] if row["cv_std"] is not None else 0,
                    fmt="o", color=COL_CV, markersize=7, capsize=3,
                    linewidth=1.2, zorder=3,
                    label="CV mean F1 ± std" if first and show_legend else "",
                )
            if row["test_f1"] is not None:
                ax.scatter(
                    row["test_f1"], y, marker="D",
                    color=COL_TEST, s=45, zorder=3,
                    label="Single-model test F1" if first and show_legend else "",
                )
            if row["ho_mean"] is not None:
                ax.errorbar(
                    row["ho_mean"], y,
                    xerr=row["ho_std"] if row["ho_std"] is not None else 0,
                    fmt="s", color=COL_HO, markersize=7, capsize=3,
                    linewidth=1.2, zorder=3,
                    label="Held-out ROI mean ± std" if first and show_legend else "",
                )

        y_labels = list(reversed([row["label"] for _, row in panel_df.iterrows()]))
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.yaxis.set_tick_params(length=0)

        # Level separators
        prev_level = None
        for plot_rank, (_, row) in enumerate(panel_df.iterrows()):
            y = n - 1 - plot_rank
            if prev_level is not None and row["level"] != prev_level:
                ax.axhline(y + 0.5, color="#dddddd", linewidth=1.0, zorder=0)
            prev_level = row["level"]

        # Level labels on right margin
        for level in panel_df["level"].unique():
            level_rows = [n - 1 - plot_rank
                          for plot_rank, (_, row) in enumerate(panel_df.iterrows())
                          if row["level"] == level]
            if level_rows:
                y_mid = np.mean(level_rows)
                ax.text(1.01, y_mid / max(n - 1, 1),
                        f"L{level}", transform=ax.transAxes,
                        ha="left", va="center", fontsize=9,
                        fontweight=500, color="#555555")

        ax.set_xlim(0, 1.05)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.set_xlabel("Accuracy / F1", fontsize=9)
        ax.set_title(title_suffix, fontsize=9, fontweight=500, pad=6)
        ax.grid(True, axis="x", alpha=0.2, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)

        # Optional annotation text (used for Level 4 note)
        if note:
            ax.text(0.02, 0.02, note, transform=ax.transAxes,
                    fontsize=7.5, color="#666666", va="bottom",
                    style="italic", wrap=True)

        if show_legend:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85,
                      title="Estimate type", title_fontsize=8)

    n_main = len(df_main)
    n_l4 = len(df_l4)
    h_main = max(3, n_main * 0.52 + 1.5)
    h_l4 = max(2, n_l4 * 0.52 + 1.5)

    fig, (ax_main, ax_l4) = plt.subplots(
        1, 2,
        figsize=(18, max(h_main, h_l4) + 1.0),
        gridspec_kw={"width_ratios": [3, 1.4], "wspace": 0.35},
    )

    _draw_panel(ax_main, df_main,
                title_suffix="Levels 1–3: class-level generalisation",
                show_legend=True)

    _draw_panel(ax_l4, df_l4,
                title_suffix="Level 4: ROI-identity classification",
                show_legend=False,
                note=(
                    "CV F1 ≈ 0: each fold withholds\n"
                    "~24 ROIs with unseen class labels.\n"
                    "Held-out ≈ 0.88: pixel accuracy\n"
                    "on novel turf ROIs is maintained\n"
                    "despite near-zero multiclass F1."
                ))

    # Shade the near-zero CV region in the L4 panel
    if not df_l4.empty:
        cv_vals = df_l4["cv_mean"].dropna()
        if len(cv_vals):
            shade_max = max(cv_vals.max() + 0.04, 0.08)
            ax_l4.axvspan(0, shade_max, color="#185FA5", alpha=0.06, zorder=0,
                          label="CV F1 ≈ 0 zone")

    fig.suptitle(
        f"CV F1 vs test F1 vs held-out ROI accuracy — "
        f"{'Weighted' if weighted else 'Unweighted'} models\n"
        f"Spectra {', '.join(spectra_types)}",
        fontsize=11, y=1.01,
    )

    if out_path is None:
        out_path = output_dir / "cv_vs_held_out_split.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Split chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Family share stacked bar  (cross-model, Chart 1)
# ---------------------------------------------------------------------------

def load_shap_by_family(
        output_dir: Path,
        spectra: str,
        level: int,
        weighted: bool = True,
) -> "pd.DataFrame | None":
    """Loads shap_by_family.csv for one model. Returns None if missing."""
    suffix = "" if weighted else "_unweighted"
    path = output_dir / f"spectra_{spectra}" / f"level_{level}{suffix}" / "shap_by_family.csv"
    if not path.exists():
        logger.warning(f"Missing: {path}")
        return None
    df = pd.read_csv(path, index_col=0)
    logger.info(f"Loaded: {path}")
    return df


def load_scale_response(
        output_dir: Path,
        spectra: str,
        level: int,
        weighted: bool = True,
) -> "pd.DataFrame | None":
    """Loads scale_response.csv for one model. Returns None if missing."""
    suffix = "" if weighted else "_unweighted"
    path = output_dir / f"spectra_{spectra}" / f"level_{level}{suffix}" / "scale_response.csv"
    if not path.exists():
        logger.warning(f"Missing: {path}")
        return None
    df = pd.read_csv(path)
    df["spectra"] = spectra
    df["level"] = level
    logger.info(f"Loaded: {path}")
    return df


def family_share_bar(
        output_dir: Path,
        levels: list[int] = None,
        spectra_types: list[str] = None,
        weighted: bool = True,
        shap_col: str = "mean_abs_shap_global",
        out_path: "Path | None" = None,
        **kwargs,
) -> None:
    """
    Stacked bar chart showing each feature family's fractional share of total
    mean |SHAP| across all 16 models (4 spectra × 4 levels).

    Bars are grouped by level with dashed separators and level headers above.
    Families: spectral (blue), glcm (teal), sdiv (amber).
    A percentage label is printed inside each segment when ≥ 6%.

    Reads shap_by_family.csv from each model directory. The CSV must contain
    a column matching `shap_col` and a 'family' column (or family in the index).

    Args:
        output_dir:    Root output directory (e.g. Path('outputs')).
        levels:        Hierarchy levels to include. Defaults to [3, 2, 1].
        spectra_types: List of spectra labels. Defaults to ['A','B','C','D'].
        weighted:      Use weighted model outputs.
        shap_col:      SHAP column to aggregate by (default: mean_abs_shap_global).
        out_path:      Output PNG path. Auto-generated if None.
    """
    levels = levels or [3, 2, 1]
    spectra_types = spectra_types or ["A", "B", "C", "D"]

    # ── collect ───────────────────────────────────────────────────────────────
    records = []
    for level in levels:
        for sp in spectra_types:
            df = load_shap_by_family(output_dir, sp, level, weighted)
            if df is None:
                continue
            if "family" in df.columns:
                df = df.set_index("family")
            if shap_col not in df.columns:
                logger.warning(
                    f"Column '{shap_col}' not in shap_by_family for {sp} L{level}; "
                    f"available: {list(df.columns)}"
                )
                continue
            for family, row in df.iterrows():
                records.append({
                    "spectra": sp,
                    "level": level,
                    "family": str(family),
                    "shap": float(row[shap_col]),
                })

    if not records:
        logger.error("No shap_by_family data found — aborting family_share_bar.")
        return

    long = pd.DataFrame(records)

    # fractional share within each model
    totals = long.groupby(["spectra", "level"])["shap"].transform("sum")
    long["share"] = long["shap"] / totals.replace(0, np.nan)

    # pivot → (level, spectra) × family
    wide = long.pivot_table(
        index=["level", "spectra"], columns="family", values="share", aggfunc="mean"
    ).fillna(0)

    family_order = [f for f in ["spectral", "glcm", "sdiv"] if f in wide.columns]
    family_order += [c for c in wide.columns if c not in family_order]
    wide = wide[family_order]

    # ── build x positions with inter-level gap ────────────────────────────────
    bar_labels = []
    x_positions = []
    level_midpoints = {}
    x = 0
    for level in levels:
        group_start = x
        for sp in spectra_types:
            if (level, sp) in wide.index:
                bar_labels.append(sp)
                x_positions.append(x)
                x += 1
        group_end = x - 1
        level_midpoints[level] = (group_start + group_end) / 2
        x += 0.9  # inter-group gap

    ordered_index = [
        (level, sp)
        for level in levels for sp in spectra_types
        if (level, sp) in wide.index
    ]
    wide = wide.loc[ordered_index]

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(9, len(x_positions) * 0.78 + 2.5), 5.0))

    bottoms = np.zeros(len(wide))
    for family in family_order:
        values = wide[family].values
        colour = FAMILY_COLOURS.get(family, "#888780")
        label = FAMILY_LABELS.get(family, family.capitalize())
        ax.bar(x_positions, values, bottom=bottoms,
               color=colour, label=label,
               width=0.72, edgecolor="white", linewidth=0.5, zorder=2)
        for xi, (v, b) in enumerate(zip(values, bottoms)):
            if v >= 0.06:
                ax.text(x_positions[xi], b + v / 2, f"{v:.0%}",
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight=500)
        bottoms += values

    # level separators (between groups, not at edges)
    x = 0
    sep_xs = []
    for level in levels:
        n_in = sum(1 for sp in spectra_types if (level, sp) in set(ordered_index))
        x += n_in
        sep_xs.append(x - 0.5 + 0.45)
        x += 0.9
    for sx in sep_xs[:-1]:
        ax.axvline(sx, color="#bbbbbb", linewidth=1.0, linestyle="--", zorder=0)

    # level headers above bars via transAxes
    x_span = max(x_positions) - min(x_positions) if len(x_positions) > 1 else 1
    for level, mid in level_midpoints.items():
        frac = (mid - min(x_positions)) / x_span if x_span else 0.5
        ax.text(frac, 1.025, f"Level {level}",
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=9, fontweight=500, color="#333333")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_xlabel("Spectra", fontsize=10)
    ax.set_ylabel("Share of mean |SHAP|", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_xlim(min(x_positions) - 0.5, max(x_positions) + 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5, zorder=0)
    ax.set(**_ax_kwargs(kwargs))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85,
              title="Feature family", title_fontsize=9)

    fig.suptitle(
        f"Feature family share of SHAP importance — "
        f"{'Weighted' if weighted else 'Unweighted'} models",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()

    if out_path is None:
        lvl_str = "_".join(str(l) for l in levels)
        out_path = output_dir / f"family_share_bar_levels{lvl_str}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Family share bar saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-feature scale response (small multiples, Chart 3)
# ---------------------------------------------------------------------------

import re as _re

_GLCM_FEAT_RE = _re.compile(r"^(energy|entropy|homogeneity|contrast)_window_(\d+)$")
_SDIV_FEAT_RE = _re.compile(r"^sdiv_(.+)_plot_(\d+)$")


def _parse_feature(col: str) -> tuple[str, str, int] | None:
    """
    Returns (family, metric, window_size) for GLCM/sdiv columns, else None.
    family  : 'glcm' | 'sdiv'
    metric  : e.g. 'contrast', 'alpha_local'
    window_size: int pixels
    """
    m = _GLCM_FEAT_RE.match(col)
    if m:
        return "glcm", m.group(1), int(m.group(2))
    m = _SDIV_FEAT_RE.match(col)
    if m:
        return "sdiv", m.group(1), int(m.group(2))
    return None


def feature_scale_strip(
        output_dir: Path,
        levels: list[int] = None,
        spectra_types: list[str] = None,
        weighted: bool = True,
        shap_col: str = "mean_abs_shap_global",
        glcm_order: list[str] | None = None,
        sdiv_order: list[str] | None = None,
        out_path: "Path | None" = None,
        **kwargs,
) -> None:
    """
    Small-multiples strip plot of scale-response curves broken out by individual
    feature (metric × window size) for GLCM and sdiv families.

    Layout: two column groups sharing a common figure.
      Left  — GLCM: 2×2 grid of subplots (energy, entropy, homogeneity, contrast).
      Right — sdiv: 2×1 column (alpha_local, beta_local).

    Each subplot:
      x-axis  — window size on a log₂ scale (labelled with actual pixel sizes).
      y-axis  — mean |SHAP| (independent per panel so peak shapes are legible).
      Dots    — individual (spectra × level) models, jittered horizontally.
      Encoding: colour = reflectance green / derivative purple;
                marker = circle (A/B) / diamond (C/D);
                opacity = level (L3 opaque → L1 faint).
      Grey IQR band + dark median line across all 16 models.

    Reads feature_importance_shap.csv from each model directory.
    Window size and metric are parsed directly from column names using the
    same regex as get_family().

    Args:
        output_dir:   Root output directory.
        levels:       Levels to include. Defaults to [3, 2, 1].
        spectra_types: Spectra to include. Defaults to ['A','B','C','D'].
        weighted:     Use weighted model outputs.
        shap_col:     SHAP column in feature_importance_shap.csv
                      (default: mean_abs_shap_global).
        glcm_order:   Display order for GLCM metrics.
                      Defaults to ['contrast','dissimilarity','homogeneity','energy'].
        sdiv_order:   Display order for sdiv metrics.
                      Defaults to ['alpha_local','beta_local'].
        out_path:     Output PNG path. Auto-generated if None.
    """
    import matplotlib.lines as mlines
    import matplotlib.gridspec as gridspec

    levels = levels or [3, 2, 1]
    spectra_types = spectra_types or ["A", "B", "C", "D"]
    glcm_order = glcm_order or ["contrast", "dissimilarity", "homogeneity", "energy"]
    sdiv_order = sdiv_order or ["alpha_local", "beta_local"]

    SP_COLOUR = {"A": "#009E73", "B": "#CC79A7", "C": "#009E73", "D": "#CC79A7"}
    SP_MARKER = {"A": "o", "B": "o", "C": "D", "D": "D"}
    LEVEL_ALPHA = {3: 1.0, 2: 0.62, 1: 0.35}

    # ── collect per-feature SHAP values from importance CSVs ─────────────────
    records = []
    for level in levels:
        for sp in spectra_types:
            imp = load_importance(output_dir, sp, level, weighted)
            if imp is None or shap_col not in imp.columns:
                continue
            for col, row in imp[[shap_col]].iterrows():
                parsed = _parse_feature(col)
                if parsed is None:
                    continue
                family, metric, ws = parsed
                records.append({
                    "spectra": sp,
                    "level": level,
                    "family": family,
                    "metric": metric,
                    "window_size": ws,
                    "shap": float(row[shap_col]),
                })

    if not records:
        logger.error("No GLCM/sdiv features found in importance CSVs — aborting.")
        return

    df = pd.DataFrame(records)

    # determine which metrics are actually present
    glcm_metrics = [m for m in glcm_order if m in df.loc[df.family == "glcm", "metric"].unique()]
    sdiv_metrics = [m for m in sdiv_order if m in df.loc[df.family == "sdiv", "metric"].unique()]
    # append any unlisted metrics at the end (defensive)
    for m in df.loc[df.family == "glcm", "metric"].unique():
        if m not in glcm_metrics:
            glcm_metrics.append(m)
    for m in df.loc[df.family == "sdiv", "metric"].unique():
        if m not in sdiv_metrics:
            sdiv_metrics.append(m)

    n_glcm = len(glcm_metrics)  # typically 4
    n_sdiv = len(sdiv_metrics)  # typically 2

    # ── figure layout via GridSpec ────────────────────────────────────────────
    # Left block: ceil(n_glcm/2) rows × 2 cols for GLCM
    # Right block: n_sdiv rows × 1 col for sdiv
    # Separated by a wider wspace
    glcm_cols = 2
    glcm_rows = (n_glcm + 1) // 2
    sdiv_rows = n_sdiv
    fig_rows = max(glcm_rows, sdiv_rows)

    fig_w = 5.5 * glcm_cols + 0.6 + 3.2  # left block + gap + right block
    fig_h = max(3.5 * fig_rows, 5.0)

    fig = plt.figure(figsize=(fig_w, fig_h))
    # outer grid: two columns (GLCM block | sdiv block), separated by extra space
    outer = gridspec.GridSpec(
        1, 2,
        width_ratios=[glcm_cols * 5.5, 3.2],
        wspace=0.12,
        figure=fig,
    )
    # inner grids
    glcm_gs = gridspec.GridSpecFromSubplotSpec(
        glcm_rows, glcm_cols, subplot_spec=outer[0], hspace=0.55, wspace=0.35
    )
    sdiv_gs = gridspec.GridSpecFromSubplotSpec(
        sdiv_rows, 1, subplot_spec=outer[1], hspace=0.55
    )

    rng = np.random.default_rng(42)

    def _draw_panel(ax, family, metric, colour_label):
        sub = df[(df.family == family) & (df.metric == metric)]
        if sub.empty:
            ax.set_visible(False)
            return

        window_sizes = sorted(sub["window_size"].unique())
        log_x = {ws: float(np.log2(ws)) for ws in window_sizes}
        xs = [log_x[ws] for ws in window_sizes]

        # IQR band + median
        agg = (
            sub.groupby("window_size")["shap"]
            .agg(q25=lambda s: s.quantile(0.25),
                 median="median",
                 q75=lambda s: s.quantile(0.75))
            .loc[window_sizes]
        )
        ax.fill_between(xs, agg["q25"].values, agg["q75"].values,
                        color="#cccccc", alpha=0.55, zorder=1)
        ax.plot(xs, agg["median"].values,
                color="#555555", linewidth=1.6, zorder=3)

        # individual model dots
        for _, row in sub.iterrows():
            sp = str(row["spectra"])
            lvl = int(row["level"])
            ws = float(row["window_size"])
            xj = log_x[ws] + rng.uniform(-0.09, 0.09)
            ax.scatter(xj, float(row["shap"]),
                       color=SP_COLOUR[sp], marker=SP_MARKER[sp],
                       s=36, alpha=LEVEL_ALPHA.get(lvl, 0.6),
                       edgecolors="none", zorder=4)

        # highlight the window with the highest median SHAP
        peak_ws = agg["median"].idxmax()
        peak_x = log_x[peak_ws]
        # y anchor: top of data range (use max of q75 + dots, not ax.get_ylim()
        # which is still the matplotlib default [0,1] at this point)
        data_top = max(sub["shap"].max(), agg["q75"].max())
        ax.axvline(peak_x, color="#999999", linewidth=0.8,
                   linestyle=":", zorder=2, alpha=0.7)
        ax.text(peak_x, data_top * 1.04,
                f" {int(peak_ws)}px", fontsize=7, color="#666666",
                va="bottom", ha="left", zorder=5)

        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(ws)) for ws in window_sizes], fontsize=7.5)
        ax.set_xlabel("Window (px)", fontsize=8)
        ax.set_ylabel("Mean |SHAP|", fontsize=8)
        ax.set_title(colour_label, fontsize=9, fontweight=500,
                     color=FAMILY_COLOURS.get(family, "#333333"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=7.5)

    # ── draw GLCM panels ──────────────────────────────────────────────────────
    for i, metric in enumerate(glcm_metrics):
        r, c = divmod(i, glcm_cols)
        ax = fig.add_subplot(glcm_gs[r, c])
        _draw_panel(ax, "glcm", metric, metric.capitalize())

    # ── draw sdiv panels ──────────────────────────────────────────────────────
    sdiv_display = {"alpha_local": "α_local", "beta_local": "β_local"}
    for i, metric in enumerate(sdiv_metrics):
        ax = fig.add_subplot(sdiv_gs[i, 0])
        _draw_panel(ax, "sdiv", metric, sdiv_display.get(metric, metric))

    # ── column group headers ──────────────────────────────────────────────────
    # Draw via fig.text using approximate normalised coordinates
    glcm_centre = outer[0].get_position(fig).x0 + outer[0].get_position(fig).width / 2
    sdiv_centre = outer[1].get_position(fig).x0 + outer[1].get_position(fig).width / 2
    top_y = outer[0].get_position(fig).y1 + 0.03
    fig.text(glcm_centre, top_y, "GLCM texture features",
             ha="center", va="bottom", fontsize=10, fontweight=600,
             color=FAMILY_COLOURS["glcm"])
    fig.text(sdiv_centre, top_y, "Spectral diversity features",
             ha="center", va="bottom", fontsize=10, fontweight=600,
             color=FAMILY_COLOURS["sdiv"])

    # ── shared legend — placed outside right edge to avoid obscuring sdiv panels ──
    legend_elements = [
        mlines.Line2D([0], [0], color="#555555", linewidth=1.6,
                      label="Median (all models)"),
        mpatches.Patch(facecolor="#cccccc", alpha=0.7, label="IQR (all models)"),
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="#009E73", markeredgecolor="none",
                      markersize=7, label="Reflectance A/B (○)"),
        mlines.Line2D([0], [0], marker="D", color="w",
                      markerfacecolor="#009E73", markeredgecolor="none",
                      markersize=7, label="Reflectance C/D (◆)"),
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="#CC79A7", markeredgecolor="none",
                      markersize=7, label="Derivative B/D (purple)"),
    ]
    for lvl, alph in sorted(LEVEL_ALPHA.items(), reverse=True):
        legend_elements.append(
            mlines.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#555555", markeredgecolor="none",
                          markersize=7, alpha=alph, label=f"Level {lvl}")
        )
    fig.legend(handles=legend_elements, fontsize=8, framealpha=0.85,
               title="Encoding", title_fontsize=8,
               loc="lower left",
               bbox_to_anchor=(1.0, 0.0))

    fig.suptitle(
        f"Per-feature scale-response curves — mean |SHAP| vs window size\n"
        f"({'Weighted' if weighted else 'Unweighted'}, "
        f"Levels {levels}, Spectra {spectra_types})",
        fontsize=11, y=1.01,
    )
    fig.subplots_adjust(right=0.88)  # reserve right margin for external legend

    if out_path is None:
        lvl_str = "_".join(str(l) for l in levels)
        out_path = output_dir / f"feature_scale_strip_levels{lvl_str}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature scale strip saved: {out_path}")


# ---------------------------------------------------------------------------
# Scale response strip  (cross-model, Chart 2)
# ---------------------------------------------------------------------------

def scale_response_strip(
        output_dir: Path,
        levels: list[int] = None,
        spectra_types: list[str] = None,
        weighted: bool = True,
        shap_col: str = "mean_abs_shap",
        out_path: "Path | None" = None,
        **kwargs,
) -> None:
    """
    Strip + IQR-band plot of scale-response curves across all 16 models.
    Two subplots: GLCM (left) and sdiv (right).

    x-axis: window size plotted on a log2 scale (linear tick spacing,
            labelled with actual pixel sizes).
    y-axis: mean |SHAP| per window size per model.

    Each dot = one (spectra × level) model:
      Colour  — reflectance A/C = green  (#009E73),
                derivative  B/D = purple (#CC79A7).
      Marker  — A/B = circle, C/D = diamond.
      Opacity — encodes level: L3 opaque → L1 faint.

    A grey IQR band and median line summarise the 16-model distribution
    per window size.

    Reads scale_response.csv from each model directory. Expects columns:
        family, window_size, <shap_col>.

    Args:
        output_dir:    Root output directory.
        levels:        Levels to include. Defaults to [3, 2, 1].
        spectra_types: Spectra to include. Defaults to ['A','B','C','D'].
        weighted:      Use weighted model outputs.
        shap_col:      SHAP column in scale_response.csv (default: mean_abs_shap).
        out_path:      Output PNG path. Auto-generated if None.
    """
    import matplotlib.lines as mlines

    levels = levels or [3, 2, 1]
    spectra_types = spectra_types or ["A", "B", "C", "D"]

    SP_COLOUR = {"A": "#009E73", "B": "#CC79A7", "C": "#009E73", "D": "#CC79A7"}
    SP_MARKER = {"A": "o", "B": "o", "C": "D", "D": "D"}
    LEVEL_ALPHA = {3: 1.0, 2: 0.62, 1: 0.35}  # highest level most opaque

    # ── collect ───────────────────────────────────────────────────────────────
    frames = []
    for level in levels:
        for sp in spectra_types:
            df = load_scale_response(output_dir, sp, level, weighted)
            if df is not None:
                frames.append(df)

    if not frames:
        logger.error("No scale_response data found — aborting scale_response_strip.")
        return

    all_data = pd.concat(frames, ignore_index=True)

    if shap_col not in all_data.columns:
        candidates = [c for c in all_data.columns if "shap" in c.lower()]
        if candidates:
            shap_col = candidates[0]
            logger.warning(f"shap_col not found; using '{shap_col}'.")
        else:
            logger.error(f"No SHAP column found in scale_response data — aborting.")
            return

    families = [f for f in ["glcm", "sdiv"] if f in all_data["family"].unique()]
    family_titles = {"glcm": "GLCM texture", "sdiv": "Spectral diversity (sdiv)"}

    fig, axes = plt.subplots(1, len(families), figsize=(6.2 * len(families), 5.0),
                             sharey=False)
    if len(families) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, family in zip(axes, families):
        sub = all_data[all_data["family"] == family].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        window_sizes = sorted(sub["window_size"].unique())
        log_x = {ws: float(np.log2(ws)) for ws in window_sizes}

        # ── IQR band + median line ─────────────────────────────────────────
        agg = (
            sub.groupby("window_size")[shap_col]
            .agg(q25=lambda s: s.quantile(0.25),
                 median="median",
                 q75=lambda s: s.quantile(0.75))
            .loc[window_sizes]
        )

        xs = [log_x[ws] for ws in window_sizes]
        ax.fill_between(xs, agg["q25"].values, agg["q75"].values,
                        color="#cccccc", alpha=0.55, zorder=1, label="IQR")
        ax.plot(xs, agg["median"].values,
                color="#555555", linewidth=1.8, zorder=3, label="Median")

        # ── individual model dots ─────────────────────────────────────────
        for _, row in sub.iterrows():
            sp = str(row["spectra"])
            lvl = int(row["level"])
            ws = float(row["window_size"])
            val = float(row[shap_col])
            xj = log_x[ws] + rng.uniform(-0.09, 0.09)
            ax.scatter(xj, val,
                       color=SP_COLOUR[sp], marker=SP_MARKER[sp],
                       s=42, alpha=LEVEL_ALPHA.get(lvl, 0.6),
                       edgecolors="none", zorder=4)

        # ── x-axis on log2 scale ──────────────────────────────────────────
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(ws)) for ws in window_sizes], fontsize=9)
        ax.set_xlabel("Window size (pixels)", fontsize=10)
        ax.set_title(family_titles.get(family, family), fontsize=10, fontweight=500)
        if ax is axes[0]:
            ax.set_ylabel("Mean |SHAP|", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    # ── shared legend ─────────────────────────────────────────────────────────
    legend_elements = [
        mlines.Line2D([0], [0], color="#555555", linewidth=1.8, label="Median (all models)"),
        mpatches.Patch(facecolor="#cccccc", alpha=0.7, label="IQR (all models)"),
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="#009E73", markeredgecolor="none",
                      markersize=7, label="Reflectance A/B (circle)"),
        mlines.Line2D([0], [0], marker="D", color="w",
                      markerfacecolor="#009E73", markeredgecolor="none",
                      markersize=7, label="Reflectance C/D (diamond)"),
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="#CC79A7", markeredgecolor="none",
                      markersize=7, label="Derivative B/D (purple)"),
    ]
    for lvl, alph in sorted(LEVEL_ALPHA.items(), reverse=True):
        legend_elements.append(
            mlines.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="#555555", markeredgecolor="none",
                          markersize=7, alpha=alph, label=f"Level {lvl}")
        )
    axes[-1].legend(handles=legend_elements, fontsize=8, framealpha=0.85,
                    title="Encoding", title_fontsize=8,
                    loc="upper right")

    fig.suptitle(
        f"Scale-response curves — mean |SHAP| vs window size across all models\n"
        f"({'Weighted' if weighted else 'Unweighted'}, "
        f"Levels {levels}, Spectra {spectra_types})",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()

    if out_path is None:
        lvl_str = "_".join(str(l) for l in levels)
        out_path = output_dir / f"scale_response_strip_levels{lvl_str}.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Scale response strip saved: {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def held_out_dot_plot(
        comparison_csv: Path,
        levels: list[int] | None = None,
        out_path: Path | None = None,
        **kwargs,
) -> None:
    """
    Dot plot comparing test set accuracy vs held-out ROI accuracy across spectra,
    one sub-figure per level sharing y-axis [0, 1.08].

    Reads held_out_accuracy_comparison.csv produced by summarise_held_out().

    Spectra grouped as A/B and C/D. Reflectance (A, C) = green (#009E73),
    derivative (B, D) = purple (#CC79A7). A/B = triangle markers, C/D = squares.
    Main (test set) = filled, held-out = hollow. Arrow from main → held-out.
    Value labels shown above each marker.

    Args:
        comparison_csv: Path to held_out_accuracy_comparison.csv.
        levels:         Levels to plot. Defaults to [3, 2, 1].
        out_path:       Output PNG path. Auto-generated alongside CSV if None.
    """
    import matplotlib.lines as mlines

    levels = levels or [3, 2, 1]
    df = pd.read_csv(comparison_csv)

    COLOUR = {"A": "#009E73", "B": "#CC79A7", "C": "#009E73", "D": "#CC79A7"}
    MARKER = {"A": "^", "B": "^", "C": "s", "D": "s"}
    X_POS = {"A": 0, "B": 1, "C": 2.2, "D": 3.2}
    XTICKS = [0, 1, 2.2, 3.2]
    XLABELS = ["A", "B", "C", "D"]

    n_levels = len(levels)
    fig, axes = plt.subplots(1, n_levels, figsize=(n_levels * 3.5, 5.0),
                             sharey=True)
    if n_levels == 1:
        axes = [axes]

    for ax, level in zip(axes, levels):
        sub = df[df["level"] == level]

        for _, row in sub.iterrows():
            sp = str(row["spectra"])
            x = X_POS[sp]
            colour = COLOUR[sp]
            marker = MARKER[sp]
            main_val = float(row["test_macro_f1"])
            held_val = float(row["held_out_mean"])

            # Arrow from main → held-out (tip at held-out marker edge)
            ax.annotate(
                "", xy=(x, held_val), xytext=(x, main_val),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#444444", alpha=0.7,
                    linestyle="dotted",
                    lw=1.2,
                    mutation_scale=8,
                ),
                zorder=1,
            )
            # Main — filled
            ax.scatter(x, main_val, color=colour, marker=marker,
                       s=70, zorder=3, edgecolors=colour, linewidths=1.2)
            # Held-out — hollow
            ax.scatter(x, held_val, color="white", marker=marker,
                       s=70, zorder=3, edgecolors=colour, linewidths=1.2)

            # Value labels
            ax.text(x, main_val + 0.012, f"{main_val:.3f}",
                    ha="center", va="bottom", fontsize=7, color="#444444")
            ax.text(x, held_val - 0.018, f"{held_val:.3f}",
                    ha="center", va="top", fontsize=7, color="#444444")

        ax.set_xlim(-0.5, 3.7)
        ax.set_ylim(0, 1.08)  # extra headroom so labels near 1.0 don't clip
        ax.set_xticks(XTICKS)
        ax.set_xticklabels(XLABELS, fontsize=10)
        ax.set_xlabel("Spectra", fontsize=10)
        ax.set_title(f"Level {level}", fontsize=10)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(1.6, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)

    axes[0].set_ylabel("Accuracy", fontsize=10)

    legend_elements = [
        mlines.Line2D([0], [0], marker="^", color="w", markerfacecolor="#009E73",
                      markeredgecolor="#009E73", markersize=7, label="Reflectance (A, C)"),
        mlines.Line2D([0], [0], marker="^", color="w", markerfacecolor="#CC79A7",
                      markeredgecolor="#CC79A7", markersize=7, label="Second derivative (B, D)"),
        mlines.Line2D([0], [0], marker="s", color="w", markerfacecolor="#555555",
                      markeredgecolor="#555555", markersize=7, label="C/D = squares"),
        mlines.Line2D([0], [0], marker="^", color="w", markerfacecolor="#555555",
                      markeredgecolor="#555555", markersize=7, label="Test set"),
        mlines.Line2D([0], [0], marker="^", color="w", markerfacecolor="white",
                      markeredgecolor="#555555", markersize=7, label="Held-out"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=8, framealpha=0.8,
                   loc="lower left", title="Spectra type / source", title_fontsize=8)

    fig.suptitle("Test set vs held-out accuracy by spectra and level",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    if out_path is None:
        out_path = Path(comparison_csv).parent / "held_out_accuracy_dot_plot.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Held-out dot plot saved: {out_path}")


def entropy_dot_plot(
        comparison_csv: Path,
        levels: list[int] | None = None,
        out_path: Path | None = None,
        **kwargs,
) -> None:
    """
    Dot plot comparing main dataset entropy vs held-out entropy across spectra,
    one sub-figure per level sharing y-axis [0.0, 0.3].

    Reads entropy_comparison.csv produced by summarise_entropy().

    Same colour/marker/grouping conventions as held_out_dot_plot():
    green (#009E73) for reflectance (A, C), purple (#CC79A7) for derivative (B, D).
    A/B = triangle markers, C/D = squares. Main = filled, held-out = hollow.
    Arrow from main → held-out. Value labels shown above/below each marker.

    Args:
        comparison_csv: Path to entropy_comparison.csv.
        levels:         Levels to plot. Defaults to [3, 2, 1].
        out_path:       Output PNG path. Auto-generated alongside CSV if None.
    """
    import matplotlib.lines as mlines

    levels = levels or [3, 2, 1]
    df = pd.read_csv(comparison_csv)

    COLOUR = {"A": "#009E73", "B": "#CC79A7", "C": "#009E73", "D": "#CC79A7"}
    MARKER = {"A": "^", "B": "^", "C": "s", "D": "s"}
    X_POS = {"A": 0, "B": 1, "C": 2.2, "D": 3.2}
    XTICKS = [0, 1, 2.2, 3.2]
    XLABELS = ["A", "B", "C", "D"]

    n_levels = len(levels)
    fig, axes = plt.subplots(1, n_levels, figsize=(n_levels * 3.5, 5.0),
                             sharey=True)
    if n_levels == 1:
        axes = [axes]

    for ax, level in zip(axes, levels):
        sub = df[df["level"] == level]

        for _, row in sub.iterrows():
            sp = str(row["spectra"])
            x = X_POS[sp]
            colour = COLOUR[sp]
            marker = MARKER[sp]
            main_val = float(row["main_mean_entropy"])
            held_val = float(row["held_out_mean_entropy"])

            # Arrow from main → held-out
            ax.annotate(
                "", xy=(x, held_val), xytext=(x, main_val),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="#444444", alpha=0.7,
                    linestyle="dotted",
                    lw=1.2,
                    mutation_scale=8,
                ),
                zorder=1,
            )
            # Main — filled
            ax.scatter(x, main_val, color=colour, marker=marker,
                       s=70, zorder=3, edgecolors=colour, linewidths=1.2)
            # Held-out — hollow
            ax.scatter(x, held_val, color="white", marker=marker,
                       s=70, zorder=3, edgecolors=colour, linewidths=1.2)

            # Value labels
            ax.text(x, held_val + 0.006, f"{held_val:.3f}",
                    ha="center", va="bottom", fontsize=7, color="#444444")
            ax.text(x, main_val - 0.006, f"{main_val:.3f}",
                    ha="center", va="top", fontsize=7, color="#444444")

        ax.set_xlim(-0.5, 3.7)
        ax.set_ylim(0.0, 0.5)
        ax.set_xticks(XTICKS)
        ax.set_xticklabels(XLABELS, fontsize=10)
        ax.set_xlabel("Spectra", fontsize=10)
        ax.set_title(f"Level {level}", fontsize=10)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(1.6, color="#cccccc", linewidth=0.8, linestyle="--", zorder=0)

    axes[0].set_ylabel("Entropy (normalised)", fontsize=10)

    legend_elements = [
        mlines.Line2D([0], [0], marker="^", color="w", markerfacecolor="#009E73",
                      markeredgecolor="#009E73", markersize=7, label="Reflectance (A, C)"),
        mlines.Line2D([0], [0], marker="^", color="w", markerfacecolor="#CC79A7",
                      markeredgecolor="#CC79A7", markersize=7, label="Second derivative (B, D)"),
        mlines.Line2D([0], [0], marker="s", color="w", markerfacecolor="#555555",
                      markeredgecolor="#555555", markersize=7, label="C/D = squares"),
        mlines.Line2D([0], [0], marker="^", color="w", markerfacecolor="#555555",
                      markeredgecolor="#555555", markersize=7, label="Main dataset"),
        mlines.Line2D([0], [0], marker="^", color="w", markerfacecolor="white",
                      markeredgecolor="#555555", markersize=7, label="Held-out"),
    ]
    axes[0].legend(handles=legend_elements, fontsize=8, framealpha=0.8,
                   loc="upper left", title="Spectra type / source", title_fontsize=8)

    fig.suptitle("Main dataset vs held-out entropy by spectra and level",
                 fontsize=11, y=1.01)
    fig.tight_layout()

    if out_path is None:
        out_path = Path(comparison_csv).parent / "entropy_dot_plot.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Entropy dot plot saved: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-model feature rank visualisations for the algal turf pipeline."
    )
    parser.add_argument("--type",
                        choices=["bump", "heatmap", "biplot", "wavelength", "beeswarm", "waterfall", "interesting",
                                 "umap", "pairwise", "cv_held_out", "cv_stability", "cv_delta_heatmap",
                                 "held_out_dot", "entropy_dot",
                                 "family_share", "scale_response_strip",
                                 "feature_scale_strip",
                                 "both"], default="both",
                        help="Which chart(s) to produce (default: both)")
    parser.add_argument("--model-a", nargs=2, metavar=("SPECTRA", "LEVEL"),
                        default=None,
                        help="Model A for biplot e.g. --model-a A 2")
    parser.add_argument("--model-b", nargs=2, metavar=("SPECTRA", "LEVEL"),
                        default=None,
                        help="Model B for biplot e.g. --model-b A 4")
    parser.add_argument("--class-a", type=str, default=None,
                        help="First class name for pairwise comparison e.g. turf_algae")
    parser.add_argument("--class-b", type=str, default=None,
                        help="Second class name for pairwise comparison e.g. cca")
    parser.add_argument("--pairwise-raw", action="store_true",
                        help="Use raw SHAP values (default: use importance CSV)")
    parser.add_argument("--pairwise-top-n", type=int, default=30,
                        help="Top N features for pairwise plot (default: 30)")
    parser.add_argument("--umap-source", type=str, default="shap", choices=["leaf", "shap"],
                        help="Embedding source for UMAP: leaf or shap (default: shap)")
    parser.add_argument("--umap-n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors (default: 15)")
    parser.add_argument("--umap-min-dist", type=float, default=0.1,
                        help="UMAP min_dist (default: 0.1)")
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
    parser.add_argument("--cv-metric", type=str, default="pixel_accuracy",
                        help="CV metric to compare against held-out accuracy: "
                             "'pixel_accuracy' (default, comparable to prop_correct) "
                             "or 'macro_f1'")
    parser.add_argument("--held-out-pattern", type=str, default="held_out",
                        help="Substring to match held-out map directory names "
                             "(default: 'held_out')")
    parser.add_argument("--unweighted", action="store_true",
                        help="Use unweighted model outputs (default: weighted)")
    parser.add_argument("--shap-col", type=str, default="mean_abs_shap_global",
                        help="SHAP column to rank by (default: mean_abs_shap_global)")
    parser.add_argument("--comparison-csv", type=Path, default=None,
                        help="Path to *_comparison.csv for held_out_dot or entropy_dot plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weighted = not args.unweighted

    if args.type == "cv_stability":
        cv_fold_stability_strip(
            output_dir=args.output_dir,
            spectra_types=args.spectra,
            levels=args.levels,
            weighted=weighted,
            held_out_pattern=args.held_out_pattern,
            cv_metric=args.cv_metric,
        )

    if args.type == "cv_delta_heatmap":
        cv_held_out_delta_heatmap(
            output_dir=args.output_dir,
            spectra_types=args.spectra,
            levels=args.levels,
            weighted=weighted,
            held_out_pattern=args.held_out_pattern,
            cv_metric=args.cv_metric,
        )

    if args.type == "cv_held_out":
        cv_vs_held_out_chart(
            output_dir=args.output_dir,
            spectra_types=args.spectra,
            levels=args.levels,
            weighted=weighted,
            held_out_pattern=args.held_out_pattern,
            cv_metric=args.cv_metric,
        )

    if args.type == "held_out_dot":
        if args.comparison_csv is None:
            raise ValueError("--comparison-csv is required for held_out_dot")
        held_out_dot_plot(
            comparison_csv=args.comparison_csv,
            levels=args.levels,
        )

    if args.type == "entropy_dot":
        if args.comparison_csv is None:
            raise ValueError("--comparison-csv is required for entropy_dot")
        entropy_dot_plot(
            comparison_csv=args.comparison_csv,
            levels=args.levels,
        )

    if args.type == "family_share":
        family_share_bar(
            output_dir=args.output_dir,
            levels=args.levels,
            spectra_types=args.spectra,
            weighted=weighted,
            shap_col=args.shap_col,
        )

    if args.type == "feature_scale_strip":
        feature_scale_strip(
            output_dir=args.output_dir,
            levels=args.levels,
            spectra_types=args.spectra,
            weighted=weighted,
            shap_col=args.shap_col,
        )

    if args.type == "scale_response_strip":
        scale_response_strip(
            output_dir=args.output_dir,
            levels=args.levels,
            spectra_types=args.spectra,
            weighted=weighted,
        )

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

    if args.type == "pairwise":
        if args.model_a is None or args.class_a is None or args.class_b is None:
            raise ValueError(
                "--model-a, --class-a and --class-b are required for pairwise\n"
                "e.g. --model-a A 2 --class-a turf_algae --class-b cca"
            )
        pairwise_class_shap(
            output_dir=args.output_dir,
            spectra=args.model_a[0],
            level=int(args.model_a[1]),
            class_a=args.class_a,
            class_b=args.class_b,
            weighted=weighted,
            top_n=args.pairwise_top_n,
            use_raw=args.pairwise_raw,
        )

    if args.type == "umap":
        if args.model_a is None:
            raise ValueError("--model-a is required for umap e.g. --model-a A 3")
        if args.umap_source == "leaf":
            raise ValueError(
                "source='leaf' requires booster, df, feature_cols, y, le — "
                "call umap_plot() directly from Python for leaf embeddings. "
                "Use --umap-source shap for CLI usage."
            )
        umap_plot(
            output_dir=args.output_dir,
            spectra=args.model_a[0],
            level=int(args.model_a[1]),
            source=args.umap_source,
            weighted=weighted,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
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