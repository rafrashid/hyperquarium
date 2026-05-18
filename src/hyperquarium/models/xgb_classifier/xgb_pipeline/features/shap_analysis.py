"""
features/shap_analysis.py
SHAP value computation, feature importance, scale-response analysis,
and cross-spectra comparison. Core module for the turf algae hypothesis.
All outputs saved to disk — no interactive display (HPC-safe).
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from config.config import (
    SHAPConfig, SHAP_CFG, WINDOW_SIZES, TURF_ALGAE_CLASS,
    classify_column, extract_window_size,
)
from utils.io import save_csv, save_parquet, save_dataframe, save_figure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
        booster: xgb.Booster,
        dmatrix: xgb.DMatrix,
        feature_names: list[str],
        n_classes: int,
        cfg: SHAPConfig = SHAP_CFG,
        out_dir: Path | None = None,
) -> np.ndarray:
    """
    Computes SHAP values using XGBoost's built-in pred_contribs.
    Optionally subsamples rows first for speed (cfg.shap_sample_size).

    Args:
        booster:       Trained Booster.
        dmatrix:       DMatrix to compute SHAP for.
        feature_names: List of feature names.
        n_classes:     Number of classes.
        cfg:           SHAPConfig instance.
        out_dir:       If provided, saves shap_values.parquet here.

    Returns:
        SHAP array — shape:
          binary:     (n_samples, n_features)        [last col is bias; excluded]
          multiclass: (n_samples, n_classes, n_features)
    """
    raw = booster.predict(dmatrix, pred_contribs=True)

    n_feat = len(feature_names)
    n_samples = dmatrix.num_row()

    # Infer actual trained n_classes from raw output size — safer than trusting
    # the passed n_classes which may differ when data is filtered or subsampled
    raw_size = raw.size
    if raw_size == n_samples * (n_feat + 1):
        # Binary: shape (n_samples, n_features + 1)
        trained_n_classes = 2
    else:
        # Multiclass: shape (n_samples * n_classes * (n_features + 1),)
        trained_n_classes = raw_size // (n_samples * (n_feat + 1))
        if trained_n_classes * n_samples * (n_feat + 1) != raw_size:
            raise ValueError(
                f"Cannot infer n_classes from raw SHAP shape {raw.shape}. "
                f"n_samples={n_samples}, n_features={n_feat}"
            )

    if trained_n_classes != n_classes:
        logger.warning(
            f"n_classes mismatch — passed: {n_classes}, "
            f"inferred from booster: {trained_n_classes}. Using {trained_n_classes}."
        )
        n_classes = trained_n_classes

    if n_classes == 2:
        # Shape: (n_samples, n_features + 1) — last col is bias term
        bias_vals = raw[:, -1]
        shap_vals = raw[:, :-1]
        shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    else:
        # Shape: (n_samples, n_classes * (n_features + 1))
        raw_3d = raw.reshape(-1, n_classes, n_feat + 1)
        bias_vals = raw_3d[:, :, -1].mean(axis=1)
        shap_vals = raw_3d[:, :, :-1]
        cols = [f"{feat}__class{c}" for c in range(n_classes) for feat in feature_names]
        shap_df = pd.DataFrame(shap_vals.reshape(-1, n_classes * n_feat), columns=cols)

    if out_dir is not None:
        save_dataframe(shap_df, out_dir / "shap_values", index=False)
        logger.info(f"SHAP values saved — shape: {shap_df.shape}")
        # Save bias term (base value E[f(x)]) — used by shap_waterfall()
        bias_df = pd.DataFrame({"base_value": bias_vals})
        save_csv(bias_df, out_dir / "shap_base_values.csv", index=False)
        logger.info(f"SHAP base values saved — mean: {bias_vals.mean():.4f}")

    return shap_vals


# ---------------------------------------------------------------------------
# Feature importance from SHAP
# ---------------------------------------------------------------------------

def shap_feature_importance(
        shap_vals: np.ndarray,
        feature_names: list[str],
        le: LabelEncoder,
        n_classes: int,
        turf_algae_class: str = TURF_ALGAE_CLASS,
        out_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Computes mean |SHAP| per feature, globally and per class.
    For the turf algae hypothesis: isolates mean |SHAP| for turf algae specifically
    to identify which features are most diagnostic for that class.

    Args:
        shap_vals:        SHAP array from compute_shap_values().
        feature_names:    List of feature names.
        le:               Fitted LabelEncoder.
        n_classes:        Number of classes.
        turf_algae_class: Name of turf algae class for isolated importance.
        out_dir:          If provided, saves feature_importance.csv here.

    Returns:
        DataFrame with mean |SHAP| columns: global + per class.
    """
    class_names = list(le.classes_)
    importance = pd.DataFrame(index=feature_names)

    if n_classes == 2:
        importance["mean_abs_shap_global"] = np.abs(shap_vals).mean(axis=0)
        importance["mean_abs_shap_turf"] = np.abs(shap_vals).mean(axis=0)
    else:
        importance["mean_abs_shap_global"] = np.abs(shap_vals).mean(axis=(0, 1))
        for i, cls_name in enumerate(class_names):
            col = f"mean_abs_shap_{cls_name.replace(' ', '_')}"
            importance[col] = np.abs(shap_vals[:, i, :]).mean(axis=0)

    importance = importance.sort_values("mean_abs_shap_global", ascending=False)

    if out_dir is not None:
        save_csv(importance, out_dir / "feature_importance_shap.csv")

    logger.info(f"Top 5 features (global mean |SHAP|):\n{importance['mean_abs_shap_global'].head()}")
    return importance


# ---------------------------------------------------------------------------
# Feature family grouping
# ---------------------------------------------------------------------------

def assign_feature_family(feature_names: list[str]) -> pd.Series:
    """
    Assigns each feature to a family using the naming conventions in config.py.
    Uses regex-based classify_column() — exact match against known patterns.

    Returns a Series indexed by feature name with values: 'spectral', 'glcm', 'sdiv'.

    Args:
        feature_names: List of feature column names.

    Returns:
        Series indexed by feature name with family labels.
    """
    families = {f: classify_column(f) for f in feature_names}
    return pd.Series(families, name="family")


def shap_by_family(
        importance_df: pd.DataFrame,
        feature_names: list[str],
        out_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Aggregates mean |SHAP| by feature family.
    Useful for comparing spectral vs GLCM vs spectral diversity contributions.

    Args:
        importance_df: Output of shap_feature_importance().
        feature_names: List of feature names.
        out_dir:       If provided, saves shap_by_family.csv here.

    Returns:
        DataFrame of summed mean |SHAP| per family, per class column.
    """
    family_series = assign_feature_family(feature_names)
    merged = importance_df.join(family_series)
    family_summary = merged.groupby("family").sum(numeric_only=True)

    if out_dir is not None:
        save_csv(family_summary, out_dir / "shap_by_family.csv")

    return family_summary


# ---------------------------------------------------------------------------
# Scale-response analysis
# ---------------------------------------------------------------------------

# extract_window_size is imported directly from config.config
# It uses regex against the actual naming conventions:
#   GLCM: <metric>_window_<size>  e.g. contrast_window_71
#   sdiv: sdiv_<type>_<measure>_plot_<size>  e.g. sdiv_alpha_sdiv_plot_25


def scale_response_curve(
        importance_df: pd.DataFrame,
        feature_names: list[str],
        turf_col: str | None = None,
        out_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Computes aggregate mean |SHAP| per feature family per window size.
    Plots scale-response curves (importance vs log window size).

    The peak window size per family is ecologically meaningful:
    - GLCM peak = texture granularity most diagnostic for turf algae
    - Spectral diversity peak = spatial scale of within-patch mixing

    Args:
        importance_df: Output of shap_feature_importance().
        feature_names: List of feature names.
        window_sizes:  List of valid window sizes (in order).
        turf_col:      Column name for turf algae-specific SHAP (optional).
        out_dir:       If provided, saves scale_response.csv and scale_response.png here.

    Returns:
        DataFrame of aggregate SHAP per family per window size.
    """
    family_series = assign_feature_family(feature_names)
    merged = importance_df.join(family_series)
    merged["window_size"] = [extract_window_size(f) for f in merged.index]

    # Only glcm and sdiv features have a window size
    spatial = merged[merged["family"].isin(["glcm", "sdiv"])].copy()
    spatial["window_size"] = spatial["window_size"].astype(int)

    shap_col = turf_col if (turf_col and turf_col in spatial.columns) else "mean_abs_shap_global"
    scale_df = spatial.groupby(["family", "window_size"])[shap_col].sum().reset_index()
    scale_df.columns = ["family", "window_size", "sum_mean_abs_shap"]

    if out_dir is not None:
        save_csv(scale_df, out_dir / "scale_response.csv", index=False)
        _plot_scale_response(scale_df, shap_col_label=shap_col, out_dir=out_dir)

    return scale_df


def _plot_scale_response(scale_df: pd.DataFrame, shap_col_label: str, out_dir: Path) -> None:
    """Plots scale-response curves per spatial feature family on a log x-axis."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, grp in scale_df.groupby("family"):
        grp_sorted = grp.sort_values("window_size")
        ax.plot(grp_sorted["window_size"], grp_sorted["sum_mean_abs_shap"],
                marker="o", label=family, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xticks(scale_df["window_size"].unique())
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set(
        xlabel="Window size (log scale)",
        ylabel=f"Sum mean |SHAP| ({shap_col_label})",
        title="Scale-response curve — spatial feature families",
    )
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    save_figure(fig, out_dir / "scale_response.png")


# ---------------------------------------------------------------------------
# SHAP dependence plots
# ---------------------------------------------------------------------------

def plot_shap_dependence(
        shap_vals: np.ndarray,
        feature_names: list[str],
        X: np.ndarray,
        n_classes: int,
        turf_class_idx: int | None,
        top_n: int = 20,
        out_dir: Path | None = None,
) -> None:
    """
    Plots SHAP dependence plots for top N features.
    For multiclass, uses turf algae class SHAP if turf_class_idx is provided.
    Each plot: feature value (x) vs SHAP value (y) — reveals relationship shape.

    Args:
        shap_vals:      SHAP array from compute_shap_values().
        feature_names:  List of feature names.
        X:              Feature matrix (n_samples, n_features).
        n_classes:      Number of classes.
        turf_class_idx: Class index for turf algae (multiclass only).
        top_n:          Number of top features to plot.
        out_dir:        Output directory for PNG files.
    """
    if out_dir is None:
        return

    dep_dir = out_dir / "shap_dependence"
    dep_dir.mkdir(parents=True, exist_ok=True)

    if n_classes == 2:
        shap_2d = shap_vals  # (n_samples, n_features)
    else:
        idx = turf_class_idx if turf_class_idx is not None else 0
        shap_2d = shap_vals[:, idx, :]  # (n_samples, n_features)

    mean_abs = np.abs(shap_2d).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:top_n]

    for rank, fi in enumerate(top_indices):
        feat_name = feature_names[fi]
        feat_vals = X[:, fi]
        shap_feat = shap_2d[:, fi]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(feat_vals, shap_feat, alpha=0.3, s=4, rasterized=True)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set(
            xlabel=feat_name,
            ylabel="SHAP value",
            title=f"SHAP dependence — {feat_name} (rank {rank + 1})",
        )
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        safe_name = feat_name.replace("/", "_").replace(" ", "_")
        save_figure(fig, dep_dir / f"rank{rank + 1:02d}_{safe_name}.png", dpi=100)

    logger.info(f"Saved {top_n} SHAP dependence plots to {dep_dir}")


# ---------------------------------------------------------------------------
# Cross-spectra comparison
# ---------------------------------------------------------------------------

def compare_shap_across_spectra(
        importance_dfs: dict[str, pd.DataFrame],
        out_dir: Path,
) -> pd.DataFrame:
    """
    Compares mean |SHAP| feature rankings across all spectra types (A, B, C, D).
    Identifies features that are consistently important (robust signal)
    vs features unique to derivative spectra (subtle absorption features).

    Args:
        importance_dfs: Dict mapping spectra label -> importance DataFrame
                        (output of shap_feature_importance(), one per spectra).
        out_dir:        Output directory for comparison outputs.

    Returns:
        DataFrame with rank and mean |SHAP| per feature per spectra type.
    """
    frames = []
    for spectra_label, df in importance_dfs.items():
        col = df["mean_abs_shap_global"].rename(f"shap_{spectra_label}")
        frames.append(col)

    comparison = pd.concat(frames, axis=1).fillna(0)

    # Rank features within each spectra
    for spectra_label in importance_dfs:
        comparison[f"rank_{spectra_label}"] = (
            comparison[f"shap_{spectra_label}"].rank(ascending=False).astype(int)
        )

    # Consistency score: low std in rank across spectra = consistently important
    rank_cols = [c for c in comparison.columns if c.startswith("rank_")]
    comparison["rank_std"] = comparison[rank_cols].std(axis=1)
    comparison["rank_mean"] = comparison[rank_cols].mean(axis=1)
    comparison = comparison.sort_values("rank_mean")

    save_csv(comparison, out_dir / "cross_spectra_shap_comparison.csv")
    logger.info(f"Cross-spectra SHAP comparison saved: {out_dir / 'cross_spectra_shap_comparison.csv'}")
    return comparison


# ---------------------------------------------------------------------------
# Full SHAP run
# ---------------------------------------------------------------------------

def run_shap_analysis(
        booster: xgb.Booster,
        dmatrix: xgb.DMatrix,
        X: np.ndarray,
        feature_names: list[str],
        le: LabelEncoder,
        n_classes: int,
        cfg: SHAPConfig = SHAP_CFG,
        out_dir: Path = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Runs the full SHAP analysis suite:
    SHAP values → feature importance → family grouping → scale-response → dependence plots.

    Args:
        booster:       Trained Booster.
        dmatrix:       DMatrix (can be test set or a subsample).
        X:             Raw feature matrix corresponding to dmatrix rows.
        feature_names: List of feature names.
        le:            Fitted LabelEncoder.
        n_classes:     Number of classes.
        cfg:           SHAPConfig instance.
        out_dir:       Output directory.

    Returns:
        (shap_vals array, importance DataFrame)
    """
    logger.info(f"Running SHAP analysis — output: {out_dir}")

    class_names = list(le.classes_)
    turf_class_idx = (
        class_names.index(TURF_ALGAE_CLASS) if TURF_ALGAE_CLASS in class_names else None
    )
    turf_col = (
        f"mean_abs_shap_{TURF_ALGAE_CLASS.replace(' ', '_')}"
        if turf_class_idx is not None else None
    )

    shap_vals = compute_shap_values(booster, dmatrix, feature_names, n_classes, cfg, out_dir)
    importance = shap_feature_importance(shap_vals, feature_names, le, n_classes, out_dir=out_dir)
    shap_by_family(importance, feature_names, out_dir=out_dir)
    scale_response_curve(importance, feature_names, turf_col=turf_col, out_dir=out_dir)
    plot_shap_dependence(
        shap_vals, feature_names, X, n_classes,
        turf_class_idx=turf_class_idx,
        top_n=cfg.n_top_features,
        out_dir=out_dir,
    )

    return shap_vals, importance


# ---------------------------------------------------------------------------
# Dimensionality reduction — PCA → t-SNE
# ---------------------------------------------------------------------------

def plot_pca_tsne(
        embedding_matrix: np.ndarray,
        y: np.ndarray,
        le: LabelEncoder,
        turf_algae_class: str = TURF_ALGAE_CLASS,
        pca_components: int = 50,
        sample_size: int = 10_000,
        random_seed: int = 42,
        out_dir: Path | str | None = None,
        title_suffix: str = "",
) -> None:
    """
    Produces a PCA → t-SNE 2D scatter plot coloured by class.
    PCA first reduces to pca_components dimensions; t-SNE then projects to 2D.
    This two-step approach is faster and more stable than running t-SNE on raw
    high-dimensional input (leaf embeddings or SHAP values).

    Turf algae points are plotted on top with a larger marker and distinct edge
    so their overlap with other classes is immediately visible.

    Saves: pca_tsne.png and pca_tsne_data.csv (2D coordinates + labels).

    Args:
        embedding_matrix: Input matrix (n_samples, n_features).
                          Typically leaf embeddings (pred_leaf) or SHAP values.
        y:                Integer-encoded true labels.
        le:               Fitted LabelEncoder.
        turf_algae_class: Class name for turf algae — plotted on top.
        pca_components:   Number of PCA components before t-SNE (default 50).
                          Capped automatically if n_features < pca_components.
        sample_size:      Max rows to use — subsampled if dataset is larger.
        random_seed:      Random seed for reproducibility.
        out_dir:          Output directory.
        title_suffix:     Appended to plot title e.g. "spectra A — level 3".
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if out_dir is not None:
        out_dir = Path(out_dir)

    class_names = list(le.classes_)
    rng = np.random.default_rng(random_seed)

    # Subsample if needed
    n = len(embedding_matrix)
    if n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X = embedding_matrix[idx]
        y_s = y[idx]
        logger.info(f"PCA→t-SNE subsampled to {sample_size:,} rows from {n:,}")
    else:
        X, y_s = embedding_matrix, y
        logger.info(f"PCA→t-SNE using all {n:,} rows")

    # Step 1 — PCA
    n_components = min(pca_components, X.shape[1], X.shape[0])
    logger.info(f"PCA: {X.shape[1]} → {n_components} components")
    pca = PCA(n_components=n_components, random_state=random_seed)
    X_pca = pca.fit_transform(X)
    var_explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA variance explained: {var_explained:.1%}")

    # Step 2 — t-SNE
    logger.info("Running t-SNE (this may take a few minutes on large samples)...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(X_pca) - 1),
        random_state=random_seed,
        n_jobs=-1,
    )
    X_2d = tsne.fit_transform(X_pca)
    logger.info("t-SNE complete")

    # Save 2D coordinates
    if out_dir is not None:
        coords_df = pd.DataFrame({
            "tsne_1": X_2d[:, 0],
            "tsne_2": X_2d[:, 1],
            "class": le.inverse_transform(y_s),
        })
        save_csv(coords_df, out_dir / "pca_tsne_data.csv", index=False)

    import re as _re
    labels = le.inverse_transform(y_s)

    # Detect Level 4: labels contain _ROI_ suffix e.g. turf_algae_ROI_042
    is_level4 = any("_ROI_" in c for c in class_names)

    fig, ax = plt.subplots(figsize=(9, 7))

    if is_level4:
        # Extract Level 2 parent and ROI number from each label
        # e.g. turf_algae_ROI_042 -> parent="turf_algae", roi_num=42
        roi_pattern = _re.compile(r"^(.+)_ROI_(\d+)$")

        # Unique Level 2 parents — used for colour mapping
        parents = sorted(set(
            roi_pattern.match(c).group(1) if roi_pattern.match(c) else c
            for c in class_names
        ))
        cmap = plt.cm.get_cmap("tab10", len(parents))
        colours = {p: cmap(i) for i, p in enumerate(parents)}

        # Plot each point as its ROI number text, coloured by Level 2 parent
        # White stroke path effect added for contrast in dense overlapping regions
        from matplotlib.patheffects import withStroke
        white_edge = [withStroke(linewidth=1.0, foreground="white")]

        for lbl, x, y_ in zip(labels, X_2d[:, 0], X_2d[:, 1]):
            m = roi_pattern.match(lbl)
            if m:
                parent = m.group(1)
                roi_num = str(int(m.group(2)))  # Drop leading zeros: 042 -> 42
            else:
                parent = lbl
                roi_num = "?"
            colour = colours.get(parent, "gray")
            ax.text(x, y_, roi_num, fontsize=5, color=colour,
                    ha="center", va="center", alpha=0.9,
                    fontweight="normal", clip_on=False,
                    path_effects=white_edge)

        # Set axis limits explicitly from data — ax.text() does not auto-update limits
        pad = ((np.nanmax(X_2d[:, 0]) - np.nanmin(X_2d[:, 0])) + (np.nanmax(X_2d[:, 1]) - np.nanmin(X_2d[:, 1]))) * 0.05
        ax.set_xlim(np.nanmin(X_2d[:, 0]) - pad, np.nanmax(X_2d[:, 0]) + pad)
        ax.set_ylim(np.nanmin(X_2d[:, 1]) - pad, np.nanmax(X_2d[:, 1]) + pad)

        # Legend: one entry per Level 2 parent (not per ROI)
        import matplotlib.patches as _patches
        legend_handles = [
            _patches.Patch(color=colours[p], label=p)
            for p in parents
        ]
        ax.legend(handles=legend_handles, fontsize=8, framealpha=0.7,
                  loc="best", title="Level 2 class", title_fontsize=8)

    else:
        # Standard scatter — one colour per class, turf algae on top
        other_cls = [c for c in class_names if c != turf_algae_class]
        plot_order = other_cls + ([turf_algae_class] if turf_algae_class in class_names else [])
        cmap = plt.cm.get_cmap("tab10", len(plot_order))
        colours = {cls: cmap(i) for i, cls in enumerate(plot_order)}

        for cls in plot_order:
            mask = labels == cls
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[colours[cls]],
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

    title = f"PCA → t-SNE — leaf embeddings"
    if title_suffix:
        title += f"\n{title_suffix}"
    title += f"\n(PCA {n_components} components, {var_explained:.0%} variance explained)"
    if is_level4:
        title += "  |  markers = ROI number, colour = Level 2 class"

    ax.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if out_dir is not None:
        save_figure(fig, out_dir / "pca_tsne.png", dpi=150)
    else:
        plt.close(fig)