"""
evaluation/evaluator.py
Model evaluation: metrics, confusion matrices, PR curves, learning curves.
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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

from utils.io import save_csv, save_json, save_figure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
        booster: xgb.Booster,
        dmatrix: xgb.DMatrix,
        n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns predicted class labels and probability arrays.

    Args:
        booster:   Trained Booster.
        dmatrix:   DMatrix of samples to predict.
        n_classes: Number of classes (2 for binary, >2 for multiclass).

    Returns:
        (y_pred_labels, y_pred_proba)
        y_pred_proba shape: (n_samples,) for binary, (n_samples, n_classes) for multiclass.
    """
    raw = booster.predict(dmatrix)

    if n_classes == 2:
        y_pred_proba = raw
        y_pred = (raw >= 0.5).astype(int)
    else:
        # Infer trained n_classes from raw output size, not the passed argument.
        # These differ when the dataset is a subset that lacks some classes
        # (e.g. a stratified sample used for testing the pipeline locally).
        trained_n_classes = raw.size // dmatrix.num_row()
        y_pred_proba = raw.reshape(-1, trained_n_classes)
        y_pred = np.argmax(y_pred_proba, axis=1)

    return y_pred, y_pred_proba


def predict_leaf(booster: xgb.Booster, dmatrix: xgb.DMatrix) -> np.ndarray:
    """
    Returns leaf node indices for each sample across all trees.
    Useful for embedding-based analysis (UMAP/t-SNE).

    Args:
        booster: Trained Booster.
        dmatrix: DMatrix of samples.

    Returns:
        Array of shape (n_samples, n_trees).
    """
    return booster.predict(dmatrix, pred_leaf=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        le: LabelEncoder,
        n_classes: int,
        out_dir: Path,
) -> dict:
    """
    Computes and saves classification metrics.
    Saves: metrics.json, classification_report.csv

    Args:
        y_true:       True integer labels.
        y_pred:       Predicted integer labels.
        y_pred_proba: Predicted probabilities.
        le:           Fitted LabelEncoder (for class names).
        n_classes:    Number of classes.
        out_dir:      Output directory.

    Returns:
        Metrics dictionary.
    """
    class_names = list(le.classes_)

    # Restrict to classes present in y_true — handles subsampled datasets
    # where some classes may have been dropped, without requiring any change
    # to LEVEL_CONFIGS or the label encoder.
    labels_present = sorted(set(y_true))
    names_present = [class_names[i] for i in labels_present]
    n_classes_present = len(labels_present)

    if n_classes_present < len(class_names):
        logger.warning(
            f"{len(class_names) - n_classes_present} class(es) absent from evaluation set: "
            f"{[class_names[i] for i in range(len(class_names)) if i not in labels_present]}"
        )

    report = classification_report(
        y_true, y_pred,
        labels=labels_present,
        target_names=names_present,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    save_csv(report_df, out_dir / "classification_report.csv")

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Average precision (AUC-PR) — only for classes present in y_true.
    # y_pred_proba columns are indexed 0..trained_n_classes-1 (from the booster),
    # so we index by the original class integer (labels_present[i]), not by i.
    if n_classes_present == 2 and len(class_names) == 2:
        ap_scores = {class_names[1]: average_precision_score(y_true, y_pred_proba)}
    else:
        trained_n_classes = y_pred_proba.shape[1] if y_pred_proba.ndim == 2 else n_classes_present
        y_bin = label_binarize(y_true, classes=labels_present)
        ap_scores = {}
        for i, class_idx in enumerate(labels_present):
            if class_idx < trained_n_classes:
                ap_scores[names_present[i]] = average_precision_score(
                    y_bin[:, i], y_pred_proba[:, class_idx]
                )
            else:
                logger.warning(f"Skipping AUC-PR for '{names_present[i]}' — class index {class_idx} "
                               f"out of range for proba array with {trained_n_classes} columns.")

    pixel_accuracy = float(np.mean(y_true == y_pred))

    metrics = {
        "macro_f1": round(macro_f1, 5),
        "pixel_accuracy": round(pixel_accuracy, 5),
        "average_precision_per_class": {k: round(v, 5) for k, v in ap_scores.items()},
        "per_class_f1": {
            class_names[i]: round(
                f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0), 5
            )
            for i in labels_present
        },
    }
    save_json(metrics, out_dir / "metrics.json")
    logger.info(f"Macro F1: {macro_f1:.4f}  Pixel accuracy: {pixel_accuracy:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        le: LabelEncoder,
        out_dir: Path,
        normalise: bool = True,
) -> None:
    """
    Plots and saves a confusion matrix, normalised by true label (rows sum to 1).
    Normalisation is the default — prevents dominant class swamping rare classes visually.
    Also saves the raw matrix as CSV.

    Args:
        y_true:    True integer labels.
        y_pred:    Predicted integer labels.
        le:        Fitted LabelEncoder.
        out_dir:   Output directory.
        normalise: If True, normalise rows to [0, 1].
    """
    class_names = list(le.classes_)
    cm = confusion_matrix(y_true, y_pred)

    # Restrict to classes actually present in y_true
    labels_present = sorted(set(y_true) | set(y_pred))
    class_names = [class_names[i] for i in labels_present]

    # Save raw counts
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    save_csv(cm_df, out_dir / "confusion_matrix_counts.csv")

    # Normalise
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_plot = cm_norm if normalise else cm

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalise else None)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix (normalised by true label)" if normalise else "Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm_plot.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = f"{cm_plot[i, j]:.2f}" if normalise else str(cm[i, j])
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black", fontsize=9)

    fig.tight_layout()
    save_figure(fig, out_dir / "confusion_matrix.png")


# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_pr_curves(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        le: LabelEncoder,
        n_classes: int,
        out_dir: Path,
) -> None:
    """
    Plots and saves per-class precision-recall curves.
    Preferred over ROC for imbalanced class evaluation.

    Args:
        y_true:       True integer labels.
        y_pred_proba: Predicted probabilities.
        le:           Fitted LabelEncoder.
        n_classes:    Number of classes.
        out_dir:      Output directory.
    """
    class_names = list(le.classes_)
    labels_present = sorted(set(y_true))
    fig, ax = plt.subplots(figsize=(8, 6))

    if len(class_names) == 2:
        prec, rec, _ = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        ax.plot(rec, prec, label=f"{class_names[1]} (AP={ap:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=labels_present)
        for col_idx, class_idx in enumerate(labels_present):
            prec, rec, _ = precision_recall_curve(y_bin[:, col_idx], y_pred_proba[:, class_idx])
            ap = average_precision_score(y_bin[:, col_idx], y_pred_proba[:, class_idx])
            ax.plot(rec, prec, label=f"{class_names[class_idx]} (AP={ap:.3f})")

    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves", xlim=[0, 1], ylim=[0, 1])
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir / "pr_curves.png")


# ---------------------------------------------------------------------------
# Learning curve
# ---------------------------------------------------------------------------

def plot_learning_curve(
        evals_result: dict,
        eval_metric: str | list[str],
        out_dir: Path,
) -> None:
    """
    Plots train vs validation metric per boosting round (from evals_result).
    If eval_metric is a list, plots one panel per metric (e.g. mlogloss + merror).
    Early stopping target is always the last metric in the list.

    Args:
        evals_result: Dict returned by xgb.train() via evals_result parameter.
        eval_metric:  Metric name or list of metric names.
        out_dir:      Output directory.
    """
    # Normalise to list
    metrics = [eval_metric] if isinstance(eval_metric, str) else eval_metric
    n = len(metrics)

    fig, axes = plt.subplots(1, n, figsize=(9 * n, 4), squeeze=False)

    for ax, metric in zip(axes[0], metrics):
        train_vals = evals_result.get("train", {}).get(metric, [])
        val_vals = evals_result.get("val", {}).get(metric, [])

        if not train_vals:
            logger.warning(f"No evals_result data for metric '{metric}' — skipping.")
            continue

        rounds = np.arange(1, len(train_vals) + 1)
        best_round = np.argmin(val_vals) + 1
        ax.plot(rounds, train_vals, label="Train", linewidth=1.2)
        ax.plot(rounds, val_vals, label="Validation", linewidth=1.2)
        ax.axvline(best_round, color="red", linestyle="--", linewidth=1,
                   label=f"Best round ({best_round})")
        ax.set(xlabel="Boosting round", ylabel=metric,
               title=f"Learning curve — {metric}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_dir / "learning_curve.png")


# ---------------------------------------------------------------------------
# Boundary sample identification
# ---------------------------------------------------------------------------

def find_boundary_samples(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        le: LabelEncoder,
        n_classes: int,
        turf_algae_class: str,
        out_dir: Path,
        uncertainty_threshold: float = 0.35,
) -> pd.DataFrame:
    """
    Identifies samples that are uncertain or misclassified at the turf algae boundary.
    These are the most scientifically interesting samples for the overlap hypothesis.

    Saves: boundary_samples.csv

    Args:
        y_true:               True integer labels.
        y_pred_proba:         Predicted probabilities.
        le:                   Fitted LabelEncoder.
        n_classes:            Number of classes.
        turf_algae_class:     Name of the turf algae class.
        out_dir:              Output directory.
        uncertainty_threshold: Max margin between top-2 class probabilities to flag as uncertain.

    Returns:
        DataFrame of flagged samples.
    """
    class_names = list(le.classes_)
    if turf_algae_class not in class_names:
        logger.warning(f"'{turf_algae_class}' not found in classes — skipping boundary sample analysis.")
        return pd.DataFrame()

    turf_idx = list(class_names).index(turf_algae_class)

    if n_classes == 2:
        turf_proba = y_pred_proba
    else:
        turf_proba = y_pred_proba[:, turf_idx]

    # Flag: high turf probability but not labelled turf (or vice versa)
    turf_true_mask = y_true == turf_idx
    turf_pred_mask = turf_proba >= 0.5

    boundary_df = pd.DataFrame({
        "true_label": le.inverse_transform(y_true),
        "turf_probability": turf_proba,
        "predicted_turf": turf_pred_mask,
        "true_turf": turf_true_mask,
        "misclassified": turf_true_mask != turf_pred_mask,
    })

    # Uncertain: turf probability between (uncertainty_threshold, 1 - uncertainty_threshold)
    boundary_df["uncertain"] = (
            (turf_proba > uncertainty_threshold) & (turf_proba < 1 - uncertainty_threshold)
    )

    flagged = boundary_df[boundary_df["misclassified"] | boundary_df["uncertain"]]
    save_csv(flagged, out_dir / "boundary_samples.csv")
    logger.info(
        f"Boundary samples — misclassified: {boundary_df['misclassified'].sum():,} | "
        f"uncertain: {boundary_df['uncertain'].sum():,}"
    )
    return flagged



def plot_confusion_matrix_level2_from_level4(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        le: LabelEncoder,
        out_dir: Path,
) -> None:
    """
    For Level 4 (ROI-level) models only.
    Recovers Level 2 class labels by stripping the '_ROI_NNN' suffix from
    Level 4 predictions and true labels, then plots a confusion matrix at
    the Level 2 granularity — no re-prediction needed.

    This allows direct comparison with the Level 2 model confusion matrix.

    Saves: confusion_matrix_level2_from_level4.png
           confusion_matrix_level2_from_level4_counts.csv

    Args:
        y_true:  True integer labels (Level 4 encoded).
        y_pred:  Predicted integer labels (Level 4 encoded).
        le:      Fitted LabelEncoder for Level 4 labels.
        out_dir: Output directory.
    """
    import re

    # Decode integer labels back to Level 4 strings
    true_l4 = le.inverse_transform(y_true)
    pred_l4 = le.inverse_transform(y_pred)

    # Strip '_ROI_NNN' suffix to recover Level 2 class
    # e.g. turf_algae_ROI_042 -> turf_algae
    roi_pattern = re.compile(r"_ROI_\d+$")
    true_l2 = np.array([roi_pattern.sub("", lbl) for lbl in true_l4])
    pred_l2 = np.array([roi_pattern.sub("", lbl) for lbl in pred_l4])

    # Unique Level 2 classes in sorted order
    class_names = sorted(set(true_l2) | set(pred_l2))

    # Build confusion matrix
    from sklearn.metrics import confusion_matrix as sk_cm
    cm = sk_cm(true_l2, pred_l2, labels=class_names)

    # Save raw counts
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    save_csv(cm_df, out_dir / "confusion_matrix_level2_from_level4_counts.csv")

    # Normalise by true label (rows sum to 1)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted (Level 2, recovered)",
        ylabel="True (Level 2, recovered)",
        title="Confusion matrix — Level 4 predictions mapped back to Level 2",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm_norm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black", fontsize=9)

    fig.tight_layout()
    save_figure(fig, out_dir / "confusion_matrix_level2_from_level4.png")
    logger.info("Level 2 confusion matrix (from Level 4 predictions) saved.")

# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(
        booster: xgb.Booster,
        dtest: xgb.DMatrix,
        y_test: np.ndarray,
        evals_result: dict,
        le: LabelEncoder,
        n_classes: int,
        eval_metric: str,
        turf_algae_class: str,
        out_dir: Path,
) -> dict:
    """
    Runs the full evaluation suite for one model:
    metrics, confusion matrix, PR curves, learning curve, boundary samples.

    Args:
        booster:          Trained Booster.
        dtest:            Test DMatrix.
        y_test:           True test labels.
        evals_result:     Eval history from training.
        le:               Fitted LabelEncoder.
        n_classes:        Number of classes.
        eval_metric:      Eval metric name for learning curve plot.
        turf_algae_class: Name of the turf algae class for boundary analysis.
        out_dir:          Output directory.

    Returns:
        Metrics dictionary.
    """
    logger.info(f"Running evaluation — output: {out_dir}")

    y_pred, y_pred_proba = predict(booster, dtest, n_classes)

    metrics = compute_metrics(y_test, y_pred, y_pred_proba, le, n_classes, out_dir)
    plot_confusion_matrix(y_test, y_pred, le, out_dir)
    plot_pr_curves(y_test, y_pred_proba, le, n_classes, out_dir)
    plot_learning_curve(evals_result, eval_metric, out_dir)
    find_boundary_samples(y_test, y_pred_proba, le, n_classes, turf_algae_class, out_dir)

    # Level 4 only: additional confusion matrix mapped back to Level 2
    if n_classes > 50:  # Heuristic: Level 4 always has many more classes than other levels
        plot_confusion_matrix_level2_from_level4(y_test, y_pred, le, out_dir)

    return metrics


# ---------------------------------------------------------------------------
# Held-out evaluation summary
# ---------------------------------------------------------------------------

def summarise_held_out(
        output_dir: str | Path,
        spectra_types: list[str] | None = None,
        levels: list[int] | None = None,
        weighted: bool = True,
        held_out_stems: list[str] | None = None,
        maps_dir: str | Path | None = None,
        out_path: str | Path | None = None,
) -> "pd.DataFrame":
    """
    Compiles a comparison table of model accuracy on:
        (a) original test set — from metrics.json
        (b) held-out ROIs    — from prop_correct attrs in NetCDF map files

    Reads held-out maps from source-specific subdirectories under maps/ to
    prevent mixing with main dataset predictions. Each subdirectory is named
    after the parquet file stem e.g. maps/held_out_20pct_seed42_spectraA/.

    Args:
        output_dir:       Root output directory.
        spectra_types:    Spectra labels. Defaults to ['A','B','C','D'].
        levels:           Hierarchy levels. Defaults to [3, 2, 1].
        weighted:         Use weighted model outputs.
        held_out_stems:   List of parquet file stems for held-out data.
                          e.g. ['held_out_20pct_seed42_spectraA',
                                'held_out_20pct_seed42_spectraB', ...]
                          If None, auto-discovers subdirs matching
                          'held_out*' pattern under maps/.
        maps_dir:         Root maps directory. Defaults to output_dir / 'maps'.
        out_path:         Output CSV path.

    Returns:
        Summary DataFrame.
    """
    import json
    import pandas as pd
    import xarray as xr

    logger.info("Compiling held-out accuracy summary...")

    output_dir = Path(output_dir)
    spectra_types = spectra_types or ["A", "B", "C", "D"]
    levels = levels or [3, 2, 1]
    maps_dir = Path(maps_dir) if maps_dir else output_dir / "maps"
    suffix = "" if weighted else "_unweighted"

    records = []

    for spectra in spectra_types:
        for level in levels:
            model_dir = output_dir / f"spectra_{spectra}" / f"level_{level}{suffix}"
            entry_base = {"spectra": spectra, "level": level, "weighted": weighted}

            # ── (a) Original test set accuracy ──────────────────────────────
            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                records.append({
                    **entry_base,
                    "source": "original_test_set",
                    "roi_id": None,
                    "macro_f1": m.get("macro_f1"),
                    "prop_correct": None,
                    "n_rois": None,
                })
            else:
                logger.warning(f"metrics.json not found: {metrics_path}")

            # ── (b) Held-out ROI accuracy from NetCDF maps ──────────────────
            if not maps_dir.exists():
                continue

            # Find held-out subdirectories for this spectra type
            if held_out_stems:
                stems = [s for s in held_out_stems
                         if f"spectra{spectra}".lower() in s.lower()
                         or f"spectra_{spectra}".lower() in s.lower()]
            else:
                # Auto-discover: match 'held_out*' (covers both old turf_held_out_*
                # and new held_out_*) for the correct spectra type
                stems = [d.name for d in maps_dir.iterdir()
                         if d.is_dir() and "held_out" in d.name.lower()
                         and (f"spectra{spectra}".lower() in d.name.lower()
                              or f"spectra_{spectra}".lower() in d.name.lower())]

            if not stems:
                logger.info(f"No held-out map subdirs found for spectra {spectra} in {maps_dir}")
                continue

            nc_files = []
            for stem in stems:
                subdir = maps_dir / stem
                if subdir.exists():
                    nc_files.extend(sorted(subdir.glob(f"roi_*_spectra{spectra}_L{level}.nc")))

            if not nc_files:
                logger.info(
                    f"No NetCDF files found for spectra {spectra} level {level} "
                    f"in held-out subdirs"
                )
                continue

            roi_props = []
            for nc_file in nc_files:
                try:
                    ds = xr.open_dataset(nc_file)
                    prop = ds.attrs.get("prop_correct")
                    roi = ds.attrs.get("roi_ID", nc_file.stem)
                    ds.close()
                    if prop is not None and not pd.isna(prop):
                        roi_props.append({"roi_id": roi, "prop_correct": float(prop)})
                except Exception as e:
                    logger.warning(f"Could not read {nc_file.name}: {e}")

            if roi_props:
                roi_df = pd.DataFrame(roi_props)
                mean_prop = roi_df["prop_correct"].mean()
                std_prop = roi_df["prop_correct"].std()
                n_rois = len(roi_df)

                records.append({
                    **entry_base,
                    "source": "held_out_rois",
                    "roi_id": None,
                    "macro_f1": None,
                    "prop_correct": round(mean_prop, 4),
                    "prop_correct_std": round(std_prop, 4),
                    "n_rois": n_rois,
                })

                # Per-ROI rows for detailed inspection
                for row in roi_props:
                    records.append({
                        **entry_base,
                        "source": "held_out_roi_detail",
                        "roi_id": row["roi_id"],
                        "macro_f1": None,
                        "prop_correct": round(row["prop_correct"], 4),
                        "prop_correct_std": None,
                        "n_rois": 1,
                    })

    summary = pd.DataFrame(records)

    # Print comparison table — original test vs held-out mean
    print(f"\n{'=' * 70}")
    print("Held-out accuracy summary")
    print(f"{'=' * 70}")
    print(
        f"{'Spectra':<10} {'Level':<8} {'Test macro F1':>14} {'Held-out mean':>14} "
        f"{'Held-out std':>13} {'N ROIs':>8}"
    )
    print("-" * 70)

    for spectra in spectra_types:
        for level in levels:
            test_row = summary[(summary["spectra"] == spectra) &
                               (summary["level"] == level) &
                               (summary["source"] == "original_test_set")]
            held_row = summary[(summary["spectra"] == spectra) &
                               (summary["level"] == level) &
                               (summary["source"] == "held_out_rois")]

            f1_str = (f"{test_row['macro_f1'].iloc[0]:.4f}"
                      if not test_row.empty and test_row['macro_f1'].iloc[0] is not None
                      else "n/a")
            prop_str = f"{held_row['prop_correct'].iloc[0]:.4f}" if not held_row.empty else "n/a"
            std_str = f"{held_row['prop_correct_std'].iloc[0]:.4f}" if not held_row.empty else "n/a"
            n_str = f"{int(held_row['n_rois'].iloc[0])}" if not held_row.empty else "n/a"

            flag = ""
            if f1_str != "n/a" and prop_str != "n/a":
                drop = float(f1_str) - float(prop_str)
                if drop > 0.05:
                    flag = "  <- ⚠ DROP >5%"
                elif drop > 0.02:
                    flag = "  <- note"

            print(
                f"  {spectra:<8} {level:<8} {f1_str:>14} {prop_str:>14} "
                f"{std_str:>13} {n_str:>8}{flag}"
            )

    print(f"{'=' * 70}\n")

    if out_path is None:
        out_path = output_dir / "held_out_accuracy_summary.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info(f"Held-out summary saved: {out_path}")

    return summary


def summarise_entropy(
        output_dir: str | Path,
        spectra_types: list[str] | None = None,
        levels: list[int] | None = None,
        weighted: bool = True,
        held_out_stems: list[str] | None = None,
        maps_dir: str | Path | None = None,
        out_path: str | Path | None = None,
) -> "pd.DataFrame":
    """
    Compares mean per-pixel entropy between main dataset maps and held-out
    ROI maps. Higher entropy on held-out ROIs = more uncertainty on unseen
    data, consistent with within-class variability.

    Reads entropy DataArray from NetCDF files produced by predict.py.
    Entropy is normalised [0,1] so values are comparable across levels.

    Saves: {output_dir}/entropy_summary.csv

    Args:
        output_dir:     Root output directory.
        spectra_types:  Spectra labels. Defaults to ['A','B','C','D'].
        levels:         Hierarchy levels. Defaults to [3, 2, 1].
        weighted:       Use weighted model outputs.
        held_out_stems: Parquet file stems for held-out data. Auto-discovered
                        from maps/held_out* if None.
        maps_dir:       Root maps directory. Defaults to output_dir / 'maps'.
        out_path:       Output CSV path.

    Returns:
        Summary DataFrame.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    logger.info("Compiling entropy summary...")

    output_dir = Path(output_dir)
    spectra_types = spectra_types or ["A", "B", "C", "D"]
    levels = levels or [3, 2, 1]
    maps_dir = Path(maps_dir) if maps_dir else output_dir / "maps"

    def _read_entropy_from_dir(subdir: Path, spectra: str, level: int) -> list[dict]:
        """Read mean entropy per ROI from all .nc files in a subdir."""
        rows = []
        nc_files = sorted(subdir.glob(f"roi_*_spectra{spectra}_L{level}.nc"))
        for nc_file in nc_files:
            try:
                ds = xr.open_dataset(nc_file)
                if "entropy" not in ds.data_vars:
                    ds.close()
                    continue
                ent = ds["entropy"].values
                roi_id = ds.attrs.get("roi_ID", nc_file.stem)
                source = ds.attrs.get("source_file", subdir.name)
                ds.close()
                valid = ent[~np.isnan(ent)]
                if len(valid) > 0:
                    rows.append({
                        "roi_id": roi_id,
                        "source_file": source,
                        "mean_entropy": float(np.mean(valid)),
                        "std_entropy": float(np.std(valid)),
                        "n_pixels": int(len(valid)),
                    })
            except Exception as e:
                logger.warning(f"Could not read {nc_file.name}: {e}")
        return rows

    records = []

    for spectra in spectra_types:
        for level in levels:

            # ── Main dataset maps ──────────────────────────────────────────
            main_subdir = maps_dir / f"spectra_{spectra}"
            if main_subdir.exists():
                main_rows = _read_entropy_from_dir(main_subdir, spectra, level)
                if main_rows:
                    main_df = pd.DataFrame(main_rows)
                    records.append({
                        "spectra": spectra,
                        "level": level,
                        "source": "main_dataset",
                        "n_rois": len(main_df),
                        "mean_entropy": round(main_df["mean_entropy"].mean(), 4),
                        "std_entropy": round(main_df["mean_entropy"].std(), 4),
                        "median_entropy": round(main_df["mean_entropy"].median(), 4),
                    })

            # ── Held-out maps ──────────────────────────────────────────────
            if held_out_stems:
                stems = [s for s in held_out_stems
                         if f"spectra{spectra}".lower() in s.lower()
                         or f"spectra_{spectra}".lower() in s.lower()]
            else:
                # Auto-discover: match 'held_out*' (covers both old turf_held_out_*
                # and new held_out_*) for the correct spectra type
                stems = [d.name for d in maps_dir.iterdir()
                         if d.is_dir() and "held_out" in d.name.lower()
                         and (f"spectra{spectra}".lower() in d.name.lower()
                              or f"spectra_{spectra}".lower() in d.name.lower())]

            held_rows = []
            for stem in stems:
                subdir = maps_dir / stem
                if subdir.exists():
                    held_rows.extend(_read_entropy_from_dir(subdir, spectra, level))

            if held_rows:
                held_df = pd.DataFrame(held_rows)
                records.append({
                    "spectra": spectra,
                    "level": level,
                    "source": "held_out",
                    "n_rois": len(held_df),
                    "mean_entropy": round(held_df["mean_entropy"].mean(), 4),
                    "std_entropy": round(held_df["mean_entropy"].std(), 4),
                    "median_entropy": round(held_df["mean_entropy"].median(), 4),
                })

    summary = pd.DataFrame(records)

    # ── Print comparison table ─────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("Entropy summary (normalised [0,1] — higher = more uncertain)")
    print(f"{'=' * 72}")
    print(
        f"{'Spectra':<10} {'Level':<8} {'Main mean':>10} {'Held-out':>10} "
        f"{'Δ entropy':>10} {'N held':>8}"
    )
    print("-" * 72)

    for spectra in spectra_types:
        for level in levels:
            main_row = summary[(summary["spectra"] == spectra) &
                               (summary["level"] == level) &
                               (summary["source"] == "main_dataset")]
            held_row = summary[(summary["spectra"] == spectra) &
                               (summary["level"] == level) &
                               (summary["source"] == "held_out")]

            main_str = (f"{main_row['mean_entropy'].iloc[0]:.4f}"
                        if not main_row.empty else "n/a")
            held_str = (f"{held_row['mean_entropy'].iloc[0]:.4f}"
                        if not held_row.empty else "n/a")
            n_str = f"{int(held_row['n_rois'].iloc[0])}" if not held_row.empty else "n/a"

            delta_str = "n/a"
            flag = ""
            if main_str != "n/a" and held_str != "n/a":
                delta = float(held_str) - float(main_str)
                delta_str = f"{delta:+.4f}"
                if delta > 0.05:
                    flag = "  <- ⚠ higher uncertainty on held-out"
                elif delta > 0.02:
                    flag = "  <- note"

            print(
                f"  {spectra:<8} {level:<8} {main_str:>10} {held_str:>10} "
                f"{delta_str:>10} {n_str:>8}{flag}"
            )

    print(f"{'=' * 72}\n")

    if out_path is None:
        out_path = output_dir / "entropy_summary.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info(f"Entropy summary saved: {out_path}")

    return summary