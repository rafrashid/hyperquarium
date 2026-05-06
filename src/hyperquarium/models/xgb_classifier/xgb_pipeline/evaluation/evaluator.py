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

    # Average precision (AUC-PR) — only for classes present in y_true
    if n_classes_present == 2 and len(class_names) == 2:
        ap_scores = {class_names[1]: average_precision_score(y_true, y_pred_proba)}
    else:
        y_bin = label_binarize(y_true, classes=labels_present)
        ap_scores = {
            names_present[i]: average_precision_score(y_bin[:, i], y_pred_proba[:, labels_present[i]])
            for i in range(n_classes_present)
        }

    metrics = {
        "macro_f1": round(macro_f1, 5),
        "average_precision_per_class": {k: round(v, 5) for k, v in ap_scores.items()},
        "per_class_f1": {
            class_names[i]: round(
                f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0), 5
            )
            for i in range(n_classes)
        },
    }
    save_json(metrics, out_dir / "metrics.json")
    logger.info(f"Macro F1: {macro_f1:.4f}")
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
        eval_metric: str,
        out_dir: Path,
) -> None:
    """
    Plots train vs validation metric per boosting round (from evals_result).
    Used to detect overfitting and confirm early stopping point.

    Args:
        evals_result: Dict returned by xgb.train() via evals_result parameter.
        eval_metric:  Metric name (e.g. 'mlogloss', 'aucpr').
        out_dir:      Output directory.
    """
    train_vals = evals_result.get("train", {}).get(eval_metric, [])
    val_vals = evals_result.get("val", {}).get(eval_metric, [])

    if not train_vals:
        logger.warning("No evals_result data found — skipping learning curve plot.")
        return

    rounds = np.arange(1, len(train_vals) + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rounds, train_vals, label="Train", linewidth=1.2)
    ax.plot(rounds, val_vals, label="Validation", linewidth=1.2)
    ax.axvline(np.argmin(val_vals) + 1, color="red", linestyle="--", linewidth=1, label="Best round")
    ax.set(xlabel="Boosting round", ylabel=eval_metric, title=f"Learning curve — {eval_metric}")
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

    return metrics