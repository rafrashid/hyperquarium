"""
hac_pipeline/features/feature_selector.py
------------------------------------------
Select the top-N features for HAC clustering using mean |SHAP| importance
computed from Level 4 XGBoost models, filtered to turf_algae classes only.

Hard dependency: raises FileNotFoundError if the SHAP file is missing.
No fallback — the HAC pipeline requires explicit feature selection from
the XGBoost pipeline to ensure cross-pipeline consistency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SHAP_FILENAME = "feature_importance_shap.csv"


def select_features(
        shap_dir: Path,
        n_top: int,
        turf_substring: str,
        output_path: Path,
) -> list[str]:
    """Select top-N features by mean |SHAP| across turf_algae Level 4 classes.

    Parameters
    ----------
    shap_dir : Path
        Directory containing feature_importance_shap.csv
        (typically outputs/spectra_{X}/level_4/).
    n_top : int
        Number of top features to retain.
    turf_substring : str
        Substring used to identify turf_algae columns in the SHAP file
        (e.g. 'turf_algae').
    output_path : Path
        Where to save selected_features.json.

    Returns
    -------
    list[str]
        Ordered list of selected feature names (most important first).

    Raises
    ------
    FileNotFoundError
        If feature_importance_shap.csv does not exist in shap_dir.
    ValueError
        If no turf_algae columns are found in the SHAP file.
    """
    shap_path = shap_dir / SHAP_FILENAME
    if not shap_path.exists():
        raise FileNotFoundError(
            f"SHAP feature importance file not found: {shap_path}\n"
            f"Run the xgb_pipeline SHAP stage for this spectra type before HAC.\n"
            f"Expected: outputs/spectra_{{X}}/level_4/{SHAP_FILENAME}"
        )

    logger.info(f"Loading SHAP importance: {shap_path}")
    shap_df = pd.read_csv(shap_path, index_col=0)
    # Expected shape: (n_features, n_level4_classes)
    # Columns: {feature}__class{c} — one per Level 4 class

    # Filter columns to turf_algae classes only
    turf_cols = [c for c in shap_df.columns if turf_substring in c]
    if not turf_cols:
        raise ValueError(
            f"No columns containing '{turf_substring}' found in {shap_path}.\n"
            f"Available column sample: {list(shap_df.columns[:5])}"
        )

    logger.info(
        f"Found {len(turf_cols)} turf_algae class columns out of "
        f"{shap_df.shape[1]} total in SHAP file."
    )

    # Mean |SHAP| per feature across turf columns
    importance = shap_df[turf_cols].mean(axis=1).sort_values(ascending=False)

    selected = importance.head(n_top)
    selected_features = selected.index.tolist()

    logger.info(
        f"Selected top {len(selected_features)} features. "
        f"Top 5: {selected_features[:5]}"
    )

    # Save audit record
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "source_shap_file": str(shap_path),
        "n_turf_classes": len(turf_cols),
        "n_top_features": n_top,
        "features": [
            {"name": feat, "mean_abs_shap": float(importance[feat])}
            for feat in selected_features
        ],
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Selected features saved: {output_path}")

    return selected_features


def validate_features_in_dataframe(
        features: list[str],
        df: pd.DataFrame,
) -> list[str]:
    """Check all selected features exist as columns in df.

    Returns the validated feature list. Raises ValueError if any are missing.
    """
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} selected features not found in DataFrame columns.\n"
            f"Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}\n"
            f"Check that the parquet and SHAP file are from the same pipeline run."
        )
    logger.info(f"All {len(features)} selected features validated in DataFrame.")
    return features
