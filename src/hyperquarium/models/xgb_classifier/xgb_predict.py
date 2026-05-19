"""
predict.py
PBS-ready prediction and spatial reprojection script.
Loads a trained model, predicts class probabilities for every pixel,
and writes one NetCDF file per ROI containing:
  - One xr.DataArray per class (predicted probability, dims: line x sample)
  - One xr.DataArray of predicted class labels (string dtype, dims: line x sample)
  - ROI metadata as dataset attributes
  - Proportion of correct classifications as an attribute (if true labels available)

Usage:
    python3 predict.py <data_path> <level> <weighted> [--labelset pilot]

PBS usage:
    module load python3/3.14.4
    python3 scripts/predict.py data/spectra_A.parquet 3 true --labelset reefcompare
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict and reproject to spatial NetCDF maps."
    )
    parser.add_argument("data_path", type=Path)
    parser.add_argument("level", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("weighted", type=str)
    parser.add_argument("--labelset", type=str, default="pilot")
    parser.add_argument("--spectra", type=str, required=True,
                        help="Spectra type label (A, B, C, or D) — identifies which model to load")
    parser.add_argument("--compact", action="store_true",
                        help="Save only predicted_class, max_probability, entropy "
                             "(skips per-class prob arrays; auto-enabled for Level 4)")
    parser.add_argument("--no-subsample", action="store_true",
                        help="Skip turf ROI subsampling (use for held-out data)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory for NetCDF files (default: outputs/maps/)")
    return parser.parse_args()


def parse_weighted(val: str) -> bool:
    if val.strip().lower() in ("true", "1", "yes"):  return True
    if val.strip().lower() in ("false", "0", "no"):  return False
    raise ValueError(f"weighted must be true/false, got: '{val}'")


def make_netcdf(
        roi_id: str,
        roi_df,
        proba: "np.ndarray",
        le: "LabelEncoder",
        n_classes: int,
        label_col: str,
        meta_cols: list[str],
        spectra: str,
        level: int,
        weighted: bool,
        out_dir: Path,
        source_file: str = "",
        compact: bool = False,
) -> None:
    """Builds and saves one xr.Dataset for a single ROI."""
    import numpy as np
    import xarray as xr

    lines = roi_df["line"].values.astype(int)
    samples = roi_df["sample"].values.astype(int)

    line_min, line_max = lines.min(), lines.max()
    sample_min, sample_max = samples.min(), samples.max()
    n_lines = line_max - line_min + 1
    n_samples = sample_max - sample_min + 1

    # Row indices into the 2D grid
    li = lines - line_min
    si = samples - sample_min

    class_names = list(le.classes_)

    # Predicted class index per pixel
    if proba.ndim == 1:
        # Binary — proba is P(class 1)
        pred_idx = (proba >= 0.5).astype(int)
    else:
        pred_idx = proba.argmax(axis=1)

    data_vars = {}

    if not class_names:
        raise ValueError(f"class_names is empty for ROI {roi_id} — check LabelEncoder.")
    if len(proba.shape) > 1 and proba.shape[1] != len(class_names):
        raise ValueError(
            f"proba shape {proba.shape} does not match n_classes={len(class_names)} "
            f"for ROI {roi_id}."
        )
    if n_lines <= 0 or n_samples <= 0:
        raise ValueError(
            f"Invalid grid dimensions ({n_lines} x {n_samples}) for ROI {roi_id}. "
            f"Check line/sample columns."
        )

    # Predicted class label (string dtype) — always saved
    str_grid = np.full((n_lines, n_samples), "", dtype=object)
    str_grid[li, si] = np.array(class_names)[pred_idx]
    data_vars["predicted_class"] = xr.DataArray(
        str_grid.astype(str),
        dims=["line", "sample"],
        attrs={"long_name": "Predicted class label"},
    )

    # Max probability (confidence of predicted class) — always saved
    if proba.ndim == 1:
        max_p = np.where(pred_idx == 1, proba, 1 - proba)
    else:
        max_p = proba.max(axis=1)
    max_grid = np.full((n_lines, n_samples), np.nan, dtype=np.float32)
    max_grid[li, si] = max_p.astype(np.float32)
    data_vars["max_probability"] = xr.DataArray(
        max_grid, dims=["line", "sample"],
        attrs={"long_name": "Predicted class probability (confidence)",
               "units": "probability"},
    )

    # Entropy — normalised by log(n_classes) so range is [0, 1] — always saved
    if proba.ndim == 1:
        p_arr = np.column_stack([1 - proba, proba])
    else:
        p_arr = proba
    entropy_vals = -np.sum(p_arr * np.log(p_arr + 1e-10), axis=1)
    entropy_norm = entropy_vals / np.log(len(class_names) + 1e-10)
    ent_grid = np.full((n_lines, n_samples), np.nan, dtype=np.float32)
    ent_grid[li, si] = entropy_norm.astype(np.float32)
    data_vars["entropy"] = xr.DataArray(
        ent_grid, dims=["line", "sample"],
        attrs={"long_name": "Prediction entropy (normalised)",
               "units": "dimensionless", "range": "[0, 1]"},
    )

    # Per-class probability arrays — only in non-compact mode
    if not compact:
        for c_idx, cls_name in enumerate(class_names):
            grid = np.full((n_lines, n_samples), np.nan, dtype=np.float32)
            if proba.ndim == 1:
                p = proba if c_idx == 1 else 1 - proba
            else:
                p = proba[:, c_idx]
            grid[li, si] = p.astype(np.float32)
            data_vars[f"prob_{cls_name}"] = xr.DataArray(
                grid, dims=["line", "sample"],
                attrs={"long_name": f"Predicted probability — {cls_name}",
                       "units": "probability"},
            )

    # Proportion correct (if true labels available and known to the encoder)
    prop_correct = np.nan
    if label_col in roi_df.columns:
        try:
            true_encoded = le.transform(roi_df[label_col].values)
            prop_correct = float((pred_idx == true_encoded).mean())
        except ValueError:
            # Labels unseen during training (e.g. held-out Level 4 ROIs)
            prop_correct = np.nan

    # Metadata attrs
    attrs = {
        "roi_ID": roi_id,
        "spectra": spectra,
        "level": level,
        "weighted": str(weighted),
        "class_mapping": json.dumps({str(i): str(c) for i, c in enumerate(class_names)}),
        "prop_correct": float(round(prop_correct, 4)),
        "n_valid_pixels": int((~np.isnan(data_vars["max_probability"].values)).sum()),
        "source_file": str(source_file),
        "line_min": int(line_min),
        "line_max": int(line_max),
        "sample_min": int(sample_min),
        "sample_max": int(sample_max),
    }
    for col in meta_cols:
        if col in roi_df.columns:
            vals = roi_df[col].dropna().unique()
            # Cast to plain Python str — NetCDF4 does not accept np.str_
            attrs[col] = str(vals[0]) if len(vals) == 1 else [str(v) for v in vals]

    ds = xr.Dataset(data_vars, attrs=attrs)

    safe_roi = roi_id.replace("/", "_").replace(" ", "_")
    out_path = out_dir / f"roi_{safe_roi}_spectra{spectra}_L{level}.nc"
    ds.to_netcdf(out_path)


def main() -> None:
    import numpy as np

    args = parse_args()
    weighted = parse_weighted(args.weighted)
    level = args.level
    data_path = Path(args.data_path)
    spectra = args.spectra.upper()

    from utils.logger import get_logger
    from config.config import OUTPUT_DIR, LOG_DIR, LABEL_COLUMNS, LEVEL_CONFIGS, METADATA_COLUMNS
    run_id = f"predict_spectra_{spectra}_level{level}_{'weighted' if weighted else 'unweighted'}"
    logger = get_logger(run_id, LOG_DIR)

    logger.info("=" * 60)
    logger.info(f"PREDICT  {run_id}")
    logger.info(f"  data_path : {data_path}")
    logger.info(f"  level     : {level}")
    logger.info(f"  weighted  : {weighted}")
    logger.info(f"  labelset  : {args.labelset}")
    logger.info("=" * 60)

    from config.config import SPLIT, TURF_ALGAE_CLASS, ROI_ID_COLUMN
    from data.loader import (load_spectra, remap_labels, subsample_turf_rois,
                             get_feature_columns, encode_labels, make_dmatrix,
                             )
    from models.trainer import load_model
    from utils.io import make_output_dir

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    lvl_cfg = LEVEL_CONFIGS[level]
    model_dir = make_output_dir(OUTPUT_DIR, spectra, level, weighted)
    model_path = model_dir / "model.json"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path} — run train.py first.")
        sys.exit(1)

    # Subdirectory named after source parquet file — prevents mixing main and held-out maps
    source_stem = data_path.stem
    base_maps = Path(args.out_dir) if args.out_dir else Path(OUTPUT_DIR) / "maps"
    out_dir = base_maps / source_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # ---- Load full dataset (no split — predict all pixels) ----------------
    df = load_spectra(data_path)
    label_col = LABEL_COLUMNS[level]

    # Remap labels only if the raw label column exists (skipped for unlabelled data)
    if ROI_ID_COLUMN in df.columns and label_col not in df.columns:
        from config.config import RAW_LABEL_COLUMN
        if RAW_LABEL_COLUMN in df.columns:
            df = remap_labels(df, dataset=args.labelset)
            logger.info("Label remapping applied.")
        else:
            logger.info("No raw label column found — skipping remapping (unseen data).")
    elif label_col in df.columns:
        logger.info(f"Label column '{label_col}' already present — skipping remapping.")

    # Apply subsampling unless --no-subsample is set (e.g. for held-out turf data)
    if not args.no_subsample:
        df = subsample_turf_rois(df, spectra=spectra, random_seed=42)
        logger.info("Turf ROI subsampling applied.")
    else:
        logger.info("Turf ROI subsampling skipped (--no-subsample).")

    feature_cols = get_feature_columns(df)

    # Level 4 dynamic n_classes
    if level == 4:
        # Read n_classes from roi_label_mapping.csv saved by train.py
        roi_map_path = model_dir / "roi_label_mapping.csv"
        if roi_map_path.exists():
            import pandas as _pd
            _roi_map = _pd.read_csv(roi_map_path)
            n_rois = _roi_map["label_level4"].nunique()
            LEVEL_CONFIGS[4].n_classes = n_rois
            logger.info(f"Level 4 n_classes loaded from roi_label_mapping.csv: {n_rois}")
        else:
            logger.warning(
                f"roi_label_mapping.csv not found at {roi_map_path} — "
                f"run train.py first. Falling back to df count."
            )
            n_rois = df[LABEL_COLUMNS[4]].nunique()
            LEVEL_CONFIGS[4].n_classes = n_rois

    # Reconstruct label encoder from saved training metadata — works on unseen data
    # that may not have a label column or may have fewer classes than training.
    from sklearn.preprocessing import LabelEncoder
    from utils.io import load_json
    meta = load_json(model_dir / "training_metadata.json")
    class_map = meta["class_mapping"]  # {str(int): class_name}
    classes = [class_map[str(i)] for i in range(len(class_map))]
    le = LabelEncoder()
    le.fit(classes)
    n_classes = len(classes)
    logger.info(f"Classes ({n_classes}) from training metadata: {classes}")

    # ---- Load model -------------------------------------------------------
    booster = load_model(model_path)

    # ---- Metadata columns to store as attrs ------------------------------
    meta_cols = [c for c in ["scan_ID", "exposure", "n_valid_pixels", "dataset"]
                 if c in df.columns]

    # Level 4 always compact (too many classes); otherwise follow --compact flag
    use_compact = args.compact or (level == 4)
    if use_compact:
        logger.info("Compact mode: saving predicted_class, max_probability, entropy only.")

    # ---- Predict and reproject per ROI ------------------------------------
    roi_col = ROI_ID_COLUMN
    rois = df[roi_col].unique()
    logger.info(f"Reprojecting {len(rois)} ROIs...")

    for i, roi_id in enumerate(rois, 1):
        roi_df = df[df[roi_col] == roi_id]

        # Build plain DMatrix (ref=False — no QuantileDMatrix needed for inference)
        dm = make_dmatrix(roi_df, feature_cols,
                          np.zeros(len(roi_df), dtype=int), ref=False)
        raw = booster.predict(dm)

        if n_classes == 2:
            proba = raw  # shape: (n_pixels,)
        else:
            proba = raw.reshape(-1, n_classes)  # shape: (n_pixels, n_classes)

        try:
            make_netcdf(
                roi_id=roi_id,
                roi_df=roi_df,
                proba=proba,
                le=le,
                n_classes=n_classes,
                label_col=label_col,
                meta_cols=meta_cols,
                spectra=spectra,
                level=level,
                weighted=weighted,
                out_dir=out_dir,
                source_file=source_stem,
                compact=use_compact,
            )
        except Exception as e:
            import traceback
            logger.error(f"  ROI {roi_id} failed: {type(e).__name__}: {e}")
            logger.error(traceback.format_exc())
            continue

        if i % 10 == 0 or i == len(rois):
            logger.info(f"  {i}/{len(rois)} ROIs written")

    logger.info(f"All NetCDF files saved to: {out_dir}")


if __name__ == "__main__":
    main()