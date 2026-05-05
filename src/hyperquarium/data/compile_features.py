"""
compile_features.py
====================
Compiles spectrum, GLCM, and spectral diversity features into a single
pixel-level DataFrame ready for XGBoost.

One row = one pixel, identified by (roi_ID, line, sample).

File naming
-----------
All feature files share the stem:  {roi_ID}_{RESAMPLE_METHOD}-{BLOCK_SIZE}
Defaults: RESAMPLE_METHOD="bilinear", BLOCK_SIZE="1x1"

Expected directory layout
--------------------------
data/
  {stem}.nc                          ← spectrum DataArray
  {stem}_specdiv/run*.nc             ← specdiv spatial Datasets (all non-empty runs loaded, integer-numbered)

output/
  {stem}_contrast.nc
  {stem}_energy.nc
  {stem}_entropy.nc
  {stem}_homogeneity.nc
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration — edit these to match your setup
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
SAVE_PATH = Path("compiled_features.parquet")  # or .csv if preferred

GLCM_FEATURES = ["contrast", "energy", "entropy", "homogeneity"]

# Resampling parameters — controls the filename stem used to locate all feature files.
# Pattern: {roi_ID}_{RESAMPLE_METHOD}-{BLOCK_SIZE}.nc
# Change these if you generate features at a different resolution or method.
RESAMPLE_METHOD: str = "bilinear"
BLOCK_SIZE: str = "1x1"

# Spectral diversity variables to extract from each run Dataset
# Set to None to extract ALL variables automatically
SPECDIV_VARS: list[str] | None = None

# Spectral band selection and interpolation (applied in load_spectrum).
# Bands are selected by wavelength value using the 'wavelength' coordinate.
# Set WAVELENGTH_MIN / WAVELENGTH_MAX to None to skip band selection.
# Set WAVELENGTH_STEP to None to skip interpolation.
WAVELENGTH_MIN: float | None = 425.0  # nm
WAVELENGTH_MAX: float | None = 705.0  # nm
WAVELENGTH_STEP: int | None = 2  # nm — interpolation interval

# Labels to include in the final dataset.
# e.g. ["turf_algae", "not_turf_algae"] — set to [] to include all labels.
INCLUDE_LABELS: list[str] = []

# ---------------------------------------------------------------------------


def file_stem(roi_id: str, resample_method: str = RESAMPLE_METHOD, block_size: str = BLOCK_SIZE) -> str:
    """Return the shared filename stem for all feature files of a given ROI.

    Pattern: {roi_ID}_{resample_method}-{block_size}
    Example: 20240723-142628-05_bilinear-1x1
    """
    return f"{roi_id}_{resample_method}-{block_size}"


def discover_roi_ids(data_dir: Path, output_dir: Path) -> list[str]:
    """Find all ROI IDs that have at least a spectrum file."""
    stem_suffix = f"_{RESAMPLE_METHOD}-{BLOCK_SIZE}.nc"
    pattern = re.compile(rf"^(.+){re.escape(stem_suffix)}$")
    roi_ids = []
    for f in sorted(data_dir.glob(f"*{stem_suffix}")):
        m = pattern.match(f.name)
        if m:
            roi_ids.append(m.group(1))
    return roi_ids


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

def load_spectrum(roi_id: str, data_dir: Path) -> pd.DataFrame:
    """
    Load spectrum DataArray (line, band, sample) and reshape to a DataFrame
    with columns [line, sample, band_0 … band_N] plus metadata from attrs.

    Checks that:
      - dims are exactly {"line", "band", "sample"}
      - len(line) == len(sample)  (square spatial footprint expected)
    Auto-transposes to (line, band, sample) if the order differs.
    """
    path = data_dir / f"{file_stem(roi_id)}.nc"
    da = xr.open_dataarray(path)

    # --- label check — skip ROI early if label not in INCLUDE_LABELS ---
    label = da.attrs.get("label")
    if INCLUDE_LABELS and label not in INCLUDE_LABELS:
        print(f"  [SKIP] {roi_id}: label '{label}' not in INCLUDE_LABELS")
        da.close()
        return None

    # --- dim presence check ---
    expected_dims = {"line", "band", "sample"}
    actual_dims = set(da.dims)
    if actual_dims != expected_dims:
        raise ValueError(
            f"[{roi_id}] Unexpected dims {actual_dims}. Expected {expected_dims}."
        )

    # --- ensure canonical order (line, band, sample) ---
    if da.dims != ("line", "band", "sample"):
        print(f"  [INFO] {roi_id}: transposing dims {da.dims} → (line, band, sample)")
        da = da.transpose("line", "band", "sample")

    # --- spatial square check ---
    n_line, n_sample = da.sizes["line"], da.sizes["sample"]
    if n_line != n_sample:
        raise ValueError(
            f"[{roi_id}] line ({n_line}) ≠ sample ({n_sample}). "
            "Expected a square spatial footprint."
        )

    # --- wavelength band selection ---
    if WAVELENGTH_MIN is not None or WAVELENGTH_MAX is not None:
        if "wavelength" not in da.coords:
            raise ValueError(f"[{roi_id}] No 'wavelength' coordinate found on 'band' dim.")
        wl_min = WAVELENGTH_MIN if WAVELENGTH_MIN is not None else float(da.wavelength.min())
        wl_max = WAVELENGTH_MAX if WAVELENGTH_MAX is not None else float(da.wavelength.max())
        mask = (da.wavelength >= wl_min) & (da.wavelength <= wl_max)
        da = da.isel(band=mask.values)

    # --- wavelength interpolation ---
    if WAVELENGTH_STEP is not None:
        if "wavelength" not in da.coords:
            raise ValueError(f"[{roi_id}] No 'wavelength' coordinate found for interpolation.")
        wl_min = WAVELENGTH_MIN if WAVELENGTH_MIN is not None else float(da.wavelength.min())
        wl_max = WAVELENGTH_MAX if WAVELENGTH_MAX is not None else float(da.wavelength.max())
        wl_grid = np.arange(wl_min, wl_max + WAVELENGTH_STEP / 2, WAVELENGTH_STEP)
        # Swap band index to wavelength values, interpolate onto fixed grid
        da = da.assign_coords(band=da.wavelength.values).interp(
            band=wl_grid, method="linear"
        )

    # Stack (line, sample) → pixel, leaving band as columns
    da_stacked = da.stack(pixel=("line", "sample"))  # (band, pixel)
    df = da_stacked.to_pandas().T  # (pixel, band)
    df.index.names = ["line", "sample"]

    # Name columns by wavelength: band dim holds wavelength values after interp,
    # or use the wavelength coord if no interpolation was done
    if WAVELENGTH_STEP is not None:
        # band dim was replaced with wavelength values during interp
        df.columns = [f"{int(b)}_nm" for b in da.band.values]
    elif "wavelength" in da.coords:
        df.columns = [f"{int(da.wavelength.sel(band=b).values)}_nm" for b in da.band.values]
    else:
        df.columns = [f"band_{int(b)}" for b in da.band.values]

    df = df.reset_index()

    # Attach metadata from attrs
    attrs = da.attrs
    df["roi_ID"] = attrs.get("roi_ID", roi_id)
    df["label"] = attrs.get("label", np.nan)
    df["dataset"] = attrs.get("dataset", np.nan)
    df["scan_ID"] = attrs.get("scan_ID", np.nan)
    df["exposure"] = attrs.get("exposure", np.nan)
    df["n_valid_pixels"] = attrs.get("n_valid_pixels", np.nan)

    da.close()
    return df


# ---------------------------------------------------------------------------
# GLCM
# ---------------------------------------------------------------------------

def load_glcm(roi_id: str, output_dir: Path) -> pd.DataFrame:
    """
    Load all four GLCM feature Datasets for this ROI.
    Each Dataset has variables window_7, window_9, … window_51.
    Returns a wide DataFrame indexed by (line, sample) with columns:
        {feature}_window_{size}   e.g. homogeneity_window_7
    """
    dfs = []
    for feature in GLCM_FEATURES:
        path = output_dir / f"{file_stem(roi_id)}_{feature}.nc"
        if not path.exists():
            print(f"  [WARN] GLCM file missing: {path}")
            continue

        ds = xr.open_dataset(path)
        rows = []
        for var in ds.data_vars:
            # var names are window_7, window_9, … — extract size
            m = re.match(r"window_(\d+)", var)
            if not m:
                continue
            size = int(m.group(1))
            col = f"{feature}_window_{size}"
            da = ds[var]
            sub = da.to_dataframe(name=col)[col].reset_index()
            rows.append(sub.set_index(["line", "sample"])[[col]])

        if rows:
            dfs.append(pd.concat(rows, axis=1))
        ds.close()

    if not dfs:
        return pd.DataFrame()

    glcm_df = pd.concat(dfs, axis=1).reset_index()
    return glcm_df


# ---------------------------------------------------------------------------
# Spectral diversity
# ---------------------------------------------------------------------------

def _iter_run_files(roi_id: str, data_dir: Path):
    """Yield (run_number, Path) for all run*.nc files, in natural numeric order."""
    run_dir = data_dir / f"{file_stem(roi_id)}_specdiv"
    if not run_dir.exists():
        return
    for p in sorted(
            run_dir.glob("run*.nc"),
            key=lambda p: int(re.search(r"run(\d+)\.nc$", p.name).group(1))
    ):
        yield int(re.search(r"run(\d+)\.nc$", p.name).group(1)), p


def _run_is_empty(ds: xr.Dataset) -> bool:
    """Return True if every data variable in the Dataset is all-NaN."""
    return all(
        bool(ds[v].isnull().all().values)
        for v in ds.data_vars
    )


def load_specdiv(roi_id: str, data_dir: Path) -> pd.DataFrame:
    """
    Load ALL non-empty specdiv run Datasets for this ROI.

    Each run's variables are read from the global attribute 'fact' (plot size)
    and named: sdiv_{var}_plot_{fact}
    e.g. sdiv_alpha_sdiv_plot_7, sdiv_beta_lcsd_plot_101

    Runs where every variable is entirely NaN are skipped.
    Returns a wide DataFrame with columns for every (var × plot_size) combination,
    joined on (line, sample).
    """
    all_frames = []

    for run_num, run_path in _iter_run_files(roi_id, data_dir):
        ds = xr.open_dataset(run_path)

        if _run_is_empty(ds):
            print(f"  [SKIP] {roi_id} run {run_num}: all NaN — skipping")
            ds.close()
            continue

        fact = ds.attrs.get("fact")
        if fact is None:
            print(f"  [WARN] {roi_id} run {run_num}: no 'fact' attribute — skipping")
            ds.close()
            continue

        vars_to_use = SPECDIV_VARS if SPECDIV_VARS else list(ds.data_vars)
        frames = []
        for var in vars_to_use:
            if var not in ds:
                continue
            col = f"sdiv_{var}_plot_{fact}"
            sub = ds[var].to_dataframe(name=col)[col].reset_index()
            frames.append(sub.set_index(["line", "sample"])[[col]])

        ds.close()

        if frames:
            all_frames.append(pd.concat(frames, axis=1))

    if not all_frames:
        print(f"  [WARN] No valid specdiv runs found for {roi_id}")
        return pd.DataFrame()

    # Join all plot sizes on (line, sample)
    result = all_frames[0]
    for frame in all_frames[1:]:
        result = result.join(frame, how="outer")

    return result.reset_index()


# ---------------------------------------------------------------------------
# Per-ROI assembly
# ---------------------------------------------------------------------------

def compile_roi(roi_id: str, data_dir: Path, output_dir: Path) -> pd.DataFrame | None:
    """Compile all features for a single ROI into a pixel-level DataFrame."""
    try:
        spectrum_df = load_spectrum(roi_id, data_dir)
    except Exception as e:
        print(f"  [ERROR] Spectrum load failed for {roi_id}: {e}")
        return None

    # Use (line, sample) as the join key
    df = spectrum_df.copy()

    # --- GLCM ---
    glcm_df = load_glcm(roi_id, output_dir)
    if not glcm_df.empty:
        df = df.merge(glcm_df, on=["line", "sample"], how="left")
    else:
        print(f"  [WARN] No GLCM features for {roi_id}")

    # --- Spectral diversity ---
    sdiv_df = load_specdiv(roi_id, data_dir)
    if not sdiv_df.empty:
        # Reindex to spectrum grid if coords differ (same coords per spec,
        # but we use a left-merge to be safe)
        df = df.merge(sdiv_df, on=["line", "sample"], how="left")
    else:
        print(f"  [WARN] No specdiv features for {roi_id}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compile_all(
        data_dir: Path = DATA_DIR,
        output_dir: Path = OUTPUT_DIR,
        save_path: Path = SAVE_PATH,
        include_labels: list[str] = INCLUDE_LABELS,
) -> pd.DataFrame:
    roi_ids = discover_roi_ids(data_dir, output_dir)
    print(f"Found {len(roi_ids)} ROIs: {roi_ids[:5]}{'...' if len(roi_ids) > 5 else ''}")
    if include_labels:
        print(f"Including only labels: {include_labels}")

    all_dfs = []
    for roi_id in tqdm(roi_ids, desc="Compiling ROIs"):
        roi_df = compile_roi(roi_id, data_dir, output_dir)
        if roi_df is not None:
            all_dfs.append(roi_df)

    if not all_dfs:
        raise RuntimeError("No ROIs compiled — check your data/output paths.")

    final_df = pd.concat(all_dfs, ignore_index=True)

    # Reorder: metadata first, then features, label last
    meta_cols = ["roi_ID", "scan_ID", "dataset", "exposure", "n_valid_pixels",
                 "line", "sample"]
    # Spectrum cols: wavelength-named (wl_425.0) or band-indexed (band_0)
    band_cols = sorted(
        [c for c in final_df.columns if c.endswith("_nm")],
        key=lambda x: int(x.replace("_nm", ""))
    ) or sorted(
        [c for c in final_df.columns if c.startswith("band_")],
        key=lambda x: int(x.split("_")[1])
    )
    glcm_cols = sorted([c for c in final_df.columns
                        if any(c.startswith(f) for f in GLCM_FEATURES)])
    sdiv_cols = sorted([c for c in final_df.columns if c.startswith("sdiv_")])
    label_col = ["label"]

    ordered_cols = (
            [c for c in meta_cols if c in final_df.columns] +
            band_cols + glcm_cols + sdiv_cols +
            [c for c in label_col if c in final_df.columns]
    )
    # Append any remaining columns not explicitly ordered
    remaining = [c for c in final_df.columns if c not in ordered_cols]
    final_df = final_df[ordered_cols + remaining]

    # Save
    if str(save_path).endswith(".parquet"):
        final_df.to_parquet(save_path, index=False)
    else:
        final_df.to_csv(save_path, index=False)

    print(f"\nDone. Shape: {final_df.shape}")
    print(f"Saved to: {save_path}")
    print(f"\nColumn groups:")
    print(f"  Metadata : {len([c for c in meta_cols if c in final_df.columns])}")
    wl_label = (f"{WAVELENGTH_MIN}–{WAVELENGTH_MAX} nm @ {WAVELENGTH_STEP} nm"
                if WAVELENGTH_STEP is not None else f"{WAVELENGTH_MIN}–{WAVELENGTH_MAX} nm")
    print(f"  Spectrum : {len(band_cols)} bands ({wl_label})")
    print(f"  GLCM     : {len(glcm_cols)} features")
    print(f"  Specdiv  : {len(sdiv_cols)} features")
    print(f"  Label    : {final_df['label'].nunique()} classes → {final_df['label'].unique().tolist()}")

    return final_df


if __name__ == "__main__":
    df = compile_all()

    # Quick sanity checks
    print("\n--- Sanity checks ---")
    print(df.dtypes.value_counts())
    print(f"NaN summary:\n{df.isna().sum()[df.isna().sum() > 0]}")
    print(df.head(2).T)