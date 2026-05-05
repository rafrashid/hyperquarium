"""
texture.py
GLCM texture feature extraction from PCA score DataArrays.

Designed to work with the output of pca_dataarray() from specdiv.py, which
returns (ds, loadings_da, valid_da). The first element, ds, is an xr.Dataset
with one data variable per PC (e.g. "PC1", "PC2"), each a 2-D DataArray
with dims (line, sample), dtype float64, and NaN for masked pixels.

Key design decisions
--------------------
- PCA scores are continuous floats; GLCM requires uint8/uint16 integer images.
  Each PC band is independently quantised (percentile-clipped + rescaled) before
  GLCM computation.
- The inner loop over spatial windows is the bottleneck. Vectorisation with
  view_as_windows + numpy keeps memory reasonable while avoiding Python overhead.
- A high-level wrapper (glcm_texture_dataset) iterates over PCs and window
  sizes and returns a single xr.Dataset keyed as f"{feature}_PC{n}_w{size}".
"""

import logging
from typing import Sequence

import numpy as np
import xarray as xr
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_windows

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GLCM feature names — all supported by skimage.feature.graycoprops
# ---------------------------------------------------------------------------
GLCM_FEATURES = ("contrast", "dissimilarity", "homogeneity", "energy",
                 "correlation", "ASM", "mean", "variance", "std", "entropy")


# ---------------------------------------------------------------------------
# Quantisation helper
# ---------------------------------------------------------------------------

def quantise_band(band: np.ndarray,
                  levels: int = 64,
                  lower_pct: float = 2.0,
                  upper_pct: float = 98.0) -> np.ndarray:
    """
    Clip and linearly rescale a 2-D float array to [0, levels-1] uint8/uint16.

    NaN pixels are preserved as NaN in a float output; callers that need an
    integer array should mask NaNs before passing to graycomatrix.

    Parameters
    ----------
    band       : 2-D float array (line × sample).
    levels     : number of grey levels (≤ 256 → uint8, else uint16).
    lower_pct  : lower percentile for clipping (reduces outlier influence).
    upper_pct  : upper percentile for clipping.

    Returns
    -------
    Quantised array, dtype uint8 or uint16, same shape as input.
    NaN pixels are set to 0 (background) — track them separately.
    """
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)

    lo, hi = np.percentile(valid, [lower_pct, upper_pct])
    if hi == lo:
        return np.zeros_like(band, dtype=np.uint8)

    clipped = np.clip(band, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * (levels - 1)
    scaled = np.round(scaled)
    scaled[~np.isfinite(scaled)] = 0
    scaled = scaled.astype(np.uint8 if levels <= 256 else np.uint16)
    return scaled


# ---------------------------------------------------------------------------
# Core: rolling GLCM for a single 2-D band
# ---------------------------------------------------------------------------

def rolling_glcm(image: np.ndarray,
                 feature: str = "contrast",
                 window_size: int = 9,
                 distances: int | Sequence[int] = 1,
                 angles: Sequence[float] = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
                 levels: int = 64,
                 nan_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Compute a GLCM texture feature over a rolling window on a 2-D integer image.

    The output is smaller than the input by (window_size - 1) in each spatial
    dimension (valid convolution, no padding). Use ``glcm_texture_dataarray``
    for a version that pads back to the original shape.

    Parameters
    ----------
    image       : 2-D integer array (e.g. uint8), already quantised.
    feature     : GLCM property — one of GLCM_FEATURES.
    window_size : side length of the square sliding window (must be odd).
    distances   : pixel distance(s) for GLCM co-occurrence.
    angles      : list of angles (radians) for GLCM. Results are averaged.
    levels      : number of grey levels (must match quantisation).
    nan_mask    : boolean array, same shape as *image*, True where original
                  data was NaN. Those output pixels are set to NaN.

    Returns
    -------
    result : 2-D float32 array of shape
             (H - window_size + 1, W - window_size + 1).
    """
    if feature not in GLCM_FEATURES:
        raise ValueError(f"feature must be one of {GLCM_FEATURES}, got {feature!r}")

    h, w = image.shape
    if window_size > h or window_size > w:
        raise ValueError(
            f"window_size={window_size} exceeds image dimensions ({h}×{w}). "
            f"Maximum valid window_size is {min(h, w)}."
        )

    distances = [distances] if isinstance(distances, int) else list(distances)
    angles = list(angles)

    # view_as_windows: shape (out_h, out_w, window_size, window_size)
    windows = view_as_windows(image, window_shape=(window_size, window_size))
    out_h, out_w = windows.shape[:2]
    results = np.full((out_h, out_w), np.nan, dtype=np.float32)

    logger.info("Computing GLCM '%s', window=%d, shape %dx%d → %dx%d",
                feature, window_size, image.shape[0], image.shape[1], out_h, out_w)

    for i in range(out_h):
        for j in range(out_w):
            patch = windows[i, j]
            glcm = graycomatrix(patch, distances=distances,
                                angles=angles, levels=levels, normed=True)
            # graycoprops returns shape (n_distances, n_angles); average both
            results[i, j] = graycoprops(glcm, feature).mean()

    # Propagate NaN mask: if any pixel in a window was originally NaN, mark NaN
    if nan_mask is not None:
        nan_windows = view_as_windows(
            nan_mask.astype(np.uint8), window_shape=(window_size, window_size)
        )
        window_has_nan = nan_windows.any(axis=(-2, -1))  # (out_h, out_w)
        results[window_has_nan] = np.nan

    return results


# ---------------------------------------------------------------------------
# xarray wrapper: single PC band, single window size, all features
# ---------------------------------------------------------------------------

def glcm_texture_dataarray(pc_band: xr.DataArray,
                           window_size: int = 9,
                           features: Sequence[str] = GLCM_FEATURES,
                           distances: int | Sequence[int] = 1,
                           angles: Sequence[float] = (0, np.pi / 4,
                                                      np.pi / 2,
                                                      3 * np.pi / 4),
                           levels: int = 64,
                           lower_pct: float = 2.0,
                           upper_pct: float = 98.0) -> xr.Dataset:
    """
    Compute GLCM texture features for a single 2-D PC score DataArray.

    The output is padded back to the original (line, sample) shape using
    ``np.pad`` with edge mode so that coordinates are preserved.

    Parameters
    ----------
    pc_band     : 2-D xr.DataArray with dims (line, sample), float64.
                  Typically one slice from pca_dataarray() scores, e.g.
                      scores.sel(pc="PC1")
    window_size : side length of square sliding window (must be odd ≥ 3).
    features    : subset of GLCM_FEATURES to compute.
    distances   : GLCM distance(s).
    angles      : GLCM angles.
    levels      : number of grey levels for quantisation.
    lower_pct   : lower percentile for quantisation clipping.
    upper_pct   : upper percentile for quantisation clipping.

    Returns
    -------
    xr.Dataset with one variable per feature, each a 2-D DataArray
    with dims (line, sample) and the same coordinates as pc_band.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    values = pc_band.values  # (line, sample)
    n_lines, n_samples = values.shape
    max_window = min(n_lines, n_samples)
    if window_size > max_window:
        logger.warning(
            "window_size=%d exceeds the smallest image dimension (%d). "
            "Returning an all-NaN Dataset for this window size.",
            window_size, max_window,
        )
        nan_array = np.full(values.shape, np.nan, dtype=np.float32)
        return xr.Dataset({
            feat: xr.DataArray(
                nan_array.copy(),
                dims=pc_band.dims,
                coords=pc_band.coords,
                attrs={"glcm_feature": feat, "window_size": window_size,
                       "levels": levels, "skipped": True},
            )
            for feat in features
        })

    half = window_size // 2
    nan_mask = ~np.isfinite(values)

    # Quantise (NaN → 0 internally; tracked via nan_mask)
    quantised = quantise_band(values, levels=levels,
                              lower_pct=lower_pct, upper_pct=upper_pct)

    data_vars = {}
    for feat in features:
        raw = rolling_glcm(quantised, feature=feat,
                           window_size=window_size,
                           distances=distances, angles=angles,
                           levels=levels, nan_mask=nan_mask)

        # Pad back to original spatial shape (edge padding, then re-apply NaN)
        padded = np.pad(raw, pad_width=half, mode="edge")
        # Trim to exact original shape in case of rounding
        padded = padded[:values.shape[0], :values.shape[1]]
        padded[nan_mask] = np.nan

        data_vars[feat] = xr.DataArray(
            padded.astype(np.float32),
            dims=pc_band.dims,
            coords=pc_band.coords,
            attrs={"glcm_feature": feat, "window_size": window_size,
                   "levels": levels},
        )

    return xr.Dataset(data_vars)


# ---------------------------------------------------------------------------
# High-level: all PCs × all window sizes → dict of per-feature xr.Datasets
# ---------------------------------------------------------------------------

def glcm_texture_dataset(scores: xr.Dataset,
                         window_sizes: Sequence[int] = (9, 21),
                         features: Sequence[str] = GLCM_FEATURES,
                         distances: int | Sequence[int] = 1,
                         angles: Sequence[float] = (0, np.pi / 4,
                                                    np.pi / 2,
                                                    3 * np.pi / 4),
                         levels: int = 64,
                         lower_pct: float = 2.0,
                         upper_pct: float = 98.0,
                         pc_names: Sequence[str] | None = None) -> dict[str, xr.Dataset]:
    """
    Compute GLCM texture features across all PCs and window sizes.

    Wraps ``glcm_texture_dataarray`` in a loop over PCs and window sizes.
    Returns one Dataset per feature, with variables named ``window_{size}``
    (e.g. ``window_21``), making it easy to compare window sizes for a
    given feature.

    Parameters
    ----------
    scores      : xr.Dataset returned as the first element of pca_dataarray(),
                  with one data variable per PC (e.g. "PC1", "PC2"), each a
                  2-D DataArray with dims (line, sample).
    window_sizes: iterable of window sizes to compute (all must be odd ≥ 3).
    features    : subset of GLCM_FEATURES.
    distances   : GLCM pixel distance(s).
    angles      : GLCM angles (radians).
    levels      : number of grey levels.
    lower_pct   : lower percentile for per-band quantisation clipping.
    upper_pct   : upper percentile for per-band quantisation clipping.
    pc_names    : optional subset of PC variable names to process
                  (e.g. ["PC1", "PC2"]). Defaults to all variables in scores.

    Returns
    -------
    dict mapping each feature name (e.g. "contrast") to an xr.Dataset with
    variables named ``window_{size}`` (e.g. "window_21"), each a 2-D
    DataArray with dims (line, sample). Source PC attrs are preserved and
    GLCM metadata (pc, window_size, glcm_feature) is added.

    Examples
    --------
    >>> ds, loadings_da, valid_da = pca_dataarray(da, n_components=3)
    >>> tex = glcm_texture_dataset(
    ...     ds,
    ...     window_sizes=[9, 21, 41],
    ...     features=["contrast", "homogeneity"],
    ...     levels=64,
    ... )
    >>> tex["contrast"]["window_21"]          # 2-D DataArray
    >>> tex["contrast"]["window_21"].attrs    # {"pc": "PC1", "window_size": 21, ...}
    """
    if not isinstance(scores, xr.Dataset):
        raise TypeError("scores must be an xr.Dataset (the first return value of pca_dataarray).")

    all_pcs = list(scores.data_vars)
    pcs_to_use = pc_names if pc_names is not None else all_pcs

    invalid_pcs = set(pcs_to_use) - set(all_pcs)
    if invalid_pcs:
        raise ValueError(f"Requested PCs not found in scores: {invalid_pcs}")

    # Validate all window sizes up front and warn about any that will be skipped
    first_band = scores[all_pcs[0]]
    image_shape = (first_band.sizes["line"], first_band.sizes["sample"])
    max_window = min(image_shape)
    valid_window_sizes, skipped_window_sizes = [], []
    for w in window_sizes:
        if w % 2 == 0:
            logger.warning(
                "window_size=%d is even and will be skipped (must be odd).", w
            )
            skipped_window_sizes.append(w)
        elif w > max_window:
            logger.warning(
                "window_size=%d exceeds the smallest image dimension "
                "(%d×%d → min=%d) and will be skipped.",
                w, image_shape[0], image_shape[1], max_window,
            )
            skipped_window_sizes.append(w)
        else:
            valid_window_sizes.append(w)

    if not valid_window_sizes:
        logger.warning(
            "No valid window sizes remain after filtering. "
            "Image shape is %d×%d; all requested sizes %s were invalid. "
            "Returning an empty Dataset.",
            image_shape[0], image_shape[1], list(window_sizes),
        )
        return {}

    result = {}

    for pc in pcs_to_use:
        pc_band = scores[pc]  # 2-D DataArray (line, sample)
        logger.info("Processing %s ...", pc)

        for w in valid_window_sizes:
            logger.info("  window_size = %d", w)
            ds = glcm_texture_dataarray(
                pc_band,
                window_size=w,
                features=features,
                distances=distances,
                angles=angles,
                levels=levels,
                lower_pct=lower_pct,
                upper_pct=upper_pct,
            )
            for feat, da in ds.data_vars.items():
                da.attrs.update({**pc_band.attrs,
                                 "pc": pc, "window_size": w,
                                 "glcm_feature": feat})
                result.setdefault(feat, {})[f"window_{w}"] = da

    return {feat: xr.Dataset(windows) for feat, windows in result.items()}
