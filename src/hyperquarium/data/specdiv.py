"""
Spectral diversity functions
Python translation of specdiv_functions.R (Etienne Laliberté, February 2019)

Input data expected as xarray.Dataset with data variables: PC1, PC2, PC3
(eigenvectors / PCA scores on spatial coordinates x, y)
"""

import numpy as np
import xarray as xr
from scipy.linalg import svd as scipy_svd


# ---------------------------------------------------------------------------
# Brightness normalisation
# ---------------------------------------------------------------------------

def bright_norm(x: np.ndarray) -> np.ndarray:
    """Brightness-normalise a 1-D spectrum array."""
    return x / np.sqrt(np.sum(x ** 2))


def bright_norm_da(da: xr.DataArray, dim: str = "wavelength") -> xr.DataArray:
    """
    Brightness-normalise each spectrum in an xarray.DataArray.

    Parameters
    ----------
    da : DataArray with a 'wavelength' (or equivalent) dimension.
    dim : name of the spectral dimension over which to normalise.

    Returns
    -------
    Normalised DataArray.
    """
    norm = np.sqrt((da ** 2).sum(dim=dim))
    return da / norm


# ---------------------------------------------------------------------------
# PCA on a 2-D numpy matrix (replaces pca_mat)
# ---------------------------------------------------------------------------

def pca_mat(X: np.ndarray,
            scaling: int = 1,
            p: float = 0.99,
            feature_names=None):
    """
    PCA on a 2-D array (samples × features), matching R pca_mat logic.

    Parameters
    ----------
    X        : (n_samples, n_features) array, will be mean-centred internally.
    scaling  : 1 → distance biplot (scores = Y @ U);
               2 → correlation biplot (scores = sqrt(n-1) * U_left).
    p        : cumulative proportion of variance threshold for PC selection.
    feature_names : optional list of feature names for the loadings output.

    Returns
    -------
    dict with keys:
        obj       – (n, n_pcs) score matrix
        descript  – (n_features, n_pcs) loading matrix
        prop      – proportion of variance per selected PC
        cumprop   – cumulative proportion per selected PC
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 2:
        raise ValueError(
            f"pca_mat requires at least 2 samples, got {n}."
        )

    # Mean-centre (no scaling to unit variance, matching R's scale(x, scale=FALSE))
    Y = X - X.mean(axis=0)

    # SVD
    U_left, d, Vt = scipy_svd(Y, full_matrices=False)
    V = Vt.T  # right singular vectors (loadings direction)

    eigenvalues = (1.0 / (n - 1)) * d ** 2

    # Drop near-zero eigenvalues (R: values > epsilon)
    epsilon = np.sqrt(np.finfo(float).eps)
    k = np.sum(eigenvalues > epsilon)
    eigenvalues = eigenvalues[:k]
    d = d[:k]
    V = V[:, :k]
    U_left = U_left[:, :k]

    prop = eigenvalues / eigenvalues.sum()
    cumprop = np.cumsum(prop)

    # Select PCs: same logic as R
    #   if p < cumprop[0]: keep [0,1]  (at least first two)
    #   else: keep all where cumprop < p, then add one more
    if p < cumprop[0]:
        which_values = np.array([0, min(1, k - 1)])
    else:
        which_values = np.where(cumprop < p)[0]

    # add the next PC (R: which.values + 1), clamped to valid range
    last = which_values[-1] + 1 if len(which_values) > 0 else 0
    last = min(last, k - 1)
    sel_idx = np.unique(np.append(which_values, last))

    eigenvalues_sel = eigenvalues[sel_idx]
    n_pcs = len(sel_idx)
    U_sel = V[:, sel_idx]  # loadings (feature × PC)
    U_left_sel = U_left[:, sel_idx]  # left singular vectors

    if scaling == 1:
        obj = Y @ U_sel  # scores
        descript = U_sel  # loadings
    else:
        obj = np.sqrt(n - 1) * U_left_sel  # scores
        descript = U_sel @ np.diag(eigenvalues_sel ** 0.5)  # loadings

    pc_names = [f"PC{i + 1}" for i in range(n_pcs)]
    prop_sel = prop[sel_idx]
    cumprop_sel = cumprop[sel_idx]

    return {
        "obj": obj,  # ndarray (n, n_pcs)
        "descript": descript,  # ndarray (n_features, n_pcs)
        "prop": dict(zip(pc_names, prop_sel)),
        "cumprop": dict(zip(pc_names, cumprop_sel)),
        "pc_names": pc_names,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# PCA on a 3-D xarray.DataArray  (line, band, sample)
# ---------------------------------------------------------------------------

def pca_dataarray(da: xr.DataArray,
                  scaling: int = 1,
                  p: float = 0.99,
                  n_components: int = None):
    """
    PCA on a 3-D DataArray of shape (line, band, sample), dtype float64.

    Only pixels that are non-NaN across ALL bands are used for fitting.
    Scores are written back onto the full (line, sample) grid, with NaN
    retained for masked pixels.

    Parameters
    ----------
    da      : xr.DataArray with dims (line, band, sample), shape e.g.
              (184, 135, 184).  The 'band' dimension is the feature axis.
    scaling : 1 = distance biplot, 2 = correlation biplot (see pca_mat).
    p            : cumulative variance threshold for PC selection.
                   Ignored if n_components is set.
    n_components : if set, retain exactly this many PCs regardless of p.
                   Useful when p fails to capture a desired PC (e.g. PC3).

    Returns
    -------
    dict with keys:
        scores     – xr.DataArray, shape (line, sample, n_pcs),
                     PC scores on the original spatial grid (NaN where
                     any input band was NaN).
        loadings   – xr.DataArray, shape (band, n_pcs),
                     PC loadings (eigenvectors, scaled by chosen biplot type).
        prop       – dict {PC1: float, ...}  variance proportion per PC.
        cumprop    – dict {PC1: float, ...}  cumulative variance proportion.
        valid_mask – xr.DataArray bool, shape (line, sample), True where
                     all bands are finite.
    """
    # Expect dims (line, band, sample); transpose to (line, sample, band)
    # so we can stack the two spatial dims cleanly.
    da = da.transpose("line", "band", "sample")

    band_coords = da.coords["band"].values
    n_bands = da.sizes["band"]

    # Stack (line, sample) → pixel; result shape: (band, n_pixels)
    stacked = da.stack(pixel=("line", "sample"))  # (band, n_pixels)
    mat = stacked.values.T  # (n_pixels, n_bands)

    # Valid pixel mask: non-NaN across all bands AND non-zero norm
    # (zero-norm pixels cause division by zero in brightness normalisation)
    norms = np.sqrt((mat ** 2).sum(axis=1))
    valid = np.all(np.isfinite(mat), axis=1) & (norms > 0)
    if valid.sum() == 0:
        raise ValueError("No valid (non-NaN, non-zero) pixels found in the DataArray.")
    if valid.sum() < 2:
        raise ValueError(
            f"Only {valid.sum()} valid pixel found — need at least 2 for PCA."
        )

    mat_valid = mat[valid]  # (n_valid, n_bands)

    # Run PCA on valid pixels only
    # If n_components is set, use a p high enough to guarantee all PCs pass
    # the threshold, then slice down to the requested count afterwards.
    result = pca_mat(mat_valid, scaling=scaling,
                     p=1.0 if n_components is not None else p,
                     feature_names=list(band_coords))

    if n_components is not None:
        max_available = result["obj"].shape[1]
        if n_components > max_available:
            raise ValueError(
                f"n_components={n_components} exceeds the number of available "
                f"PCs ({max_available}) after eigenvalue truncation."
            )
        pc_names = result["pc_names"][:n_components]
        result["obj"] = result["obj"][:, :n_components]
        result["descript"] = result["descript"][:, :n_components]
        result["prop"] = {k: result["prop"][k] for k in pc_names}
        result["cumprop"] = {k: result["cumprop"][k] for k in pc_names}
        result["pc_names"] = pc_names

    n_pixels = mat.shape[0]
    n_pcs = result["obj"].shape[1]
    pc_names = result["pc_names"]

    # Place scores back onto full pixel grid (NaN for masked pixels)
    full_scores = np.full((n_pixels, n_pcs), np.nan, dtype=np.float64)
    full_scores[valid] = result["obj"]

    # Unstack pixel → (line, sample), one DataArray per PC
    pixel_coords = stacked.coords["pixel"]
    pc_vars = {}
    for i, pc in enumerate(pc_names):
        flat = xr.DataArray(full_scores[:, i],
                            coords=[pixel_coords], dims=["pixel"])
        pc_vars[pc] = flat.unstack("pixel").transpose("line", "sample")

    # Loadings DataArray: (band, pc)
    loadings_da = xr.DataArray(
        result["descript"],
        coords={"band": band_coords, "pc": pc_names},
        dims=["band", "pc"],
        name="loadings",
    )

    # Valid mask on spatial grid
    valid_da = xr.DataArray(
        valid, coords=[pixel_coords], dims=["pixel"]
    ).unstack("pixel").transpose("line", "sample")
    valid_da.name = "valid_mask"

    # Build Dataset: PC score variables only, dims (line, sample).
    # loadings (band, pc) and valid_mask are returned separately to avoid
    # dimension conflicts; specdiv() only needs the PC score variables.
    # prop and cumprop are stored as per-variable attributes (NetCDF4-safe).
    base_attrs = dict(da.attrs)
    for pc in pc_names:
        pc_vars[pc].attrs = {**base_attrs,
                             "prop": result["prop"][pc],
                             "cumprop": result["cumprop"][pc]}

    ds = xr.Dataset(pc_vars, attrs=base_attrs)

    return ds, loadings_da, valid_da


# ---------------------------------------------------------------------------
# Sum-of-squares helpers (replaces sum_squares and sum_squares_beta)
# ---------------------------------------------------------------------------

def sum_squares(Y: np.ndarray) -> dict:
    """
    Compute total sum of squares (gamma / alpha spectral diversity).

    Parameters
    ----------
    Y : (n_samples, n_features) array.

    Returns
    -------
    dict with:
        ss    – total SS
        sdiv  – SS / (n-1)   [spectral diversity]
        lcsd  – per-sample contribution (length n)
        fcsd  – per-feature contribution (length n_features)
    """
    n = Y.shape[0]
    Y_cent = Y - Y.mean(axis=0)
    sij = Y_cent ** 2
    SS_total = sij.sum()
    SS_row = sij.sum(axis=1)  # local contribution (per sample)
    SS_col = sij.sum(axis=0)  # feature contribution
    fcsd = SS_col / SS_total
    lcsd = SS_row / SS_total
    sdiv = SS_total / (n - 1)
    return {"ss": SS_total, "sdiv": sdiv, "lcsd": lcsd, "fcsd": fcsd}


def sum_squares_beta(Y: np.ndarray, groups: np.ndarray) -> dict:
    """
    Compute beta sum of squares (between-group spectral diversity).

    Parameters
    ----------
    Y      : (n_samples, n_features) array (mean-centred globally).
    groups : (n_samples,) integer/string array of group labels.

    Returns
    -------
    dict with:
        ss     – beta SS
        sdiv   – beta SS / (n-1)
        lcss   – per-group SS (array, length n_groups)
        lcsd   – per-group relative contribution
        fcsd   – per-feature relative contribution
    """
    n = Y.shape[0]
    Y_cent = Y - Y.mean(axis=0)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_features = Y.shape[1]

    # For each group, squared group means × n_k  (matches R's mskj logic)
    mskj = np.zeros((n_groups, n_features))
    for gi, g in enumerate(unique_groups):
        mask = groups == g
        n_k = mask.sum()
        group_mean = Y_cent[mask].mean(axis=0)
        mskj[gi] = n_k * (group_mean ** 2)

    SSbk = mskj.sum(axis=1)  # per-group beta SS
    SSbj = mskj.sum(axis=0)  # per-feature beta SS
    SSb = mskj.sum()

    sdiv = SSb / (n - 1)
    fcsd = SSbj / SSb
    lcsd = SSbk / SSb

    return {
        "ss": SSb,
        "sdiv": sdiv,
        "lcss": SSbk,
        "lcsd": lcsd,
        "fcsd": fcsd,
        "groups": unique_groups,
    }


# ---------------------------------------------------------------------------
# Count non-NaN pixels per community block
# ---------------------------------------------------------------------------

def count_pixels(ds: xr.Dataset, fact: int = 40) -> xr.Dataset:
    """
    For each community block of size (fact × fact) pixels, count valid pixels.

    Parameters
    ----------
    ds   : xarray.Dataset (spatial dims x, y).
    fact : block size (aggregation factor).

    Returns
    -------
    xarray.Dataset with variables n, n_total, prop on a coarsened grid.
    """
    # Use any data variable to determine valid pixels
    any_var = list(ds.data_vars)[0]
    valid = ds[any_var].notnull()

    # Coarsen to community blocks
    # xarray coarsen with boundary='trim' drops incomplete edge blocks
    coarsened = valid.coarsen(sample=fact, line=fact, boundary="trim")
    n_valid = coarsened.sum().rename("n")
    n_total = (valid * 0 + 1).coarsen(sample=fact, line=fact, boundary="trim").sum().rename("n_total")

    prop = (n_valid / n_total).rename("prop")

    return xr.Dataset({"n": n_valid, "n_total": n_total, "prop": prop})


# ---------------------------------------------------------------------------
# Main specdiv function
# ---------------------------------------------------------------------------

def specdiv(ds: xr.Dataset,
            fact: int = 40,
            prop_threshold: float = 0.5,
            n_iter: int = 1,
            pc_vars=None) -> dict:
    """
    Partition spectral diversity into alpha, beta, gamma components.

    Parameters
    ----------
    ds             : xarray.Dataset with PC score layers (variables PC1, PC2, PC3 …).
                     Must have spatial dimensions named 'x' and 'y'.
    fact           : community block size in pixels.
    prop_threshold : minimum fraction of valid pixels required to retain a community.
    n_iter         : number of random-subsampling iterations.
    pc_vars        : which variables to use (default: all).

    Returns
    -------
    dict with keys:
        ss        – DataFrame-like dict {source, sum_squares, prop_gamma}
        sdiv      – dict {mean_alpha, beta, gamma}
        fcsd      – dict {source: fcsd_array}  (per-feature contributions)
        community_stats – dict of per-community DataArrays:
                          beta_lcsd, beta_lcss, alpha_sdiv, alpha_fcsd
    """
    if pc_vars is None:
        pc_vars = list(ds.data_vars)

    # ------------------------------------------------------------------ #
    # 1. Count valid pixels per community block
    # ------------------------------------------------------------------ #
    pixel_counts = count_pixels(ds, fact=fact)
    prop_map = pixel_counts["prop"]

    # Mask communities below threshold
    valid_comms = prop_map >= prop_threshold
    if not valid_comms.any():
        raise ValueError("No community blocks meet the prop_threshold criterion.")

    # Minimum valid pixel count across retained communities
    min_pixels = int(pixel_counts["n"].where(valid_comms).min().item())

    # ------------------------------------------------------------------ #
    # 2. Build pixel → community mapping
    # ------------------------------------------------------------------ #
    # We label each fine-resolution pixel with the community it belongs to.
    # Communities are indexed by their coarsened (cy, cx) position.
    n_features = len(pc_vars)

    # Stack ds into (pixel, feature) matrix
    stacked = ds[pc_vars].to_array(dim="band").stack(pixel=("line", "sample"))
    pixel_coords = stacked.coords["pixel"]
    mat_all = stacked.values.T  # (n_pixels, n_features)

    line_vals = pixel_coords.coords["line"].values
    sample_vals = pixel_coords.coords["sample"].values

    # Coarsened grid coordinates for community centres
    comm_ds = pixel_counts

    line_step = (ds.coords["line"].values[1] - ds.coords["line"].values[0]) if ds.sizes["line"] > 1 else 1.0
    sample_step = (ds.coords["sample"].values[1] - ds.coords["sample"].values[0]) if ds.sizes["sample"] > 1 else 1.0

    # Use valid_comms as the single source of truth for grid shape AND coordinates
    # to avoid off-by-one mismatches between coarsen coords and mask shape
    valid_comm_mask = valid_comms.values  # (n_cy, n_cx)
    n_cy, n_cx = valid_comm_mask.shape
    cy_vals = valid_comms.coords["line"].values  # length guaranteed == n_cy
    cx_vals = valid_comms.coords["sample"].values  # length guaranteed == n_cx

    def assign_blocks(coords, step, fact, max_idx):
        """Return 0-based block index for each coordinate, clamped to [0, max_idx-1]."""
        min_c = coords.min()
        idx = ((coords - min_c) / (abs(step) * fact)).astype(int)
        return np.clip(idx, 0, max_idx - 1)

    cy_idx = assign_blocks(line_vals, line_step, fact, n_cy)
    cx_idx = assign_blocks(sample_vals, sample_step, fact, n_cx)

    # Pixel-level validity flag
    pixel_in_valid_comm = valid_comm_mask[cy_idx, cx_idx]
    # Also mask out NaN pixels
    pixel_not_nan = np.all(np.isfinite(mat_all), axis=1)
    pixel_valid = pixel_in_valid_comm & pixel_not_nan

    mat_valid = mat_all[pixel_valid]
    cy_idx_v = cy_idx[pixel_valid]
    cx_idx_v = cx_idx[pixel_valid]

    # Integer community labels (flat index)
    comm_labels = cy_idx_v * n_cx + cx_idx_v
    unique_comms = np.unique(comm_labels)
    n_comms = len(unique_comms)

    # ------------------------------------------------------------------ #
    # 3. Iterative random subsampling & diversity calculation
    # ------------------------------------------------------------------ #
    gamma_ss_list = []
    gamma_sdiv_list = []
    gamma_fcsd_list = []

    alpha_sdiv_list = []  # list of (n_comms,) arrays
    alpha_fcsd_list = []  # list of (n_comms, n_features) arrays
    alpha_ss_list = []  # list of (n_comms,) arrays

    beta_ss_list = []
    beta_sdiv_list = []
    beta_fcsd_list = []
    beta_lcsd_list = []
    beta_lcss_list = []

    rng = np.random.default_rng()

    # Recompute min_pixels from actual post-mask per-community counts
    # (coarsened counts can differ from true valid pixel counts after NaN masking)
    min_pixels = min(int(np.sum(comm_labels == c)) for c in unique_comms)

    for _ in range(n_iter):
        # Subsample min_pixels from each community
        sampled_rows = []
        sampled_comm = []
        for c in unique_comms:
            idx = np.where(comm_labels == c)[0]
            # replace=True is a safe fallback but should not occur after fix above
            chosen = rng.choice(idx, size=min_pixels, replace=len(idx) < min_pixels)
            sampled_rows.append(chosen)
            sampled_comm.extend([c] * min_pixels)

        sampled_rows = np.concatenate(sampled_rows)
        sampled_comm = np.array(sampled_comm)
        Y_sample = mat_valid[sampled_rows]

        # -- Gamma --
        g = sum_squares(Y_sample)
        gamma_ss_list.append(g["ss"])
        gamma_sdiv_list.append(g["sdiv"])
        gamma_fcsd_list.append(g["fcsd"])

        # -- Alpha (per community) --
        a_sdiv = np.zeros(n_comms)
        a_fcsd = np.zeros((n_comms, n_features))
        a_ss = np.zeros(n_comms)
        for ci, c in enumerate(unique_comms):
            mask_c = sampled_comm == c
            Y_c = Y_sample[mask_c]
            res = sum_squares(Y_c)
            a_sdiv[ci] = res["sdiv"]
            a_fcsd[ci] = res["fcsd"]
            a_ss[ci] = res["ss"]
        alpha_sdiv_list.append(a_sdiv)
        alpha_fcsd_list.append(a_fcsd)
        alpha_ss_list.append(a_ss)

        # -- Beta --
        b = sum_squares_beta(Y_sample, sampled_comm)
        beta_ss_list.append(b["ss"])
        beta_sdiv_list.append(b["sdiv"])
        beta_fcsd_list.append(b["fcsd"])
        beta_lcsd_list.append(b["lcsd"])
        beta_lcss_list.append(b["lcss"])

    # ------------------------------------------------------------------ #
    # 4. Aggregate across iterations
    # ------------------------------------------------------------------ #
    ss_gamma = float(np.mean(gamma_ss_list))
    sdiv_gamma = float(np.mean(gamma_sdiv_list))
    fcsd_gamma = np.mean(gamma_fcsd_list, axis=0)

    # Alpha
    sdiv_alpha_mean_per_comm = np.mean(alpha_sdiv_list, axis=0)  # (n_comms,)
    fcsd_alpha_mean_per_comm = np.mean(alpha_fcsd_list, axis=0)  # (n_comms, n_features)
    ss_alpha_mean_per_comm = np.mean(alpha_ss_list, axis=0)  # (n_comms,)
    sdiv_alpha_mean = float(sdiv_alpha_mean_per_comm.mean())
    fcsd_alpha_mean = fcsd_alpha_mean_per_comm.mean(axis=0)
    ss_alpha_sum = float(ss_alpha_mean_per_comm.sum())

    # Beta
    ss_beta = float(np.mean(beta_ss_list))
    sdiv_beta = float(np.mean(beta_sdiv_list))
    fcsd_beta = np.mean(beta_fcsd_list, axis=0)
    lcsd_beta_mean = np.mean(beta_lcsd_list, axis=0)  # (n_comms,)
    lcss_beta_mean = np.mean(beta_lcss_list, axis=0)  # (n_comms,)

    # ------------------------------------------------------------------ #
    # 5. Map community-level results back onto spatial grids
    # ------------------------------------------------------------------ #
    # Build coordinate arrays from n_cy/n_cx (authoritative shape from valid_comm_mask)
    # rather than from coarsened dataset coords, which can be off by one.
    comm_cy = cy_vals[:n_cy]
    comm_cx = cx_vals[:n_cx]

    def _fill_comm_grid(values_per_comm, comm_ids, unique_comms, cy_vals, cx_vals, n_cy, n_cx):
        """
        Map flat per-community values back to a 2-D (line, sample) grid.
        values_per_comm : 1-D array of length n_comms
        Returns a DataArray.
        """
        grid = np.full((n_cy, n_cx), np.nan)
        for ci, c in enumerate(unique_comms):
            row = c // n_cx
            col = c % n_cx
            grid[row, col] = values_per_comm[ci]
        # Derive coords from actual grid shape to avoid off-by-one mismatches
        return xr.DataArray(grid,
                            coords={"line": cy_vals[:n_cy], "sample": cx_vals[:n_cx]},
                            dims=["line", "sample"])

    beta_lcsd_da = _fill_comm_grid(lcsd_beta_mean, comm_labels, unique_comms,
                                   comm_cy, comm_cx, n_cy, n_cx)
    beta_lcss_da = _fill_comm_grid(lcss_beta_mean, comm_labels, unique_comms,
                                   comm_cy, comm_cx, n_cy, n_cx)
    alpha_sdiv_da = _fill_comm_grid(sdiv_alpha_mean_per_comm, comm_labels, unique_comms,
                                    comm_cy, comm_cx, n_cy, n_cx)

    # alpha_fcsd: one layer per feature
    alpha_fcsd_das = {}
    for fi, var in enumerate(pc_vars):
        alpha_fcsd_das[var] = _fill_comm_grid(
            fcsd_alpha_mean_per_comm[:, fi], comm_labels, unique_comms,
            comm_cy, comm_cx, n_cy, n_cx
        )
    alpha_fcsd_ds = xr.Dataset(alpha_fcsd_das)

    # ------------------------------------------------------------------ #
    # 6. Assemble output (mirrors R specdiv output structure)
    # ------------------------------------------------------------------ #
    ss = {
        "source": ["alpha", "beta", "gamma"],
        "sum_squares": [ss_alpha_sum, ss_beta, ss_gamma],
        "prop_gamma": [ss_alpha_sum / ss_gamma, ss_beta / ss_gamma, 1.0],
    }

    sdiv = {
        "mean_alpha": sdiv_alpha_mean,
        "beta": sdiv_beta,
        "gamma": sdiv_gamma,
    }

    fcsd = {
        "source": ["mean_alpha", "beta", "gamma"],
        **{f"fcsd_{v}": [fcsd_alpha_mean[fi], fcsd_beta[fi], fcsd_gamma[fi]]
           for fi, v in enumerate(pc_vars)},
    }

    community_stats = {
        "beta_lcsd": beta_lcsd_da,
        "beta_lcss": beta_lcss_da,
        "alpha_sdiv": alpha_sdiv_da,
        "alpha_fcsd": alpha_fcsd_ds,
    }

    return {
        "ss": ss,
        "sdiv": sdiv,
        "fcsd": fcsd,
        "community_stats": community_stats,
    }


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def specdiv_to_dataframes(results: dict) -> dict:
    """
    Convert specdiv() output into a dict of pandas DataFrames.

    Parameters
    ----------
    results : dict returned by specdiv().

    Returns
    -------
    dict with keys:
        ss             – alpha/beta/gamma sum-of-squares table
        sdiv           – scalar spectral diversity values (single-row)
        fcsd           – feature contributions per source (alpha/beta/gamma)
        alpha_sdiv     – per-community alpha spectral diversity (y, x, value)
        beta_lcsd      – per-community beta LCSD (y, x, value)
        beta_lcss      – per-community beta LCSS (y, x, value)
        alpha_fcsd     – per-community alpha FCSD, one row per (y, x) location
    """
    import pandas as pd

    dfs = {}

    # ss table
    dfs["ss"] = pd.DataFrame(results["ss"])

    # sdiv: single-row DataFrame
    dfs["sdiv"] = pd.DataFrame([results["sdiv"]])

    # fcsd table
    dfs["fcsd"] = pd.DataFrame(results["fcsd"])

    # Spatial community stats — flatten each DataArray / Dataset to long form
    cs = results["community_stats"]

    for key in ("alpha_sdiv", "beta_lcsd", "beta_lcss"):
        da = cs[key]
        df = da.to_dataframe(name=key).reset_index().dropna()
        dfs[key] = df

    # alpha_fcsd is a Dataset (one variable per PC)
    alpha_fcsd_df = cs["alpha_fcsd"].to_dataframe().reset_index().dropna()
    dfs["alpha_fcsd"] = alpha_fcsd_df

    return dfs


def specdiv_to_csv(results: dict, prefix: str = "specdiv") -> list:
    """
    Save specdiv() output to CSV files, one per output table.

    Parameters
    ----------
    results : dict returned by specdiv().
    prefix  : filename prefix (e.g. "specdiv" → "specdiv_ss.csv", etc.).

    Returns
    -------
    List of file paths written.
    """

    dfs = specdiv_to_dataframes(results)
    paths = []
    for name, df in dfs.items():
        path = f"{prefix}_{name}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
        print(f"Saved: {path}")
    return paths


def specdiv_to_json(results: dict, path: str = "specdiv_results.json") -> str:
    """
    Save specdiv() output to a single JSON file.

    Scalar values (ss, sdiv, fcsd) are stored directly.
    Spatial community stats are stored as records (list of dicts).

    Parameters
    ----------
    results : dict returned by specdiv().
    path    : output file path.

    Returns
    -------
    The file path written.
    """
    import json

    dfs = specdiv_to_dataframes(results)

    out = {}
    for name, df in dfs.items():
        out[name] = df.to_dict(orient="records")

    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def specdiv_batch(ds: xr.Dataset,
                  param_grid: list,
                  pc_vars=None):
    """
    Run specdiv() over multiple parameter combinations.

    Returns both a tidy pandas DataFrame of scalar results and an
    xr.Dataset of community-level spatial DataArrays, one variable per
    (run, stat) combination, with all scalar results stored in .attrs.

    Parameters
    ----------
    ds         : xarray.Dataset passed to every specdiv() call.
    param_grid : list of dicts, each with any subset of keys:
                     fact, prop_threshold, n_iter
                 Missing keys fall back to defaults (40, 0.5, 1).
                 Example:
                     [{"fact": 5,  "prop_threshold": 0.5, "n_iter": 1},
                      {"fact": 10, "prop_threshold": 0.3, "n_iter": 5}]
    pc_vars    : passed through to specdiv().

    Returns
    -------
    df         : pandas.DataFrame — one row per (run × source), columns:
                     run, fact, prop_threshold, n_iter,
                     source, sum_squares, prop_gamma, sdiv, fcsd_PC1, ...
    spatial_ds : xr.Dataset — one variable per (run × stat), e.g.
                     run1_alpha_sdiv, run1_beta_lcsd, run1_beta_lcss,
                     run1_alpha_fcsd_PC1, ...
                 Each DataArray carries all scalar results in .attrs:
                     run, fact, prop_threshold, n_iter,
                     source, sum_squares, prop_gamma, sdiv, fcsd_PC1, ...
    """
    import pandas as pd

    defaults = {"fact": 40, "prop_threshold": 0.5, "n_iter": 1}
    rows = []
    run_datasets = {}

    for run_idx, params in enumerate(param_grid):
        p = {**defaults, **params}
        run_label = f"run{run_idx + 1}"
        spatial_vars = {}  # reset for each run
        print(f"Run {run_idx + 1}/{len(param_grid)}: {p}")

        try:
            res = specdiv(ds,
                          fact=p["fact"],
                          prop_threshold=p["prop_threshold"],
                          n_iter=p["n_iter"],
                          pc_vars=pc_vars)

            ss_df = pd.DataFrame(res["ss"])
            fcsd_df = pd.DataFrame(res["fcsd"])
            fcsd_cols = [c for c in fcsd_df.columns if c != "source"]

            # --- scalar rows (DataFrame) ---
            source_rows = []
            for src in ["mean_alpha", "beta", "gamma"]:
                ss_src = ss_df[ss_df["source"] == ("alpha" if src == "mean_alpha" else src)].iloc[0]
                fcsd_src = fcsd_df[fcsd_df["source"] == src]
                row = {
                    "run": run_idx + 1,
                    "fact": p["fact"],
                    "prop_threshold": p["prop_threshold"],
                    "n_iter": p["n_iter"],
                    "source": src,
                    "sum_squares": ss_src["sum_squares"],
                    "prop_gamma": ss_src["prop_gamma"],
                    "sdiv": res["sdiv"][src],
                }
                for c in fcsd_cols:
                    row[c] = fcsd_src.iloc[0][c] if not fcsd_src.empty else None
                source_rows.append(row)
            rows.extend(source_rows)

            # --- build attrs dict from all scalar results for this run ---
            def _make_attrs(stat_source):
                base = {
                    "run": run_idx + 1,
                    "fact": p["fact"],
                    "prop_threshold": p["prop_threshold"],
                    "n_iter": p["n_iter"],
                    "stat_source": stat_source,
                }
                for row in source_rows:
                    if row["source"] == stat_source:
                        base.update({k: v for k, v in row.items()
                                     if k not in ("run", "fact", "prop_threshold", "n_iter")})
                return base

            cs = res["community_stats"]

            da = cs["alpha_sdiv"].copy()
            da.attrs = _make_attrs("mean_alpha")
            spatial_vars["alpha_sdiv"] = da

            da = cs["beta_lcsd"].copy()
            da.attrs = _make_attrs("beta")
            spatial_vars["beta_lcsd"] = da

            da = cs["beta_lcss"].copy()
            da.attrs = _make_attrs("beta")
            spatial_vars["beta_lcss"] = da

            alpha_fcsd_ds = cs["alpha_fcsd"]
            for pc_var in alpha_fcsd_ds.data_vars:
                da = alpha_fcsd_ds[pc_var].copy()
                da.attrs = _make_attrs("mean_alpha")
                spatial_vars[f"alpha_fcsd_{pc_var}"] = da

            run_ds = xr.Dataset(spatial_vars,
                                attrs={"run": run_idx + 1,
                                       "fact": p["fact"],
                                       "prop_threshold": p["prop_threshold"],
                                       "n_iter": p["n_iter"],
                                       "failed": 0,
                                       "gamma_sdiv": float(res["sdiv"]["gamma"]),
                                       "beta_sdiv": float(res["sdiv"]["beta"])})
            run_datasets[run_label] = run_ds

        except Exception as e:
            print(f"  WARNING: run {run_idx + 1} failed — {e}. Producing empty outputs.")
            # Empty scalar rows
            for src in ["mean_alpha", "beta", "gamma"]:
                rows.append({
                    "run": run_idx + 1, "fact": p["fact"],
                    "prop_threshold": p["prop_threshold"], "n_iter": p["n_iter"],
                    "source": src, "sum_squares": np.nan, "prop_gamma": np.nan,
                    "sdiv": np.nan,
                })
            # Empty spatial Dataset (NaN arrays matching ds spatial dims)
            empty = np.full((ds.sizes["line"], ds.sizes["sample"]), np.nan)
            empty_da = xr.DataArray(empty,
                                    coords={"line": ds.coords["line"],
                                            "sample": ds.coords["sample"]},
                                    dims=["line", "sample"])
            empty_attrs = {"run": run_idx + 1, "fact": p["fact"],
                           "prop_threshold": p["prop_threshold"],
                           "n_iter": p["n_iter"], "failed": 1, "error": str(e),
                           "gamma_sdiv": np.nan, "beta_sdiv": np.nan}
            stat_names = ["alpha_sdiv", "beta_lcsd", "beta_lcss"]
            _pc_vars = pc_vars if pc_vars is not None else list(ds.data_vars)
            stat_names += [f"alpha_fcsd_{v}" for v in _pc_vars]
            run_ds = xr.Dataset(
                {name: empty_da.assign_attrs(empty_attrs) for name in stat_names},
                attrs=empty_attrs)
            run_datasets[run_label] = run_ds

    df = pd.concat([pd.DataFrame([r]) for r in rows], ignore_index=True)
    # Deduplicate: keep last occurrence of each (run, source) in case of retry
    df = df.drop_duplicates(subset=["run", "source"], keep="last")
    meta = ["run", "fact", "prop_threshold", "n_iter", "source"]
    rest = [c for c in df.columns if c not in meta]
    df = df[meta + rest]

    return df, run_datasets