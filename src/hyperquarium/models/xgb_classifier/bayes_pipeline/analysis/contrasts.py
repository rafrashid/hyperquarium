"""
analysis/contrasts.py — Step 4 posterior arithmetic.

All quantities are computed PER DRAW on the posterior, then summarised to median +
HDI. Nothing is refit. Every output table carries `n_roi` so each row self-documents
how much data backs it (the thin-class reporting filter).

Outputs per cell:
  - per_class_sigma : every class's own sigma (baseline), natural scale, + n_roi
  - turf_distances  : mu[turf] - mu[each]      (identity scale, SD units), + n_roi
  - turf_sigma_ratios: sigma[turf]/sigma[each] (from log-sigma diff), + n_roi
  - sigma_vs_gap    : sigma[turf] / |mu[X]-mu[Y]| for turf-referenced gaps, + n_roi

Scale handling:
  - mu lives on the identity scale (mean structure) -> plain differences.
  - sigma lives on the LOG scale in the model -> exponentiate per draw to natural
    scale BEFORE forming ratios with mu-differences (never mix scales in a ratio).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from bayes_pipeline.models.fit import CellData


# --------------------------------------------------------------------------- #
# Posterior extraction
# --------------------------------------------------------------------------- #
def _stack_draws(da):
    """Flatten (chain, draw) -> sample. Returns np array shape (n_samples, ...)."""
    return da.stack(sample=("chain", "draw")).transpose("sample", ...).values


def extract_class_posteriors(idata, cell: CellData):
    """
    Return dicts of per-class posterior draw arrays:
      mu[class]        -> (n_samples,)   natural/identity scale
      sigma[class]     -> (n_samples,)   natural scale (exp of log-sigma)
    Class order taken from the model's category levels.

    Bambi names the cell-means terms 'class[<level>]' for mu and
    'sigma_class[<level>]' (log scale) for the distributional part. We resolve names
    defensively by scanning the posterior variables.
    """
    post = idata.posterior
    classes = [str(c) for c in cell.df["class"].cat.categories]

    mu, sigma = {}, {}

    # ---- locate the mu term and the (log) sigma term -----------------------
    # Bambi typically exposes a coordinate dim (e.g. 'class__factor_dim' or 'class_dim')
    # with the class levels. We find the data variable whose coord values match classes.
    mu_var, mu_dim = _find_term(post, classes, prefer_contains=("class",),
                                exclude_contains=("sigma",))
    sig_var, sig_dim = _find_term(post, classes, prefer_contains=("sigma", "class"),
                                  exclude_contains=())

    for c in classes:
        mu[c] = _stack_draws(post[mu_var].sel({mu_dim: c}))
        log_sigma_c = _stack_draws(post[sig_var].sel({sig_dim: c}))
        sigma[c] = np.exp(log_sigma_c)  # log link -> natural scale

    return classes, mu, sigma


def _find_term(post, classes, prefer_contains, exclude_contains):
    """
    Find (variable_name, dim_name) in the posterior whose coordinate values equal the
    class levels and whose name matches the prefer/exclude hints.
    """
    candidates = []
    for var in post.data_vars:
        name = str(var)
        if any(x in name for x in exclude_contains):
            continue
        for dim in post[var].dims:
            if dim in ("chain", "draw"):
                continue
            coord_vals = [str(v) for v in post[var][dim].values]
            if set(coord_vals) == set(classes):
                score = sum(h in name for h in prefer_contains)
                candidates.append((score, name, dim))
    if not candidates:
        raise ValueError(
            f"Could not locate a posterior term over classes {classes}. "
            f"Available vars: {list(post.data_vars)}"
        )
    candidates.sort(reverse=True)
    _, name, dim = candidates[0]
    return name, dim


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _summary(draws: np.ndarray, ci: float) -> dict:
    import arviz as az
    hdi = az.hdi(draws, hdi_prob=ci)
    # az.hdi on a 1-D ndarray returns shape (2,)
    lo, hi = float(hdi[0]), float(hdi[1])
    return {
        "median": float(np.median(draws)),
        "mean": float(np.mean(draws)),
        "hdi_low": lo,
        "hdi_high": hi,
    }


# --------------------------------------------------------------------------- #
# The four output tables
# --------------------------------------------------------------------------- #
def per_class_sigma(classes, sigma, cell: CellData, cfg) -> pd.DataFrame:
    ci = cfg.analysis.credible_interval
    reliable = cfg.analysis.reliable_sigma_n_roi
    rows = []
    for c in classes:
        s = _summary(sigma[c], ci)
        n = cell.class_n_roi.get(c, 0)
        rows.append({
            "feature": cell.feature, "level": cell.level, "class": c,
            "sigma_median": s["median"], "sigma_hdi_low": s["hdi_low"],
            "sigma_hdi_high": s["hdi_high"], "n_roi": n,
            "sigma_reliable": n >= reliable,
        })
    return pd.DataFrame(rows)


def turf_distances(classes, mu, cell: CellData, cfg) -> pd.DataFrame:
    ci = cfg.analysis.credible_interval
    turf = cfg.labels.turf_class
    reliable = cfg.analysis.reliable_sigma_n_roi
    if turf not in classes:
        return pd.DataFrame()  # turf absent at this level (shouldn't happen for L1-3)
    rows = []
    for c in classes:
        if c == turf:
            continue
        diff = mu[turf] - mu[c]  # SD units (identity scale)
        s = _summary(diff, ci)
        n = cell.class_n_roi.get(c, 0)
        rows.append({
            "feature": cell.feature, "level": cell.level,
            "reference_class": c,
            "distance_median": s["median"], "distance_hdi_low": s["hdi_low"],
            "distance_hdi_high": s["hdi_high"],
            "excludes_zero": (s["hdi_low"] > 0) or (s["hdi_high"] < 0),
            "n_roi": n, "n_roi_reliable": n >= reliable,
        })
    return pd.DataFrame(rows)


def turf_sigma_ratios(classes, sigma, cell: CellData, cfg) -> pd.DataFrame:
    ci = cfg.analysis.credible_interval
    turf = cfg.labels.turf_class
    reliable = cfg.analysis.reliable_sigma_n_roi
    if turf not in classes:
        return pd.DataFrame()
    rows = []
    for c in classes:
        if c == turf:
            continue
        ratio = sigma[turf] / sigma[c]  # natural scale; exp already applied
        s = _summary(ratio, ci)
        n = cell.class_n_roi.get(c, 0)
        rows.append({
            "feature": cell.feature, "level": cell.level,
            "reference_class": c,
            "sigma_ratio_median": s["median"], "sigma_ratio_hdi_low": s["hdi_low"],
            "sigma_ratio_hdi_high": s["hdi_high"],
            "excludes_one": (s["hdi_low"] > 1) or (s["hdi_high"] < 1),
            "n_roi": n, "n_roi_reliable": n >= reliable,
        })
    return pd.DataFrame(rows)


def sigma_vs_gap(classes, mu, sigma, cell: CellData, cfg) -> pd.DataFrame:
    """
    turf within-spread vs each between-class gap: sigma[turf] / |mu[X] - mu[Y]|.

    Turf-referenced: gaps are turf-vs-each, i.e. |mu[turf] - mu[c]|, so this asks
    'is turf's internal spread as large as the distance from turf to class c'. Gaps
    are reported individually (never collapsed into one mean gap).
    """
    ci = cfg.analysis.credible_interval
    turf = cfg.labels.turf_class
    reliable = cfg.analysis.reliable_sigma_n_roi
    if turf not in classes:
        return pd.DataFrame()
    rows = []
    for c in classes:
        if c == turf:
            continue
        gap = np.abs(mu[turf] - mu[c])
        # guard against division by ~0 gaps
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = sigma[turf] / gap
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size == 0:
            continue
        s = _summary(ratio, ci)
        n = cell.class_n_roi.get(c, 0)
        rows.append({
            "feature": cell.feature, "level": cell.level,
            "gap_to_class": c,
            "spread_vs_gap_median": s["median"],
            "spread_vs_gap_hdi_low": s["hdi_low"],
            "spread_vs_gap_hdi_high": s["hdi_high"],
            "spread_geq_gap": s["median"] >= 1.0,
            "n_roi": n, "n_roi_reliable": n >= reliable,
        })
    return pd.DataFrame(rows)


def all_contrasts(idata, cell: CellData, cfg) -> dict[str, pd.DataFrame]:
    """Compute all four Step-4 tables for one cell."""
    classes, mu, sigma = extract_class_posteriors(idata, cell)
    return {
        "per_class_sigma": per_class_sigma(classes, sigma, cell, cfg),
        "turf_distances": turf_distances(classes, mu, cell, cfg),
        "turf_sigma_ratios": turf_sigma_ratios(classes, sigma, cell, cfg),
        "sigma_vs_gap": sigma_vs_gap(classes, mu, sigma, cell, cfg),
    }
