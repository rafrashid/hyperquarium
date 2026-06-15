"""
analysis/diagnostics.py — sampler health per cell.

With the thin-class threshold at 1, thin classes can cause divergences / low ess /
elevated r_hat on their sigma parameters. These do not crash the fit; this module
surfaces them so a cell is flagged for inspection rather than silently trusted.
"""
from __future__ import annotations

import pandas as pd


def diagnostics_table(idata, cfg) -> pd.DataFrame:
    """
    Per-parameter r_hat and ess, plus a boolean flag against config thresholds.
    Returns a tidy DataFrame.
    """
    import arviz as az

    summ = az.summary(idata, kind="diagnostics")  # r_hat, ess_bulk, ess_tail, mcse
    summ = summ.reset_index().rename(columns={"index": "parameter"})

    a = cfg.analysis
    summ["rhat_flag"] = summ.get("r_hat", 1.0) > a.rhat_warn
    if "ess_bulk" in summ.columns:
        summ["ess_flag"] = summ["ess_bulk"] < a.ess_warn
    else:
        summ["ess_flag"] = False
    summ["flagged"] = summ["rhat_flag"] | summ["ess_flag"]
    return summ


def divergence_count(idata) -> int:
    """Number of divergent transitions, if recorded."""
    try:
        return int(idata.sample_stats["diverging"].sum())
    except Exception:
        return -1  # not available


def diagnostics_summary(idata, cfg) -> dict:
    """Compact dict for the per-cell metadata JSON."""
    tbl = diagnostics_table(idata, cfg)
    return {
        "n_params": int(len(tbl)),
        "n_flagged": int(tbl["flagged"].sum()),
        "flagged_params": tbl.loc[tbl["flagged"], "parameter"].tolist(),
        "max_rhat": float(tbl["r_hat"].max()) if "r_hat" in tbl else None,
        "min_ess_bulk": float(tbl["ess_bulk"].min()) if "ess_bulk" in tbl else None,
        "divergences": divergence_count(idata),
    }
