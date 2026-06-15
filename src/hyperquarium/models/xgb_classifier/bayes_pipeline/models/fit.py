"""
models/fit.py — fit ONE cell (one feature × one level).

Model (locked spec, see knowledge base Steps 1-3):

    feature_z[i] ~ Normal( mu[class[i]], sigma[class[i]] )
    mu[c]        ~ Normal(0, 2)                 # independent per class
    log sigma[c] ~ Normal(0, 0.5)               # log link (positivity), independent

- feature_z is the GRAND-standardised ROI-level mean of the feature (Step 1).
- Cell-means parameterisation: `feat_z ~ 0 + class` gives one mu per class directly.
- Distributional: `sigma ~ 0 + class` so each class gets its own (log-linked) sigma.
- No (1|roi_ID): after aggregation, ROI IS the observation.
- ALL classes kept (thin-class handling is a reporting filter, not a fitting filter).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from bayes_pipeline.config.config import Config


@dataclass
class CellData:
    """Standardised, model-ready data for one (feature, level) cell."""
    df: pd.DataFrame  # columns: feat_z, class
    feature: str
    level: int
    grand_mean: float  # standardisation params (for back-transform if needed)
    grand_sd: float
    class_n_roi: dict[str, int]  # ROIs per class (for n_roi annotation)


def prepare_cell(summary: pd.DataFrame, feature: str, level: int,
                 cfg: Config) -> CellData:
    """
    Slice the ROI summary to one feature × one level, grand-standardise the feature,
    and return model-ready data. Standardisation is GRAND (one mean/sd over all ROIs),
    never within-class.
    """
    class_col = f"class_L{level}"
    mean_col = f"{feature}_mean"
    if mean_col not in summary.columns:
        raise ValueError(f"{mean_col} not in ROI summary.")
    if class_col not in summary.columns:
        raise ValueError(f"{class_col} not in ROI summary.")

    sub = summary[[class_col, mean_col]].dropna().copy()
    sub = sub.rename(columns={class_col: "class", mean_col: "feat"})

    grand_mean = float(sub["feat"].mean())
    grand_sd = float(sub["feat"].std(ddof=1))
    if grand_sd == 0 or np.isnan(grand_sd):
        raise ValueError(f"Feature {feature} has zero/NaN grand SD at L{level}.")

    sub["feat_z"] = (sub["feat"] - grand_mean) / grand_sd
    sub["class"] = sub["class"].astype("category")

    class_n_roi = sub["class"].value_counts().to_dict()

    return CellData(
        df=sub[["feat_z", "class"]].reset_index(drop=True),
        feature=feature, level=level,
        grand_mean=grand_mean, grand_sd=grand_sd,
        class_n_roi={str(k): int(v) for k, v in class_n_roi.items()},
    )


def build_model(cell: CellData, cfg: Config):
    """
    Build the Bambi distributional model for one cell. Imported lazily so the rest of
    the pipeline (config, aggregation) does not require pymc/bambi at import time.
    """
    import bambi as bmb

    p = cfg.priors
    priors = {
        # Cell-means intercepts: one mu per class.
        "class": bmb.Prior("Normal", mu=0, sigma=p.mu_sd),
        # Distributional sigma terms live on the log scale (Bambi applies the log link).
        "sigma": {
            "class": bmb.Prior("Normal", mu=0, sigma=p.log_sigma_sd),
        },
    }

    model = bmb.Model(
        formula="feat_z ~ 0 + class",
        data=cell.df,
        family="gaussian",
        priors=priors,
        # Distributional component: sigma predicted by class (cell-means, no intercept).
        # Bambi syntax: pass an additional formula for the sigma parameter.
        # The 0 + class mirrors the mean structure.
    )
    # Attach the distributional sigma formula. Bambi >= 0.13 supports this via
    # `bmb.Formula`; we set it explicitly for clarity and version-safety.
    return _with_sigma_formula(cell, cfg, priors)


def _with_sigma_formula(cell: CellData, cfg: Config, priors):
    """
    Construct the model using bmb.Formula so both mu and sigma are class-varying.
    Separated out because the two-formula API differs slightly across Bambi versions.
    """
    import bambi as bmb

    formula = bmb.Formula(
        "feat_z ~ 0 + class",  # mean (mu)
        "sigma ~ 0 + class",  # log-sigma, cell-means
    )
    model = bmb.Model(
        formula=formula,
        data=cell.df,
        family="gaussian",
        priors=priors,
    )
    return model


def fit_cell(cell: CellData, cfg: Config, logger=None):
    """
    Fit one cell. Returns (model, idata). Tries nutpie if requested, falls back to
    PyMC's default NUTS if nutpie is unavailable.
    """
    from bayes_pipeline.utils.logger import get_logger
    logger = logger or get_logger()
    s = cfg.sampler

    model = build_model(cell, cfg)

    sample_kwargs = dict(
        draws=s.draws, tune=s.tune, chains=s.chains,
        target_accept=s.target_accept, random_seed=s.random_seed,
    )
    if s.use_nutpie:
        try:
            import nutpie  # noqa: F401
            sample_kwargs["inference_method"] = "nutpie"
        except Exception:
            logger.warning("nutpie unavailable; falling back to default NUTS.")

    logger.info("Fitting %s @ L%d (%d ROIs, %d classes) ...",
                cell.feature, cell.level, len(cell.df), cell.df["class"].nunique())
    idata = model.fit(**sample_kwargs)
    return model, idata


def prior_predictive(cell: CellData, cfg: Config, samples: int = 500):
    """Sample from the prior predictive (gate check). Returns idata with prior groups."""
    model = build_model(cell, cfg)
    idata = model.prior_predictive(draws=samples)
    return model, idata
