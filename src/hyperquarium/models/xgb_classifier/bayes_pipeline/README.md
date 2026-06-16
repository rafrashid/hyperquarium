# bayes_pipeline

Bayesian ROI-level feature analysis — a standalone investigation, separate from the
`xgb_pipeline` and `hac_pipeline`. Asks how the **7 non-spectral features** (3 specdiv:
gamma/alpha/beta + 4 GLCM: energy/entropy/homogeneity/contrast) vary across the label
tiers L1–L3, with full uncertainty quantification.

See the knowledge base note `bayesian-roi-feature-analysis` for the full rationale.

## What it does

1. **Aggregate** the compiled pixel-level parquet to **one row per ROI** (mean, se, n
   per feature). ROI is the unit of analysis — pixels are pseudoreplicates. This is
   the only big-RAM step; its output is cached.
2. **Fit** a Bayesian distributional model per (feature × level), on grand-standardised
   features:

   ```
   feature_z[i] ~ Normal( mu[class[i]], sigma[class[i]] )
   mu[c]        ~ Normal(0, 2)
   log sigma[c] ~ Normal(0, 0.5)        # log link for positivity
   ```

   Cell-means (`0 + class`), independent per class (no pooling), no ROI random effect
   (handled by aggregation).
3. **Contrast** (Step 4, posterior arithmetic, no refit):
    - per-class sigma (baseline, every class)
    - turf-referenced mu-distances
    - turf-referenced sigma-ratios (from log-sigma differences)
    - turf sigma vs each between-class gap

Every output table carries `n_roi` so each row self-documents how much data backs it.

## Thin-class handling

All classes are kept in the fit (`min_rois_for_sigma = 1`). A 1-ROI class returns its
sigma **prior** unchanged (no within-class spread to estimate) — this is not an error,
it is the data saying nothing about that class's spread. The `reliable_sigma_n_roi`
threshold (default 15) is a **reporting/visualisation filter** (annotate / grey-out),
never a fitting filter. Check `diagnostics.csv` per cell — thin classes can raise
r_hat / lower ess on their sigma params.

## Usage

One dataset per invocation (pilot and reefcompare run separately).

```bash
# 1. Aggregate (big-RAM; once per dataset). Submit pbs_aggregate.sh or run directly:
python scripts/aggregate.py --parquet data/compiled_dataset.parquet \
    --mapping data/labelset_mapping.csv --output outputs/bayes --labelset reefcompare

# 2. Prototype ONE cell first (prove the workflow):
python scripts/fit_one.py --feature homogeneity --level 2 \
    --output outputs/bayes --labelset reefcompare --prior-check

# 3. Run all 21 cells:
python scripts/run_all.py --parquet data/compiled_dataset.parquet \
    --mapping data/labelset_mapping.csv --output outputs/bayes --labelset reefcompare
```

`--labelset` filters `labelset_mapping.csv` (default `reefcompare`; use `pilot` for the
pilot dataset) — same convention as `train.py`/`evaluate.py`/`predict.py`.
Set `--glcm-window` / `--specdiv-plot` to choose the representative size (default 25).

## Layout

```
bayes_pipeline/
├── config/config.py        — all params (dataclasses); single edit point
├── data/
│   ├── labels.py           — apply labelset_mapping.csv -> class_L1/L2/L3
│   └── aggregate.py        — parquet -> roi_summary.parquet (heavy, cached)
├── models/fit.py           — prepare_cell, build distributional model, fit
├── analysis/
│   ├── contrasts.py        — Step-4 posterior arithmetic (4 tables)
│   └── diagnostics.py      — r_hat / ess / divergences per cell
├── utils/{io,logger}.py
├── run_bayes.py            — orchestrator (run_one_cell, run_all)
└── scripts/
    ├── aggregate.py        — standalone aggregation CLI
    ├── fit_one.py          — single-cell prototype CLI
    ├── run_all.py          — full 21-cell CLI
    └── pbs_aggregate.sh    — PBS job for the aggregation stage
```

## Outputs

```
outputs/bayes/
├── roi_summary.parquet
├── <feature>/level_<1|2|3>/
│   ├── idata.nc              # full posterior (ArviZ NetCDF)
│   ├── per_class_sigma.csv
│   ├── turf_distances.csv
│   ├── turf_sigma_ratios.csv
│   ├── sigma_vs_gap.csv
│   ├── diagnostics.csv
│   └── metadata.json
```
