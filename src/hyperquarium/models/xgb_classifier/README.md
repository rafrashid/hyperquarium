## Project structure

```
xgb_pipeline/
├── config/
│   └── config.py           # All parameters, paths, model settings
├── data/
│   └── loader.py           # Data loading, splitting, DMatrix creation
├── models/
│   └── trainer.py          # Training, early stopping, model saving
├── evaluation/
│   └── evaluator.py        # Metrics, confusion matrices, PR curves
├── features/
│   └── shap_analysis.py    # SHAP values, feature importance, scale-response
├── utils/
│   └── logger.py           # Logging setup
│   └── io.py               # Saving outputs (CSV, JSON, PNG)
└── run_pipeline.py         # Main entry point — orchestrates all stages
```

## Usage

```bash
# Run full pipeline for all spectra types and levels
python run_pipeline.py

# Run for a single spectra type
python run_pipeline.py --spectra A

# Run a specific stage only
python run_pipeline.py --spectra A --level 3 --stage train
python run_pipeline.py --spectra A --level 3 --stage evaluate
python run_pipeline.py --spectra A --level 3 --stage shap
```

## Output directory structure

```
outputs/
├── spectra_A/
│   ├── level_1/
│   │   ├── model.json
│   │   ├── metrics.json
│   │   ├── evals_result.json
│   │   ├── confusion_matrix.csv
│   │   ├── confusion_matrix.png
│   │   ├── pr_curve.png
│   │   ├── learning_curve.png
│   │   ├── feature_importance.csv
│   │   ├── shap_values.parquet
│   │   ├── shap_summary.csv
│   │   ├── shap_dependence_<feature>.png
│   │   └── scale_response.csv
│   ├── level_2/
│   └── level_3/
│       └── (also: level_3_unweighted/ for hypothesis baseline)
├── spectra_B/
├── spectra_C/
├── spectra_D/
└── cross_spectra/
    ├── shap_comparison.csv
    └── feature_consistency.csv
```
