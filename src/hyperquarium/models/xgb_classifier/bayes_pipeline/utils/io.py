"""
utils/io.py — saving helpers.

Convention (mirrors output-format-decisions): summary TABLES are CSV, metadata is
JSON, and the full posterior is NetCDF (idata.nc) via ArviZ — a deliberate departure
from the parquet default, because NetCDF preserves all draws for re-analysis without
refitting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save a summary table as CSV (ecology-audience friendly, diff-able)."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_json(obj: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_idata(idata, path: Path) -> None:
    """Save ArviZ InferenceData to NetCDF. Imported lazily to keep io light."""
    ensure_dir(path.parent)
    idata.to_netcdf(str(path))


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, engine="pyarrow", index=False)
