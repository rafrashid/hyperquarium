"""
hac_pipeline/utils/io.py
--------------------------
Shared IO helpers — mirrors xgb_pipeline/utils/io.py conventions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_serialise)
    logger.debug(f"JSON saved: {path}")


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    logger.debug(f"CSV saved: {path} ({len(df):,} rows).")


def save_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
    logger.debug(f"Parquet saved: {path} ({len(df):,} rows).")


def save_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    logger.debug(f"NumPy array saved: {path} (shape {arr.shape}).")


def _json_serialise(obj):
    """Fallback serialiser for numpy scalars and similar."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")
