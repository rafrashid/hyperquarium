"""
utils/io.py
Helpers for saving all pipeline outputs: JSON, CSV, parquet, PNG.
Supports both CSV and parquet for tabular outputs — set OUTPUT_FORMAT in config.py.
Designed for HPC (no interactive display — matplotlib backend is Agg).
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — required on HPC
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def make_output_dir(base_dir: str | Path, spectra: str, level: int, weighted: bool = True) -> Path:
    """
    Creates and returns the output directory for a given spectra/level combination.

    Args:
        base_dir: Root output directory.
        spectra:  Spectra type label e.g. 'A', 'B', 'C', 'D'.
        level:    Hierarchy level (1, 2, or 3).
        weighted: If False, appends '_unweighted' suffix (Level 3 hypothesis baseline).

    Returns:
        Path to the created output directory.
    """
    suffix = "" if weighted else "_unweighted"
    out = Path(base_dir) / f"spectra_{spectra}" / f"level_{level}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(data: dict, path: str | Path) -> None:
    """Saves a dictionary as a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved JSON: {path}")


def load_json(path: str | Path) -> dict:
    """Loads a JSON file into a dictionary."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tabular — CSV and parquet
# ---------------------------------------------------------------------------

def save_csv(df: pd.DataFrame, path: str | Path, index: bool = True) -> None:
    """
    Saves a DataFrame as CSV.
    Preferred when sharing outputs with non-ML audiences (ecologists, biologists).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info(f"Saved CSV: {path}")


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Loads a CSV file into a DataFrame."""
    return pd.read_csv(path, **kwargs)


def save_parquet(df: pd.DataFrame, path: str | Path, index: bool = True) -> None:
    """
    Saves a DataFrame as parquet (requires pyarrow).
    Preferred for large outputs — typically 2–4x smaller than CSV and much faster to read/write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)
    logger.info(f"Saved parquet: {path}")


def load_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    """Loads a parquet file into a DataFrame (requires pyarrow)."""
    return pd.read_parquet(path, **kwargs)


def save_dataframe(df: pd.DataFrame, path: str | Path, fmt: str | None = None, index: bool = True) -> Path:
    """
    Saves a DataFrame in the configured format (parquet or CSV).
    The file extension is set automatically based on fmt — pass the path without extension,
    or with any extension (it will be overridden to match fmt).

    This is the main function to use throughout the pipeline for tabular outputs.
    Use save_csv() or save_parquet() directly only when the format must be forced.

    Args:
        df:    DataFrame to save.
        path:  Output path (extension will be set to match fmt).
        fmt:   "parquet" or "csv". Defaults to OUTPUT_FORMAT from config.
        index: Whether to include the index.

    Returns:
        Final path with corrected extension.
    """
    from config.config import OUTPUT_FORMAT
    fmt = (fmt or OUTPUT_FORMAT).lower()

    path = Path(path).with_suffix(f".{fmt}")

    if fmt == "parquet":
        save_parquet(df, path, index=index)
    elif fmt == "csv":
        save_csv(df, path, index=index)
    else:
        raise ValueError(f"Unsupported format: '{fmt}'. Use 'parquet' or 'csv'.")

    return path


def load_dataframe(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Loads a DataFrame from parquet or CSV, detected from the file extension.

    Args:
        path:   Path to the file (.parquet or .csv).
        kwargs: Passed to the underlying pandas reader.

    Returns:
        Loaded DataFrame.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return load_parquet(path, **kwargs)
    elif path.suffix == ".csv":
        return load_csv(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: '{path.suffix}'. Expected .parquet or .csv.")


def load_spectra_file(path: str | Path) -> pd.DataFrame:
    """
    Loads a spectra dataset from parquet or CSV, with size logging.
    Wrapper around load_dataframe() with informative logging for large files.

    Args:
        path: Path to the data file (.parquet or .csv).

    Returns:
        DataFrame with features and label columns.
    """
    path = Path(path)
    logger.info(f"Loading spectra data: {path}  ({path.stat().st_size / 1e9:.2f} GB)")
    df = load_dataframe(path)
    logger.info(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """
    Saves a matplotlib figure to PNG and immediately closes it.
    Always closes to prevent memory leaks when running many models.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {path}")
