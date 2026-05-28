"""
hac_pipeline/utils/logger.py
------------------------------
Configure logging to stdout and a file — mirrors xgb_pipeline/utils/logger.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(output_dir: Path, spectra: str) -> logging.Logger:
    """Configure root logger: stdout + hac_{spectra}.log in output_dir."""
    log_path = output_dir / f"hac_{spectra}.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="w"),
    ]

    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt,
                        handlers=handlers)

    logger = logging.getLogger("hac_pipeline")
    logger.info(f"Log file: {log_path}")
    return logger
