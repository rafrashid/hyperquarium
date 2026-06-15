"""utils/logger.py — stdout + file logging, mirroring xgb_pipeline/utils/logger.py."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str = "bayes_pipeline", logfile: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
