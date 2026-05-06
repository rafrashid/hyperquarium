"""
utils/logger.py
Logging setup for HPC — writes to both stdout and a log file.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str | Path = "outputs/logs", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a logger that writes to stdout and to a .log file.

    Args:
        name:    Logger name (typically module or run name).
        log_dir: Directory where the .log file will be written.
        level:   Logging level (default INFO).

    Returns:
        Configured Logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # Avoid duplicate handlers on re-import

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file handler
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
