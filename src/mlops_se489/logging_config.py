"""Centralized logging configuration for the ML pipeline.

This module configures both console logging and file logging. Console logs use
RichHandler for readable terminal output, while file logs use RotatingFileHandler
to prevent log files from growing forever.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Literal

from rich.logging import RichHandler

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "mlops_se489.log"

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: LogLevel = "INFO",
    log_file: Path = LOG_FILE,
    max_bytes: int = 1_000_000,
    backup_count: int = 3,
) -> None:
    """Configure application logging.

    This sets up:
    - Rich console logging for readable terminal output.
    - Rotating file logging under logs/ to prevent unlimited file growth.

    Args:
        level: Logging level such as DEBUG, INFO, WARNING, ERROR, or CRITICAL.
        log_file: Path where application logs should be written.
        max_bytes: Maximum size of one log file before rotation.
        backup_count: Number of rotated backup log files to keep.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers so repeated calls do not duplicate log messages.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
        console=None,
    )
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(fmt=DEFAULT_FORMAT, datefmt=DEFAULT_DATEFMT)
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Capture warnings from libraries.
    logging.captureWarnings(True)

    root_logger.info("Logging initialized")
    root_logger.info("Log file: %s", log_file)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.

    Args:
        name: Usually __name__ from the calling module.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)