# project_alpha/utils/logging.py
from __future__ import annotations

import logging
import sys
from typing import Optional

# Try to integrate cleanly with tqdm progress bars if present
try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False


_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


class _TqdmHandler(logging.StreamHandler):
    """
    A stream handler that writes via tqdm.write() when tqdm is in use,
    so log lines don't break progress bars.
    """
    def __init__(self, stream=None):
        super().__init__(stream or sys.stderr)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if _HAS_TQDM:
                tqdm.write(msg)
            else:
                self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


_configured_once = False


def _quiet_third_party() -> None:
    """
    Reduce noise from common libraries during backtests.
    """
    noisy = [
        "urllib3",
        "numexpr",
        "lightgbm",
        "transformers",
        "yfinance",
        "matplotlib",
        "pyarrow",
        "fsspec",
        "asyncio",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.ERROR)


def get_logger(
    name: str = "project_alpha",
    level: str | int = "INFO",
    to_stdout: bool = True,
    propagate: bool = False,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Create (or return) a configured logger.

    Parameters
    ----------
    name : logger name
    level : str|int  ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    to_stdout : if True, attach a stream handler (tqdm-friendly)
    propagate : if False, do not propagate to parent/root
    fmt : custom log format (optional)
    """
    global _configured_once

    # Normalize level
    if isinstance(level, str):
        level = _LEVELS.get(level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    # Configure handler only once per logger
    if to_stdout and not any(isinstance(h, _TqdmHandler) for h in logger.handlers):
        handler = _TqdmHandler(stream=sys.stderr)
        formatter = logging.Formatter(
            fmt or "%(asctime)s | %(levelname)s | %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

    # One-time global noise reduction
    if not _configured_once:
        _quiet_third_party()
        _configured_once = True

    return logger


def set_log_level(logger: logging.Logger, level: str | int = "INFO") -> None:
    """Convenience helper to adjust an existing logger's level."""
    if isinstance(level, str):
        level = _LEVELS.get(level.upper(), logging.INFO)
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
