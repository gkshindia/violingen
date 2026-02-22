"""
logging.py
~~~~~~~~~~

Centralised logging configuration for the violingen package.

Usage
-----
    from violingen.logging import get_logger, log_batch_start, ...

    logger = get_logger("violingen.orchestrator")
    log_batch_start(logger, n_files=10, device="mps", max_workers=1)
"""

from __future__ import annotations

import logging
import traceback


# ---------------------------------------------------------------------------
# ANSI colour codes (no external dependency)
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREY   = "\033[90m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"

_LEVEL_COLOURS = {
    "DEBUG":    _GREY,
    "INFO":     _CYAN,
    "WARNING":  _YELLOW,
    "ERROR":    _RED,
    "CRITICAL": _BOLD + _RED,
}

# Track which loggers have already been configured so we don't add duplicate
# handlers on repeated calls (e.g. in notebooks / interactive sessions).
_configured: set[str] = set()


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class _ColourFormatter(logging.Formatter):
    """StreamHandler formatter with per-level ANSI colouring."""

    _FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _DATE_FMT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        record.name      = f"{_GREY}{record.name}{_RESET}"
        record.asctime   = self.formatTime(record, self._DATE_FMT)
        return logging.Formatter(self._FMT, datefmt=self._DATE_FMT).format(record)


# ---------------------------------------------------------------------------
# Public: logger factory
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return (and lazily configure) a named logger.

    The logger is configured exactly once regardless of how many times
    ``get_logger`` is called with the same *name*.

    Parameters
    ----------
    name : str
        Logger name, e.g. ``"violingen.orchestrator"``.
    level : int
        Minimum log level.  Defaults to ``logging.DEBUG``.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if name in _configured:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(_ColourFormatter())
    logger.addHandler(handler)
    logger.propagate = False  # prevent double-printing via root logger
    _configured.add(name)
    return logger


# ---------------------------------------------------------------------------
# Public: structured log helpers
# ---------------------------------------------------------------------------

def log_batch_start(
    logger: logging.Logger,
    n_files: int,
    device: str,
    max_workers: int,
    model: str,
    stem: str,
) -> None:
    """Log a structured INFO banner before batch processing begins."""
    logger.info(
        "─" * 60
    )
    logger.info(
        "violingen | batch start  "
        f"files={n_files}  device={device}  "
        f"workers={max_workers}  model={model}  stem={stem}"
    )
    logger.info("─" * 60)


def log_file_result(
    logger: logging.Logger,
    in_path: str,
    out_path: str,
    elapsed_s: float,
) -> None:
    """Log a per-file SUCCESS line including the processing duration."""
    from violingen.utils import format_elapsed

    logger.info(
        f"✓  {in_path}  →  {out_path}  [{format_elapsed(elapsed_s)}]"
    )


def log_file_error(
    logger: logging.Logger,
    in_path: str,
    exc: BaseException,
) -> None:
    """Log a per-file ERROR line with full traceback."""
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(
        f"✗  {in_path}  —  {type(exc).__name__}: {exc}\n{tb}"
    )


def log_batch_summary(
    logger: logging.Logger,
    n_ok: int,
    n_fail: int,
    total_elapsed_s: float,
) -> None:
    """Log a final INFO summary after the batch completes."""
    from violingen.utils import format_elapsed

    logger.info("─" * 60)
    logger.info(
        f"violingen | batch done   "
        f"ok={n_ok}  failed={n_fail}  "
        f"total={format_elapsed(total_elapsed_s)}"
    )
    logger.info("─" * 60)
