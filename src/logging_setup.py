"""
Logging configuration for the recommender.

Every stage of the pipeline logs through the `music_rec` logger tree, so a
run leaves a written record of what it loaded, what it skipped, and what it
refused to say. The console stays readable (warnings and up by default)
while the log file keeps the full DEBUG-level trail.
"""

import logging
import os
import sys
from typing import Optional

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "run.log")
ROOT_LOGGER_NAME = "music_rec"

_configured = False


class _LiveStderrHandler(logging.StreamHandler):
    """
    A console handler that looks up sys.stderr at emit time.

    The stock StreamHandler captures the stream once, at construction. Since
    logging is configured a single time per process, that handle goes stale
    the moment something swaps sys.stderr underneath it (pytest's capture
    fixture, a redirect, a wrapping tool) — and logging then fails on a closed
    file instead of logging. Resolving the stream late keeps output going
    wherever stderr currently points.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        # StreamHandler.__init__ assigns here; the live lookup above wins.
        pass


def configure_logging(console_level: str = "WARNING", log_file: Optional[str] = LOG_FILE) -> logging.Logger:
    """
    Sets up console + file logging once per process and returns the root
    project logger. Safe to call repeatedly; later calls are no-ops.

    If the log file can't be opened (read-only checkout, missing perms), the
    run continues with console logging only — logging must never be the
    reason the app dies.
    """
    global _configured
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if _configured:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console = _LiveStderrHandler()
    console.setLevel(getattr(logging, console_level.upper(), logging.WARNING))
    console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(console)

    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
            )
            logger.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - environment dependent
            logger.warning("File logging disabled (%s): %s", log_file, exc)

    _configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Returns a child logger, e.g. get_logger('retriever') -> music_rec.retriever."""
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
