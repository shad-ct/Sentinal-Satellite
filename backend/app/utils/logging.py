"""
Structured logging setup for the Sentinel-2 Change Detection System.

Provides a get_logger() factory that returns a consistently configured
Python logger. Uses Rich for coloured console output in development.
Log level is controlled via the LOG_LEVEL environment variable.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

try:
    from rich.logging import RichHandler

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_configured = False


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure the root logger.
    Should be called once at application startup.

    Args:
        level: Log level string (DEBUG/INFO/WARNING/ERROR).
               Falls back to LOG_LEVEL env var, then INFO.
    """
    global _configured
    if _configured:
        return

    resolved_level = level or os.environ.get("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, resolved_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    if _RICH_AVAILABLE and sys.stderr.isatty():
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
            markup=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    handlers.append(handler)

    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "rasterio", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
