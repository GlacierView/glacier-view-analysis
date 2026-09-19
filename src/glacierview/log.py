"""Logging setup, so long-running jobs report progress the same way.

Named ``log`` rather than ``logging`` to avoid shadowing the standard library
module inside this package.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure(verbose: bool = False, logfile: str | None = None) -> None:
    """Install a root handler. Call once, from an entry point."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=_FORMAT,
        handlers=handlers,
        force=True,
    )
    # These are chatty at DEBUG and rarely what you are debugging.
    for noisy in ("matplotlib", "PIL", "rasterio", "botocore", "urllib3", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
