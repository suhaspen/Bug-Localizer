"""Logging configuration.

One place, called once from the CLI. Mining and indexing are long-running loops
whose progress and filter decisions we need to see; the counts printed here are
the raw material for the dataset stats table.
"""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Attach a single rich handler to the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level.upper())
        return

    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # GitPython is chatty at DEBUG and drowns out our own progress lines.
    logging.getLogger("git").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
