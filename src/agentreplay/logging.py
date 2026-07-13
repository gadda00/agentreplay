"""Centralised logging for AgentReplay.

Uses the standard ``logging`` module. The default level is WARNING
(silent in normal operation). Set ``AGENTREPLAY_LOG_LEVEL=DEBUG`` or
``AGENTREPLAY_VERBOSE=1`` to see interceptor activity.
"""
from __future__ import annotations

import logging
import os
import sys


def _get_level() -> int:
    """Determine the log level from environment variables."""
    if os.environ.get("AGENTREPLAY_VERBOSE", "").lower() in ("1", "true", "yes"):
        return logging.DEBUG
    level_name = os.environ.get("AGENTREPLAY_LOG_LEVEL", "WARNING").upper()
    return getattr(logging, level_name, logging.WARNING)


def _configure_root_logger() -> logging.Logger:
    """Configure the 'agentreplay' root logger once."""
    root = logging.getLogger("agentreplay")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "[%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.setLevel(_get_level())
    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'agentreplay' namespace."""
    _configure_root_logger()
    if name.startswith("agentreplay"):
        return logging.getLogger(name)
    return logging.getLogger(f"agentreplay.{name}")


def set_verbose(verbose: bool) -> None:
    """Enable or disable verbose (DEBUG) logging at runtime."""
    root = logging.getLogger("agentreplay")
    root.setLevel(logging.DEBUG if verbose else logging.WARNING)
