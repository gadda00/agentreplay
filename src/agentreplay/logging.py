"""Centralised logging for AgentReplay.

Uses the standard ``logging`` module. The default level is WARNING
(silent in normal operation). Set ``AGENTREPLAY_LOG_LEVEL=DEBUG`` or
``AGENTREPLAY_VERBOSE=1`` to see interceptor activity.
"""
from __future__ import annotations

import logging
import os
import sys


# Track whether set_verbose() was called explicitly, so get_logger()
# doesn't override it by re-reading env vars.
_user_level_override: int | None = None


def _get_level() -> int:
    """Determine the log level from environment or user override."""
    if _user_level_override is not None:
        return _user_level_override
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
    """Enable or disable verbose (DEBUG) logging at runtime.

    Once called, this overrides any env var setting. Pass False to
    restore WARNING level.
    """
    global _user_level_override
    _user_level_override = logging.DEBUG if verbose else logging.WARNING
    root = logging.getLogger("agentreplay")
    root.setLevel(_user_level_override)
