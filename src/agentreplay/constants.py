"""Enums and constants used across the agentreplay package."""
from __future__ import annotations

from enum import Enum


class Mode(str, Enum):
    """Operating mode of an interceptor or session.

    - ``LIVE``    : pass through to real client, do not record.
    - ``RECORD``  : call real client and write the request/response to the cassette.
    - ``REPLAY``  : serve from cassette; never call real client.
    - ``HYBRID``  : serve from cassette until first divergence, then fall through to live.
    """

    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"
    HYBRID = "hybrid"


class CallType(str, Enum):
    """Kind of intercepted call. Stored on every event row."""

    LLM = "llm"
    TOOL = "tool"
    HTTP = "http"
    CLOCK = "clock"
    RNG = "rng"
    OTHER = "other"


# Default cassette schema version. Bumped when the on-disk layout changes
# in a backwards-incompatible way.
CASSETTE_VERSION = "1.0.0"
