"""Anthropic SDK adapter.

Wraps an ``anthropic.Anthropic`` (or ``anthropic.AsyncAnthropic``)
client so every ``client.messages.create(...)`` call is captured.
"""
from __future__ import annotations

from typing import Any

from agentreplay.interceptors import RecordingClient


def wrap_anthropic(client: Any, session: Any, **kwargs: Any) -> RecordingClient:
    """Wrap an Anthropic client object for recording/replay.

    The returned object exposes ``.messages.create(**kwargs)`` so
    existing agent code does not need to change.
    """
    return session.wrap_anthropic(client, **kwargs)
