"""OpenAI SDK adapter.

Wraps an ``openai.OpenAI`` (or ``openai.AsyncOpenAI``) client so every
``client.chat.completions.create(...)`` call is captured. The wrapper
exposes the same ``chat.completions.create`` surface so it can be a
drop-in replacement for the raw client object.
"""
from __future__ import annotations

from typing import Any

from agentreplay.interceptors import RecordingClient


def wrap_openai(client: Any, session: Any, **kwargs: Any) -> RecordingClient:
    """Wrap an OpenAI client object for recording/replay.

    The returned object exposes ``.chat.completions.create(**kwargs)``
    so existing agent code does not need to change.
    """
    return session.wrap_openai(client, **kwargs)
