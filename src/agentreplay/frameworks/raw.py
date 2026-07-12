"""Adapter for raw / framework-less agent loops.

If you're not using LangGraph, CrewAI, AutoGen, or any SDK — just calling
the OpenAI or Anthropic client directly inside your own loop — this is
the adapter for you. It wraps the client object in place so every
``client.chat.completions.create(...)`` (or
``client.messages.create(...)``) call is transparently recorded.

Example
-------

    from agentreplay import Session
    from agentreplay.frameworks import wrap_raw_client
    from openai import OpenAI

    with Session.record("cassettes/run-1", framework="raw") as s:
        client = wrap_raw_client(OpenAI(), s, dialect="openai")
        # ... your agent loop, using `client` exactly as you would the raw client
"""
from __future__ import annotations

from typing import Any, Optional

from agentreplay.interceptors import RecordingClient, RecordingHTTP


def wrap_raw_client(
    client: Any,
    session: Any,
    *,
    dialect: str = "openai",
) -> RecordingClient:
    """Wrap a raw LLM client object for recording/replay.

    Parameters
    ----------
    client
        The real client (OpenAI / Anthropic / custom).
    session
        A :class:`agentreplay.Session` (or anything exposing
        ``wrap_openai`` / ``wrap_anthropic`` / ``wrap_custom_client``).
    dialect
        ``"openai"`` for OpenAI-style clients, ``"anthropic"`` for
        Anthropic-style clients, ``"custom"`` for objects exposing a
        ``complete(...)`` method.
    """
    if dialect == "openai":
        return session.wrap_openai(client)
    if dialect == "anthropic":
        return session.wrap_anthropic(client)
    return session.wrap_custom_client(client)


def wrap_raw_http(client: Any, session: Any, *, dialect: str = "httpx") -> RecordingHTTP:
    """Wrap an httpx / requests client for recording/replay."""
    return session.wrap_http(client, dialect=dialect)
