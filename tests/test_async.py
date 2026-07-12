"""Tests for the async LLM interceptor (acomplete)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentreplay import Cassette, Recorder, Replayer
from agentreplay.constants import Mode


class _AsyncStubLLM:
    """Async stub LLM with an ``acomplete`` coroutine method."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.live_calls = 0

    async def acomplete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        self.live_calls += 1
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


class _SyncStubLLM:
    """Sync-only stub — the async wrapper should fall back to a thread."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.live_calls = 0

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        self.live_calls += 1
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_async_record_with_async_client(tmp_path: Path):
    """Recording with an async client (with acomplete) should await it."""
    cassette = tmp_path / "cass"
    stub = _AsyncStubLLM([
        {"text": "async reply", "usage": {}},
    ])

    with Recorder.create(cassette, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        r = await client.acomplete(messages=[{"role": "user", "content": "hi"}], model="stub")

    assert r == {"text": "async reply", "usage": {}}
    assert stub.live_calls == 1

    c = Cassette.open(cassette, readonly=True)
    events = list(c.events)
    assert len(events) == 1
    assert c.records()[0].response == {"text": "async reply", "usage": {}}
    # The async flag should be recorded in metadata.
    assert events[0].metadata.get("async") is True


@pytest.mark.asyncio
async def test_async_record_falls_back_to_sync_in_thread(tmp_path: Path):
    """If the real client has no ``acomplete``, acomplete should fall
    back to running sync complete in a thread."""
    cassette = tmp_path / "cass"
    stub = _SyncStubLLM([
        {"text": "sync fallback", "usage": {}},
    ])

    with Recorder.create(cassette, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        r = await client.acomplete(messages=[{"role": "user", "content": "hi"}], model="stub")

    assert r == {"text": "sync fallback", "usage": {}}
    assert stub.live_calls == 1


@pytest.mark.asyncio
async def test_async_replay_returns_recorded_value_without_live_call(tmp_path: Path):
    """Pure async replay should return the recorded value without
    touching the real client."""
    cassette = tmp_path / "cass"
    stub = _AsyncStubLLM([{"text": "recorded", "usage": {}}])
    with Recorder.create(cassette, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        await client.acomplete(messages=[{"role": "user", "content": "hi"}], model="stub")

    # Replay with a fresh stub that has NO responses — any live call would raise.
    fresh = _AsyncStubLLM(responses=[])
    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(fresh)
        r = await client.acomplete(messages=[{"role": "user", "content": "hi"}], model="stub")

    assert r == {"text": "recorded", "usage": {}}
    assert fresh.live_calls == 0
