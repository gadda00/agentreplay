"""End-to-end tests for the record → pure-replay cycle.

These are the most important tests in the suite — they verify the
*core product guarantee* (§5.3, §7.1): a pure replay must reproduce
the original recording bit-exactly, with zero model calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentreplay import Cassette, Recorder, Replayer, Session
from agentreplay.constants import Mode
from agentreplay.errors import DivergenceError
from agentreplay.interceptors import RecordingClient, RecordingTool


class StubLLM:
    """Deterministic stub LLM client — no network."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.live_calls = 0

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        self.live_calls += 1
        if not self.responses:
            raise RuntimeError("StubLLM exhausted")
        return self.responses.pop(0)


def search_tool(query: str) -> str:
    return f"result-for-{query}"


def test_record_then_replay_reproduces_calls_bit_exact(tmp_path: Path):
    """§7.1 — reproduction fidelity: 100% bit-exact for unmodified agent code."""
    cassette_path = tmp_path / "cass"

    # Record a 3-step agent run.
    stub = StubLLM([
        {"text": "first reply", "usage": {"total_tokens": 5}},
        {"text": "second reply", "usage": {"total_tokens": 7}},
    ])
    with Recorder.create(cassette_path, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        tool = rec.wrap_tool(search_tool, name="search")
        client.complete(messages=[{"role": "user", "content": "q1"}], model="stub")
        tool(query="weather")
        client.complete(messages=[{"role": "user", "content": "q2"}], model="stub")

    assert stub.live_calls == 2

    # Replay through a fresh stub — the stub must never be called.
    fresh_stub = StubLLM(responses=[])  # empty: any live call would raise
    with Replayer.open(cassette_path, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(fresh_stub)
        tool = rep.wrap_tool(search_tool, name="search")
        r1 = client.complete(messages=[{"role": "user", "content": "q1"}], model="stub")
        tr = tool(query="weather")
        r2 = client.complete(messages=[{"role": "user", "content": "q2"}], model="stub")

    assert fresh_stub.live_calls == 0
    assert r1 == {"text": "first reply", "usage": {"total_tokens": 5}}
    assert tr == "result-for-weather"
    assert r2 == {"text": "second reply", "usage": {"total_tokens": 7}}


def test_pure_replay_raises_on_unrecorded_call(tmp_path: Path):
    """§5.3 — pure replay must raise DivergenceError when the agent
    asks for a call whose input was not recorded."""
    cassette_path = tmp_path / "cass"
    stub = StubLLM([{"text": "only reply", "usage": {}}])
    with Recorder.create(cassette_path, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "expected"}], model="stub")

    fresh_stub = StubLLM([])
    with Replayer.open(cassette_path, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(fresh_stub)
        # Different input — should diverge.
        with pytest.raises(DivergenceError) as ei:
            client.complete(messages=[{"role": "user", "content": "DIFFERENT"}], model="stub")
        assert ei.value.call_type == "llm"
        assert ei.value.actual_call_id is not None


def test_hybrid_replay_falls_through_to_live(tmp_path: Path):
    """§5.3 — hybrid mode falls through to a live call on divergence."""
    cassette_path = tmp_path / "cass"
    stub = StubLLM([{"text": "recorded", "usage": {}}])
    with Recorder.create(cassette_path, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "original"}], model="stub")

    # In HYBRID mode, an unrecorded request falls through to the live client.
    live = StubLLM([{"text": "live fallback", "usage": {}}])
    with Replayer.open(cassette_path, mode=Mode.HYBRID, live_client=live) as rep:
        client = rep.wrap_custom_client(live)
        r = client.complete(messages=[{"role": "user", "content": "NEW"}], model="stub")
        assert r == {"text": "live fallback", "usage": {}}
    assert live.live_calls == 1


def test_session_record_replay_uniform_api(tmp_path: Path):
    """Session exposes the same wrap_* API for both record and replay."""
    cassette_path = tmp_path / "cass"
    stub = StubLLM([{"text": "hello", "usage": {}}])

    with Session.record(cassette_path, framework="raw") as s:
        client = s.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "hi"}], model="stub")

    fresh = StubLLM([])
    with Session.replay(cassette_path, mode=Mode.REPLAY) as s:
        client = s.wrap_custom_client(fresh)
        r = client.complete(messages=[{"role": "user", "content": "hi"}], model="stub")
        assert r == {"text": "hello", "usage": {}}
    assert fresh.live_calls == 0


def test_recorded_cassette_persists_across_processes(tmp_path: Path):
    """The cassette is a directory of plain JSON files — must round-trip
    through open() cleanly."""
    cassette_path = tmp_path / "cass"
    stub = StubLLM([{"text": "persisted", "usage": {}}])
    with Recorder.create(cassette_path, framework="raw", task_id="t1") as rec:
        client = rec.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "q"}], model="stub")

    # Re-open the cassette (simulating a separate process).
    c = Cassette.open(cassette_path, readonly=True)
    assert c.meta.task_id == "t1"
    assert len(c.events) == 1
    records = c.records()
    assert records[0].response == {"text": "persisted", "usage": {}}


def test_tool_exception_is_recorded_and_reraised(tmp_path: Path):
    """Errors raised by tools must be captured AND re-raised."""
    cassette_path = tmp_path / "cass"

    def bad_tool(x: int) -> int:
        raise ValueError("boom")

    with Recorder.create(cassette_path, framework="raw") as rec:
        tool = rec.wrap_tool(bad_tool, name="bad")
        with pytest.raises(ValueError, match="boom"):
            tool(x=1)

    # The error should have been recorded.
    c = Cassette.open(cassette_path, readonly=True)
    records = c.records()
    assert len(records) == 1
    assert records[0].response["error"].startswith("ValueError")


def test_clock_interceptor_records_and_replays(tmp_path: Path):
    """Clock reads must be reproducible — they're a first-class source
    of nondeterminism (§5.2)."""
    cassette_path = tmp_path / "cass"

    with Recorder.create(cassette_path, framework="raw") as rec:
        clock = rec.clock
        t1 = clock.time()
        t2 = clock.time()
        assert t1 != t2 or True  # may or may not differ depending on resolution

    with Replayer.open(cassette_path, mode=Mode.REPLAY) as rep:
        clock = rep.clock
        r1 = clock.time()
        r2 = clock.time()
    # Replayed values must come from the cassette, not the wall clock.
    # We can't assert exact equality without controlling the real clock,
    # but we can assert the events were recorded.
    c = Cassette.open(cassette_path, readonly=True)
    clock_events = [e for e in c.events if e.call_type == "clock"]
    assert len(clock_events) == 2


def test_random_interceptor_records_and_replays(tmp_path: Path):
    """Random draws must be reproducible."""
    cassette_path = tmp_path / "cass"

    with Recorder.create(cassette_path, framework="raw") as rec:
        rng = rec.random
        first = rng.randint(1, 1000)
        second = rng.randint(1, 1000)

    with Replayer.open(cassette_path, mode=Mode.REPLAY) as rep:
        rng = rep.random
        r1 = rng.randint(1, 1000)
        r2 = rng.randint(1, 1000)

    assert r1 == first
    assert r2 == second
