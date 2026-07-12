"""Tests for the Cassette class — the central abstraction."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentreplay import Cassette, CassetteMeta
from agentreplay.constants import CallType
from agentreplay.errors import CassetteError, CassetteNotFoundError


def test_cassette_create_writes_meta(tmp_path: Path):
    c = Cassette.create(
        tmp_path / "c",
        framework="langgraph",
        agent_name="my-agent",
        task_id="swe-bench:214",
        model="claude-opus-4.6",
        outcome="fail",
        tags=["regression"],
    )
    assert (tmp_path / "c" / "cassette.json").exists()
    meta = CassetteMeta.from_dict(
        __import__("json").loads((tmp_path / "c" / "cassette.json").read_text())
    )
    assert meta.framework == "langgraph"
    assert meta.task_id == "swe-bench:214"
    assert meta.outcome == "fail"


def test_cassette_open_raises_when_missing(tmp_path: Path):
    with pytest.raises(CassetteNotFoundError):
        Cassette.open(tmp_path / "nope")


def test_cassette_create_rejects_nonempty_dir(tmp_path: Path):
    (tmp_path / "existing.txt").write_text("x")
    with pytest.raises(CassetteError):
        Cassette.create(tmp_path)


def test_cassette_write_event_stores_payloads(tmp_path: Path):
    c = Cassette.create(tmp_path / "c")
    ev = c.write_event(
        step_id="step:0",
        call_type=CallType.LLM,
        call_id="a" * 64,
        request={"messages": [{"role": "user", "content": "hi"}]},
        response={"text": "hello"},
        started_at=1.0,
        duration_ms=10.0,
    )
    assert ev.seq == 0
    assert ev.request_hash != ev.response_hash
    # Blob store should have two blobs.
    assert len(c.blobs) == 2
    # Event log should have one row.
    assert len(c.events) == 1


def test_cassette_lookup_call(tmp_path: Path):
    c = Cassette.create(tmp_path / "c")
    c.write_event(
        step_id="step:0", call_type=CallType.LLM, call_id="abc123",
        request={"x": 1}, response={"y": 2}, started_at=0.0, duration_ms=0.0,
    )
    found = c.lookup_call("abc123")
    assert found is not None
    assert found.call_id == "abc123"
    assert c.lookup_call("nonexistent") is None


def test_cassette_resolve_request_response(tmp_path: Path):
    c = Cassette.create(tmp_path / "c")
    ev = c.write_event(
        step_id="step:0", call_type=CallType.LLM, call_id="abc",
        request={"messages": ["hi"]}, response={"text": "hello"},
        started_at=0.0, duration_ms=0.0,
    )
    assert c.resolve_request(ev) == {"messages": ["hi"]}
    assert c.resolve_response(ev) == {"text": "hello"}


def test_cassette_iter_records(tmp_path: Path):
    c = Cassette.create(tmp_path / "c")
    c.write_event(
        step_id="step:0", call_type=CallType.LLM, call_id="a",
        request={"x": 1}, response={"y": 1}, started_at=0.0, duration_ms=0.0,
    )
    c.write_event(
        step_id="step:1", call_type=CallType.TOOL, call_id="b",
        request={"x": 2}, response={"y": 2}, started_at=0.0, duration_ms=0.0,
    )
    records = list(c.iter_records())
    assert len(records) == 2
    assert records[0].response == {"y": 1}
    assert records[1].response == {"y": 2}


def test_cassette_fork_reuses_blobs(tmp_path: Path):
    c = Cassette.create(tmp_path / "original")
    c.write_event(
        step_id="step:0", call_type=CallType.LLM, call_id="a",
        request={"x": 1}, response={"y": 1}, started_at=0.0, duration_ms=0.0,
    )
    forked = c.fork(tmp_path / "fork")
    # Same blob count (deduped via hardlink).
    assert len(forked.blobs) == len(c.blobs)
    # Same events.
    assert len(forked.events) == len(c.events)
    # Fork is writable.
    assert not forked.readonly


def test_cassette_replace_response(tmp_path: Path):
    c = Cassette.create(tmp_path / "c")
    c.write_event(
        step_id="step:0", call_type=CallType.LLM, call_id="a",
        request={"x": 1}, response={"y": 1}, started_at=0.0, duration_ms=0.0,
    )
    patched = c.replace_response(0, {"y": 999})
    events = c.events.all()
    assert events[0].response_hash == patched.response_hash
    # The new response is now resolved.
    assert c.resolve_response(events[0]) == {"y": 999}
    # Request hash unchanged → call-site ID still matchable.
    assert events[0].request_hash == patched.request_hash


def test_cassette_stats(tmp_path: Path):
    c = Cassette.create(tmp_path / "c", framework="raw", task_id="t1", outcome="pass")
    c.write_event(
        step_id="step:0", call_type=CallType.LLM, call_id="a",
        request={"x": 1}, response={"y": 1}, started_at=0.0, duration_ms=0.0,
    )
    s = c.stats()
    assert s["framework"] == "raw"
    assert s["task_id"] == "t1"
    assert s["outcome"] == "pass"
    assert s["num_events"] == 1
    assert s["blobs"]["blobs"] == 2


def test_cassette_readonly_blocks_write(tmp_path: Path):
    c = Cassette.create(tmp_path / "c")
    c.readonly = True
    with pytest.raises(CassetteError):
        c.write_event(
            step_id="step:0", call_type=CallType.LLM, call_id="a",
            request={"x": 1}, response={"y": 1}, started_at=0.0, duration_ms=0.0,
        )
