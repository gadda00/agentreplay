"""Tests for the storage layer — blob store, event log, meta index."""
from __future__ import annotations

from pathlib import Path

import pytest

from agentreplay import BlobStore, EventLog, MetaIndex
from agentreplay.types import Event


def test_blob_store_put_get_roundtrip(tmp_path: Path):
    bs = BlobStore(tmp_path)
    digest = bs.put({"model": "gpt", "messages": [{"role": "user", "content": "hi"}]})
    assert isinstance(digest, str) and len(digest) == 64
    value = bs.get(digest)
    assert value["model"] == "gpt"
    assert value["messages"][0]["content"] == "hi"


def test_blob_store_deduplicates(tmp_path: Path):
    bs = BlobStore(tmp_path)
    d1 = bs.put({"x": 1})
    d2 = bs.put({"x": 1})
    assert d1 == d2
    assert len(bs) == 1


def test_blob_store_canonical_form_dedupes_reordered_dicts(tmp_path: Path):
    """Canonicalization means reordered dicts dedupe."""
    bs = BlobStore(tmp_path)
    d1 = bs.put({"a": 1, "b": 2})
    d2 = bs.put({"b": 2, "a": 1})
    assert d1 == d2
    assert len(bs) == 1


def test_blob_store_has(tmp_path: Path):
    bs = BlobStore(tmp_path)
    d = bs.put({"x": 1})
    assert bs.has(d)
    assert not bs.has("0" * 64)


def test_blob_store_missing_raises_keyerror(tmp_path: Path):
    bs = BlobStore(tmp_path)
    with pytest.raises(KeyError):
        bs.get("0" * 64)


def test_blob_store_stats(tmp_path: Path):
    bs = BlobStore(tmp_path)
    bs.put({"x": 1})
    bs.put({"y": 2})
    stats = bs.stats()
    assert stats["blobs"] == 2
    assert stats["bytes"] > 0


def test_event_log_append_iterate(tmp_path: Path):
    log = EventLog(tmp_path)
    e1 = Event(
        seq=0, step_id="step:0", call_type="llm", call_id="a" * 64,
        request_hash="r1", response_hash="s1", started_at=1.0, duration_ms=10.0,
    )
    e2 = Event(
        seq=1, step_id="step:1", call_type="tool", call_id="b" * 64,
        request_hash="r2", response_hash="s2", started_at=2.0, duration_ms=20.0,
    )
    log.append(e1)
    log.append(e2)
    assert len(log) == 2
    events = list(log)
    assert events[0].call_id == "a" * 64
    assert events[1].call_id == "b" * 64


def test_event_log_by_call_id(tmp_path: Path):
    log = EventLog(tmp_path)
    e = Event(
        seq=0, step_id="step:0", call_type="llm", call_id="c" * 64,
        request_hash="r", response_hash="s", started_at=0.0, duration_ms=0.0,
    )
    log.append(e)
    found = log.by_call_id("c" * 64)
    assert found is not None
    assert found.seq == 0
    assert log.by_call_id("z" * 64) is None


def test_event_log_by_step(tmp_path: Path):
    log = EventLog(tmp_path)
    log.append(Event(
        seq=0, step_id="langgraph:router", call_type="llm", call_id="a" * 64,
        request_hash="r", response_hash="s", started_at=0.0, duration_ms=0.0,
    ))
    log.append(Event(
        seq=1, step_id="langgraph:router", call_type="tool", call_id="b" * 64,
        request_hash="r", response_hash="s", started_at=0.0, duration_ms=0.0,
    ))
    log.append(Event(
        seq=2, step_id="langgraph:writer", call_type="llm", call_id="c" * 64,
        request_hash="r", response_hash="s", started_at=0.0, duration_ms=0.0,
    ))
    matches = log.by_step("langgraph:router")
    assert len(matches) == 2


def test_meta_index_upsert_and_get(tmp_path: Path):
    with MetaIndex(tmp_path) as idx:
        idx.upsert({
            "id": "cass-001",
            "path": str(tmp_path / "cass-001"),
            "task_id": "swe-bench:214",
            "git_commit": "a1b2c3d",
            "model": "claude-opus-4.6",
            "framework": "langgraph",
            "outcome": "fail",
            "created_at": 1000.0,
            "duration_ms": 12_000.0,
            "num_events": 42,
            "tags": ["regression", "langgraph"],
            "extra": {},
        })
        got = idx.get("cass-001")
        assert got is not None
        assert got["task_id"] == "swe-bench:214"
        assert got["outcome"] == "fail"
        assert "regression" in got["tags"]


def test_meta_index_filter_by_outcome(tmp_path: Path):
    with MetaIndex(tmp_path) as idx:
        idx.upsert({"id": "a", "path": "/a", "outcome": "pass"})
        idx.upsert({"id": "b", "path": "/b", "outcome": "fail"})
        idx.upsert({"id": "c", "path": "/c", "outcome": "fail"})
        fails = idx.list(outcome="fail")
        assert len(fails) == 2
        passes = idx.list(outcome="pass")
        assert len(passes) == 1
