"""Tests for the EventLog in-memory index (O(1) lookups)."""
from __future__ import annotations

from pathlib import Path

from agentreplay.types import Event


def _make_event(seq: int, call_id: str, step_id: str = "step:0") -> Event:
    return Event(
        seq=seq,
        step_id=step_id,
        call_type="llm",
        call_id=call_id,
        request_hash=f"req{seq}",
        response_hash=f"resp{seq}",
        started_at=float(seq),
        duration_ms=float(seq * 10),
        metadata={},
    )


def test_event_log_index_is_lazy(tmp_path: Path):
    """The index should not be built until the first read."""
    from agentreplay.storage.event_log import EventLog
    log = EventLog(tmp_path)
    assert log._index_dirty is True
    assert len(log) == 0  # triggers index build
    assert log._index_dirty is False


def test_event_log_by_call_id_is_o1(tmp_path: Path):
    """by_call_id should find events by call_id in O(1)."""
    from agentreplay.storage.event_log import EventLog
    log = EventLog(tmp_path)
    log.append(_make_event(0, "aaa"))
    log.append(_make_event(1, "bbb"))
    log.append(_make_event(2, "ccc"))

    assert log.by_call_id("bbb") is not None
    assert log.by_call_id("bbb").seq == 1
    assert log.by_call_id("zzz") is None


def test_event_log_by_step_is_o1(tmp_path: Path):
    """by_step should find all events for a step_id in O(k)."""
    from agentreplay.storage.event_log import EventLog
    log = EventLog(tmp_path)
    log.append(_make_event(0, "a", step_id="langgraph:router"))
    log.append(_make_event(1, "b", step_id="langgraph:router"))
    log.append(_make_event(2, "c", step_id="langgraph:synth"))

    router_events = log.by_step("langgraph:router")
    assert len(router_events) == 2
    synth_events = log.by_step("langgraph:synth")
    assert len(synth_events) == 1
    assert log.by_step("nonexistent") == []


def test_event_log_at_is_o1(tmp_path: Path):
    """at() should find events by seq number in O(1)."""
    from agentreplay.storage.event_log import EventLog
    log = EventLog(tmp_path)
    log.append(_make_event(0, "a"))
    log.append(_make_event(1, "b"))
    log.append(_make_event(2, "c"))

    assert log.at(0).call_id == "a"
    assert log.at(2).call_id == "c"
    assert log.at(99) is None


def test_event_log_rebuild_index(tmp_path: Path):
    """rebuild_index should rebuild from the file."""
    from agentreplay.storage.event_log import EventLog
    log = EventLog(tmp_path)
    log.append(_make_event(0, "a"))
    log.append(_make_event(1, "b"))

    # Mutate the file externally (simulating replace_response)
    # Then rebuild
    log.rebuild_index()
    assert len(log) == 2
    assert log.by_call_id("a") is not None
    assert log.by_call_id("b") is not None


def test_event_log_count_after_append(tmp_path: Path):
    """len() should stay accurate after appends."""
    from agentreplay.storage.event_log import EventLog
    log = EventLog(tmp_path)
    assert len(log) == 0
    log.append(_make_event(0, "a"))
    assert len(log) == 1
    log.append(_make_event(1, "b"))
    assert len(log) == 2
    log.append(_make_event(2, "c"))
    assert len(log) == 3


def test_event_log_index_survives_corrupted_lines(tmp_path: Path):
    """Corrupted JSONL lines should be skipped, not crash."""
    from agentreplay.storage.event_log import EventLog
    log = EventLog(tmp_path)
    log.append(_make_event(0, "a"))
    # Append a corrupted line manually
    with log.path.open("a") as f:
        f.write("{corrupted json\n")
    log.append(_make_event(1, "b"))

    log.rebuild_index()
    assert len(log) == 2  # corrupted line skipped
    assert log.by_call_id("a") is not None
    assert log.by_call_id("b") is not None
