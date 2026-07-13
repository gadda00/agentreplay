"""Tests for streaming response support (RecordingStream / ReplayStream)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentreplay import Cassette, Recorder, Replayer
from agentreplay.constants import Mode
from agentreplay.interceptors.streaming import (
    RecordingStream,
    ReplayStream,
    is_streamed_response,
    make_streamed_response,
    _serialize_chunk,
)


class _StubStreamingLLM:
    """Stub LLM that returns an iterator of chunk dicts when stream=True."""

    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = list(chunks)
        self.live_calls = 0

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Any:
        if params.get("stream"):
            self.live_calls += 1
            return iter(list(self.chunks))
        return {"text": "non-streamed", "usage": {}}


def test_recording_stream_captures_chunks():
    """RecordingStream should capture chunks as they're consumed."""
    real_chunks = [{"text": "hello"}, {"text": " world"}, {"text": "!"}]
    captured: list = []

    stream = RecordingStream(iter(real_chunks), on_complete=captured.append)
    consumed = list(stream)

    assert consumed == real_chunks
    assert len(captured) == 1
    assert captured[0] == real_chunks
    assert stream.exhausted
    assert stream.chunks == real_chunks


def test_replay_stream_yields_recorded_chunks():
    """ReplayStream should yield chunks from a list."""
    chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    stream = ReplayStream(chunks)
    result = list(stream)
    assert result == chunks


def test_replay_stream_next_and_stopiteration():
    """ReplayStream should support __next__ and raise StopIteration."""
    stream = ReplayStream([{"x": 1}, {"x": 2}])
    assert next(stream)["x"] == 1
    assert next(stream)["x"] == 2
    with pytest.raises(StopIteration):
        next(stream)


def test_is_streamed_response():
    """is_streamed_response should detect streamed response payloads."""
    assert is_streamed_response({"chunks": [], "streamed": True}) is True
    assert is_streamed_response({"text": "hello"}) is False
    assert is_streamed_response({}) is False


def test_make_streamed_response():
    """make_streamed_response should create the correct payload shape."""
    chunks = [{"a": 1}, {"b": 2}]
    result = make_streamed_response(chunks)
    assert result["streamed"] is True
    assert result["chunks"] == chunks


def test_serialize_chunk_dict():
    """_serialize_chunk should pass dicts through."""
    assert _serialize_chunk({"a": 1}) == {"a": 1}


def test_serialize_chunk_string():
    """_serialize_chunk should pass strings through."""
    assert _serialize_chunk("hello") == "hello"


def test_serialize_chunk_pydantic_v2():
    """_serialize_chunk should handle Pydantic v2 models via model_dump()."""

    class FakePydanticModel:
        def model_dump(self) -> dict:
            return {"text": "chunk", "index": 0}

    chunk = FakePydanticModel()
    assert _serialize_chunk(chunk) == {"text": "chunk", "index": 0}


def test_streaming_record_then_replay(tmp_path: Path):
    """End-to-end: record a streaming call, then replay it bit-exact."""
    cassette = tmp_path / "cass"
    chunks = [{"text": "hello"}, {"text": " world"}]

    stub = _StubStreamingLLM(chunks)
    with Recorder.create(cassette, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        stream = client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="stub",
            stream=True,
        )
        consumed = list(stream)

    assert consumed == chunks
    assert stub.live_calls == 1

    # Verify the cassette recorded a streamed response
    c = Cassette.open(cassette, readonly=True)
    events = list(c.events)
    assert len(events) == 1
    assert events[0].metadata.get("streamed") is True
    assert events[0].metadata.get("num_chunks") == 2

    # Replay
    fresh = _StubStreamingLLM(chunks)
    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(fresh)
        stream = client.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="stub",
            stream=True,
        )
        replayed = list(stream)

    assert replayed == chunks
    assert fresh.live_calls == 0  # zero model calls during replay


def test_streaming_replay_returns_replay_stream(tmp_path: Path):
    """Replay of a streamed response should return a ReplayStream."""
    cassette = tmp_path / "cass"
    chunks = [{"text": "x"}]

    stub = _StubStreamingLLM(chunks)
    with Recorder.create(cassette, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        list(client.complete(messages=[{"role": "user", "content": "q"}], model="stub", stream=True))

    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(_StubStreamingLLM([]))
        result = client.complete(messages=[{"role": "user", "content": "q"}], model="stub", stream=True)
        assert isinstance(result, ReplayStream)
        assert list(result) == chunks
