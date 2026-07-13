"""Streaming response support for LLM interceptors.

When an agent uses ``stream=True`` (OpenAI/Anthropic), the SDK returns
an iterator of chunks instead of a single response. This module provides
a recording wrapper that captures all chunks and a replay wrapper that
serves them back from the cassette.

The key insight: a streamed response is just a list of chunks. We record
them as a single event whose response payload is ``{"chunks": [...],
"streamed": True}``. On replay, we yield the chunks one by one from the
recorded list.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional


class RecordingStream:
    """Wraps a streaming response iterator, capturing chunks as they're consumed.

    Usage in RecordingClient::

        if params.get("stream"):
            stream = self._invoke_real(messages=messages, tools=tools, **params)
            return RecordingStream(stream, on_complete=callback)

    The ``on_complete`` callback is called with the list of captured
    chunks when the stream is exhausted, so the RecordingClient can
    write them to the cassette as a single event.

    The callback is called via ``try/finally`` so it fires even if the
    consumer breaks out of the iteration early or an exception is raised
    — preventing silent data loss.
    """

    def __init__(
        self,
        real_stream: Iterator[Any],
        *,
        on_complete: Optional[Any] = None,
    ) -> None:
        self._real_stream = real_stream
        self._on_complete = on_complete
        self._chunks: List[Any] = []
        self._exhausted = False
        self._callback_fired = False

    def __iter__(self) -> Iterator[Any]:
        try:
            for chunk in self._real_stream:
                serializable = _serialize_chunk(chunk)
                self._chunks.append(serializable)
                yield chunk
            self._exhausted = True
        finally:
            self._fire_callback()

    def __aiter__(self):
        """Async iteration support — delegates to the real stream's __aiter__."""
        return self._async_iter()

    async def _async_iter(self):
        try:
            if hasattr(self._real_stream, "__aiter__"):
                async for chunk in self._real_stream:
                    serializable = _serialize_chunk(chunk)
                    self._chunks.append(serializable)
                    yield chunk
            else:
                # Sync stream in async context — iterate synchronously
                for chunk in self._real_stream:
                    serializable = _serialize_chunk(chunk)
                    self._chunks.append(serializable)
                    yield chunk
            self._exhausted = True
        finally:
            self._fire_callback()

    def _fire_callback(self) -> None:
        """Fire the on_complete callback exactly once, even on early exit."""
        if not self._callback_fired and self._on_complete is not None:
            self._callback_fired = True
            self._on_complete(self._chunks)

    @property
    def chunks(self) -> List[Any]:
        """Return captured chunks (only available after the stream is exhausted)."""
        return self._chunks

    @property
    def exhausted(self) -> bool:
        return self._exhausted


class ReplayStream:
    """Replays a recorded stream of chunks from the cassette.

    On replay, the recorded response payload is ``{"chunks": [...],
    "streamed": True}``. This wrapper yields each chunk back to the
    agent code, optionally re-hydrating them into SDK objects.

    By default, chunks are yielded as plain dicts (which is what most
    agent code actually consumes). If the agent code expects SDK objects
    (e.g. ``openai.ChatCompletionChunk``), a ``rehydrate`` callable can
    be provided to convert dicts back to SDK objects.
    """

    def __init__(
        self,
        chunks: List[Any],
        *,
        rehydrate: Optional[Any] = None,
    ) -> None:
        self._chunks = list(chunks)
        self._rehydrate = rehydrate
        self._index = 0

    def __iter__(self) -> Iterator[Any]:
        for chunk in self._chunks:
            if self._rehydrate is not None:
                yield self._rehydrate(chunk)
            else:
                yield chunk

    def __next__(self) -> Any:
        if self._index >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._index]
        self._index += 1
        if self._rehydrate is not None:
            return self._rehydrate(chunk)
        return chunk

    async def __aiter__(self):
        """Async iteration support — yields chunks from the recorded list."""
        for chunk in self._chunks:
            if self._rehydrate is not None:
                yield self._rehydrate(chunk)
            else:
                yield chunk

    def close(self) -> None:
        """No-op — compatibility with httpx/requests stream close()."""
        pass


def _serialize_chunk(chunk: Any) -> Any:
    """Convert a chunk to a JSON-serialisable form.

    Handles:
    - Pydantic models (OpenAI/Anthropic SDK chunks) → ``.model_dump()``
    - Dicts → pass through
    - Strings → pass through
    - Other → ``repr()``
    """
    if isinstance(chunk, dict):
        return chunk
    if isinstance(chunk, str):
        return chunk
    # Pydantic v2
    if hasattr(chunk, "model_dump"):
        try:
            return chunk.model_dump()
        except Exception:
            pass
    # Pydantic v1
    if hasattr(chunk, "dict"):
        try:
            return chunk.dict()
        except Exception:
            pass
    # Objects with __dict__
    if hasattr(chunk, "__dict__"):
        try:
            return json.loads(json.dumps(chunk.__dict__, default=str))
        except Exception:
            pass
    return repr(chunk)


def is_streamed_response(response: Any) -> bool:
    """Check if a recorded response payload represents a streamed response."""
    return (
        isinstance(response, dict)
        and response.get("streamed") is True
        and "chunks" in response
    )


def make_streamed_response(chunks: List[Any]) -> Dict[str, Any]:
    """Create a response payload for a streamed response."""
    return {"chunks": list(chunks), "streamed": True}
