"""QA test: BlobStore preserves 'id' and other non-deterministic keys in
stored values. Found by adversarial QA testing — the canonicalize() function
was stripping 'id' from ALL nesting levels, including tool_calls[].id which
is legitimate agent data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from agentreplay import Cassette, Recorder, Replayer, Mode


class _StubLLM:
    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        return self.response


def test_blob_store_preserves_id_in_tool_calls(tmp_path: Path):
    """The BlobStore must store the ORIGINAL value, not the canonicalized
    version. The canonicalize() function strips 'id' from all dicts (for
    hashing), but 'id' in tool_calls is legitimate agent data that must
    survive storage and replay."""
    cassette = tmp_path / "cass"
    response_with_tool_calls = {
        "text": "I'll search for that.",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "weather"}',
                },
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        "finish_reason": "tool_calls",
    }

    # Record
    stub = _StubLLM(response_with_tool_calls)
    with Recorder.create(cassette, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        r = client.complete(
            messages=[{"role": "user", "content": "search"}],
            model="gpt-4",
        )
    assert r == response_with_tool_calls

    # Verify the stored blob preserves 'id'
    c = Cassette.open(cassette, readonly=True)
    records = c.records()
    assert len(records) == 1
    stored_response = records[0].response
    assert "tool_calls" in stored_response
    assert stored_response["tool_calls"][0]["id"] == "call_abc123", \
        f"id was stripped! got keys: {list(stored_response['tool_calls'][0].keys())}"

    # Replay — must return the EXACT original response including 'id'
    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(_StubLLM({}))  # empty stub — should never be called
        r = client.complete(
            messages=[{"role": "user", "content": "search"}],
            model="gpt-4",
        )
    assert r == response_with_tool_calls, \
        f"replayed response differs from original — missing fields?"


def test_blob_store_preserves_created_field(tmp_path: Path):
    """The 'created' field (OpenAI epoch timestamp) must also be preserved
    in stored responses, even though it's in _NON_DETERMINISTIC_KEYS."""
    cassette = tmp_path / "cass"
    response_with_created = {
        "id": "chatcmpl-abc123",
        "created": 1720780800,
        "text": "hello",
        "usage": {"total_tokens": 5},
    }

    stub = _StubLLM(response_with_created)
    with Recorder.create(cassette, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "hi"}], model="gpt-4")

    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(_StubLLM({}))
        r = client.complete(messages=[{"role": "user", "content": "hi"}], model="gpt-4")

    assert r == response_with_created, \
        f"created/id fields were stripped from stored response! got: {r}"


def test_blob_store_dedup_still_works(tmp_path: Path):
    """Deduplication must still work — two values that are canonically
    equal (same content, different non-deterministic keys) should produce
    the same digest and share a blob."""
    from agentreplay.storage.blob import BlobStore

    bs = BlobStore(tmp_path / "blobs")

    # Two responses with same content but different 'id' (non-deterministic)
    v1 = {"text": "hello", "id": "abc", "usage": {}}
    v2 = {"text": "hello", "id": "xyz", "usage": {}}

    d1 = bs.put(v1)
    d2 = bs.put(v2)

    # Same digest (dedup works — canonical form is identical)
    assert d1 == d2, f"dedup broken: {d1} != {d2}"

    # But the stored value preserves the FIRST value's 'id'
    retrieved = bs.get(d1)
    assert "id" in retrieved, "id was stripped from stored value"
    assert retrieved["text"] == "hello"
