"""Shared pytest fixtures for the agentreplay test suite."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentreplay import Cassette, Recorder, Replayer
from agentreplay.constants import Mode


@pytest.fixture
def cassette_dir(tmp_path: Path) -> Path:
    """Fresh empty directory for a single cassette."""
    d = tmp_path / "cassette"
    d.mkdir()
    return d


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Fresh empty directory for a cassette corpus."""
    d = tmp_path / "corpus"
    d.mkdir()
    return d


class StubLLMClient:
    """Deterministic LLM client for tests — no network, no SDK.

    Returns a scripted response for each call so we can exercise the
    full record/replay cycle without any API key.
    """

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        self.calls.append({"messages": messages, "tools": tools, "params": params})
        if not self.responses:
            raise RuntimeError("StubLLMClient exhausted")
        return self.responses.pop(0)


@pytest.fixture
def stub_llm() -> StubLLMClient:
    return StubLLMClient(
        responses=[
            {"text": "Hello from the model.", "tool_calls": [], "usage": {"total_tokens": 10}},
            {"text": "Second reply.", "tool_calls": [], "usage": {"total_tokens": 8}},
        ]
    )


@pytest.fixture
def stub_tool():
    def search(query: str) -> str:
        return f"result for {query}"
    return search


@pytest.fixture
def recorded_cassette(cassette_dir: Path, stub_llm: StubLLMClient, stub_tool):
    """Record a tiny cassette and return (path, recorded_events)."""
    path = cassette_dir / "run-001"
    with Recorder.create(path, framework="raw", agent_name="test-agent") as rec:
        client = rec.wrap_custom_client(stub_llm)
        tool = rec.wrap_tool(stub_tool, name="search")

        # Simulate an agent loop
        client.complete(messages=[{"role": "user", "content": "hi"}], model="stub")
        tool(query="weather")
        client.complete(messages=[{"role": "user", "content": "again"}], model="stub")

    return path
