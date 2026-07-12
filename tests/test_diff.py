"""Tests for the structural diff engine."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from agentreplay import Cassette, Recorder
from agentreplay.constants import CallType
from agentreplay.diff import Diff, diff_payloads, diff_structural, render_diff


class StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        return self.responses.pop(0)


def _make_cassette(tmp_path: Path, name: str, response_text: str) -> Path:
    p = tmp_path / name
    stub = StubLLM([{"text": response_text, "usage": {}}])
    with Recorder.create(p, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "q"}], model="stub")
    return p


def test_diff_payloads_identical_payloads_have_no_diffs():
    a = {"x": 1, "y": [1, 2]}
    b = {"y": [1, 2], "x": 1}
    assert diff_payloads(a, b) == []


def test_diff_payloads_reports_value_change():
    a = {"x": 1, "y": [1, 2]}
    b = {"x": 2, "y": [1, 2]}
    diffs = diff_payloads(a, b)
    paths = [d.path for d in diffs]
    assert "x" in paths


def test_diff_structural_identical_cassettes_match(tmp_path: Path):
    a = _make_cassette(tmp_path, "a", "hello")
    b = _make_cassette(tmp_path, "b", "hello")
    ca = Cassette.open(a, readonly=True)
    cb = Cassette.open(b, readonly=True)
    diff = diff_structural(ca, cb)
    assert isinstance(diff, Diff)
    assert not diff.has_divergence
    assert diff.first_divergence is None


def test_diff_structural_detects_diverged_response(tmp_path: Path):
    a = _make_cassette(tmp_path, "a", "hello")
    b = _make_cassette(tmp_path, "b", "world")
    ca = Cassette.open(a, readonly=True)
    cb = Cassette.open(b, readonly=True)
    diff = diff_structural(ca, cb)
    assert diff.has_divergence
    assert diff.first_divergence is not None
    assert diff.first_divergence.call_type == "llm"


def test_render_diff_handles_matching_case(tmp_path: Path):
    a = _make_cassette(tmp_path, "a", "hello")
    b = _make_cassette(tmp_path, "b", "hello")
    ca = Cassette.open(a, readonly=True)
    cb = Cassette.open(b, readonly=True)
    diff = diff_structural(ca, cb)
    out = render_diff(diff)
    assert "bit-exact match" in out


def test_render_diff_handles_divergence(tmp_path: Path):
    a = _make_cassette(tmp_path, "a", "hello")
    b = _make_cassette(tmp_path, "b", "world")
    ca = Cassette.open(a, readonly=True)
    cb = Cassette.open(b, readonly=True)
    diff = diff_structural(ca, cb)
    out = render_diff(diff)
    assert "first divergence" in out


def test_diff_summary_counts(tmp_path: Path):
    a = _make_cassette(tmp_path, "a", "hello")
    b = _make_cassette(tmp_path, "b", "world")
    ca = Cassette.open(a, readonly=True)
    cb = Cassette.open(b, readonly=True)
    diff = diff_structural(ca, cb)
    s = diff.summary()
    assert s["total_steps"] == 1
    assert s["has_divergence"] is True


def test_diff_handles_different_lengths(tmp_path: Path):
    a = _make_cassette(tmp_path, "a", "hello")
    # Cassette b has TWO LLM calls
    p_b = tmp_path / "b"
    stub = StubLLM([
        {"text": "first", "usage": {}},
        {"text": "second", "usage": {}},
    ])
    with Recorder.create(p_b, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "q1"}], model="stub")
        client.complete(messages=[{"role": "user", "content": "q2"}], model="stub")
    ca = Cassette.open(a, readonly=True)
    cb = Cassette.open(p_b, readonly=True)
    diff = diff_structural(ca, cb)
    s = diff.summary()
    assert s["total_steps"] == 2  # max(1, 2)
    assert s["has_divergence"] is True
