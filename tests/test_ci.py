"""Tests for the CI regression runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentreplay import Cassette, Recorder, Replayer
from agentreplay.ci import RegressionReport, discover_cassettes, run_corpus


class StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        return self.responses.pop(0)


def _record_cassette(path: Path, responses: List[Dict[str, Any]]) -> None:
    stub = StubLLM(responses)
    with Recorder.create(path, framework="raw") as rec:
        client = rec.wrap_custom_client(stub)
        for i, r in enumerate(responses):
            client.complete(messages=[{"role": "user", "content": f"q{i}"}], model="stub")


def test_discover_cassettes_finds_cassette_dirs(tmp_path: Path):
    _record_cassette(tmp_path / "a", [{"text": "x", "usage": {}}])
    _record_cassette(tmp_path / "b", [{"text": "y", "usage": {}}])
    # Non-cassette dir should be ignored.
    (tmp_path / "not-a-cassette").mkdir()
    found = discover_cassettes(tmp_path)
    assert len(found) == 2
    assert all(p.is_dir() for p in found)


def test_run_corpus_passes_when_replay_matches(tmp_path: Path):
    _record_cassette(tmp_path / "a", [{"text": "x", "usage": {}}])
    _record_cassette(tmp_path / "b", [{"text": "y", "usage": {}}])

    def agent_run(rep: Replayer) -> Any:
        client = rep.wrap_custom_client(StubLLM([]))  # live must not be called
        client.complete(messages=[{"role": "user", "content": "q0"}], model="stub")

    report = run_corpus(tmp_path, agent_run)
    assert isinstance(report, RegressionReport)
    assert report.passed
    assert report.num_passed == 2
    assert report.num_failed == 0


def test_run_corpus_fails_on_divergence(tmp_path: Path):
    _record_cassette(tmp_path / "a", [{"text": "x", "usage": {}}])

    def agent_run(rep: Replayer) -> Any:
        client = rep.wrap_custom_client(StubLLM([]))
        # Different request than recorded → divergence.
        client.complete(messages=[{"role": "user", "content": "DIFFERENT"}], model="stub")

    report = run_corpus(tmp_path, agent_run)
    assert not report.passed
    assert report.num_failed == 1
    r = report.results[0]
    assert not r.passed
    assert r.error is not None
    assert r.diff is not None
    assert r.diff["call_type"] == "llm"


def test_run_corpus_render_includes_marks(tmp_path: Path):
    _record_cassette(tmp_path / "a", [{"text": "x", "usage": {}}])

    def agent_run(rep: Replayer) -> Any:
        client = rep.wrap_custom_client(StubLLM([]))
        client.complete(messages=[{"role": "user", "content": "q0"}], model="stub")

    report = run_corpus(tmp_path, agent_run)
    out = report.render()
    assert "✓" in out
    assert "1/1 passed" in out


def test_run_corpus_outcome_filter(tmp_path: Path):
    """Only cassettes matching --outcome are replayed."""
    p_pass = tmp_path / "pass-cass"
    p_fail = tmp_path / "fail-cass"
    # Record two cassettes. Recorder auto-sets outcome on close, so we
    # patch the "fail" cassette's meta after the fact.
    with Recorder.create(p_pass, framework="raw") as rec:
        client = rec.wrap_custom_client(StubLLM([{"text": "x", "usage": {}}]))
        client.complete(messages=[{"role": "user", "content": "q0"}], model="stub")
    with Recorder.create(p_fail, framework="raw") as rec:
        client = rec.wrap_custom_client(StubLLM([{"text": "y", "usage": {}}]))
        client.complete(messages=[{"role": "user", "content": "q0"}], model="stub")
        rec.cassette.meta.outcome = "fail"
        rec.cassette.save()

    def agent_run(rep: Replayer) -> Any:
        client = rep.wrap_custom_client(StubLLM([]))
        client.complete(messages=[{"role": "user", "content": "q0"}], model="stub")

    # Filter to only "fail" cassettes — but the recorded request was the
    # same so it should still pass.
    report = run_corpus(tmp_path, agent_run, outcome_filter="fail")
    assert len(report.results) == 1
    assert report.results[0].cassette_id.startswith("cass-")


def test_run_corpus_stop_on_first_failure(tmp_path: Path):
    _record_cassette(tmp_path / "a", [{"text": "x", "usage": {}}])
    _record_cassette(tmp_path / "b", [{"text": "y", "usage": {}}])

    def agent_run(rep: Replayer) -> Any:
        client = rep.wrap_custom_client(StubLLM([]))
        # Always diverge
        client.complete(messages=[{"role": "user", "content": "WRONG"}], model="stub")

    report = run_corpus(tmp_path, agent_run, stop_on_first_failure=True)
    assert report.num_failed == 1  # only the first failure recorded
