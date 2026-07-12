"""Tests for the agentreplay CLI."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from click.testing import CliRunner

from agentreplay import Recorder
from agentreplay.cli import cli


class StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        return self.responses.pop(0)


def _record_cassette(path: Path, response_text: str = "hello") -> Path:
    stub = StubLLM([{"text": response_text, "usage": {}}])
    with Recorder.create(path, framework="raw", task_id="t1") as rec:
        client = rec.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "q"}], model="stub")
    return path


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_show(tmp_path: Path):
    c = _record_cassette(tmp_path / "cass")
    runner = CliRunner()
    result = runner.invoke(cli, ["show", str(c)])
    assert result.exit_code == 0
    assert "t1" in result.output  # task_id appears in metadata


def test_cli_show_events(tmp_path: Path):
    c = _record_cassette(tmp_path / "cass")
    runner = CliRunner()
    result = runner.invoke(cli, ["show", str(c), "--events"])
    assert result.exit_code == 0
    assert "seq" in result.output


def test_cli_list_filesystem_fallback(tmp_path: Path):
    _record_cassette(tmp_path / "a")
    _record_cassette(tmp_path / "b")
    runner = CliRunner()
    result = runner.invoke(cli, ["list", str(tmp_path)])
    assert result.exit_code == 0
    # Two cassette IDs should appear.
    lines = [l for l in result.output.strip().split("\n") if l]
    assert len(lines) == 2


def test_cli_replay_no_entry_prints_stats(tmp_path: Path):
    c = _record_cassette(tmp_path / "cass")
    runner = CliRunner()
    result = runner.invoke(cli, ["replay", str(c)])
    assert result.exit_code == 0
    assert "events:" in result.output


def test_cli_replay_with_divergence(tmp_path: Path):
    """`agentreplay replay` should exit 2 on divergence."""
    c = _record_cassette(tmp_path / "cass")
    # Write a tiny module that runs the agent with a different request.
    mod_path = tmp_path / "agent_mod_replay.py"
    mod_path.write_text(
        """
from agentreplay import Replayer

class Stub:
    def complete(self, **kw):
        raise AssertionError('should not be called')

def run_agent(replayer):
    client = replayer.wrap_custom_client(Stub())
    client.complete(messages=[{'role': 'user', 'content': 'DIFFERENT'}], model='stub')
"""
    )
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli, ["replay", str(c), "--agent-entry", "agent_mod_replay:run_agent"]
        )
        assert result.exit_code == 2
        assert "diverged" in result.output
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("agent_mod_replay", None)


def test_cli_diff_matching(tmp_path: Path):
    a = _record_cassette(tmp_path / "a", response_text="hello")
    b = _record_cassette(tmp_path / "b", response_text="hello")
    runner = CliRunner()
    result = runner.invoke(cli, ["diff", str(a), str(b)])
    assert result.exit_code == 0
    assert "bit-exact match" in result.output


def test_cli_diff_diverged(tmp_path: Path):
    a = _record_cassette(tmp_path / "a", response_text="hello")
    b = _record_cassette(tmp_path / "b", response_text="world")
    runner = CliRunner()
    result = runner.invoke(cli, ["diff", str(a), str(b)])
    assert result.exit_code == 0
    assert "first divergence" in result.output


def test_cli_mutate_creates_fork(tmp_path: Path):
    c = _record_cassette(tmp_path / "cass")
    out = tmp_path / "mutated"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "mutate",
            str(c),
            "--seq", "0",
            "--response", '{"text": "PATCHED", "usage": {}}',
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["mutated_seq"] == 0


def test_cli_mutate_requires_target(tmp_path: Path):
    c = _record_cassette(tmp_path / "cass")
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["mutate", str(c), "--response", '{"text": "x"}', "--out", str(tmp_path / "m")],
    )
    assert result.exit_code != 0


def test_cli_ci_passes(tmp_path: Path):
    c = _record_cassette(tmp_path / "a")
    mod_path = tmp_path / "agent_mod_ci.py"
    mod_path.write_text(
        """
from agentreplay import Replayer

class Stub:
    def complete(self, **kw):
        raise AssertionError('should not be called')

def run_agent(replayer):
    client = replayer.wrap_custom_client(Stub())
    client.complete(messages=[{'role': 'user', 'content': 'q'}], model='stub')
"""
    )
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        runner = CliRunner()
        result = runner.invoke(
            cli, ["ci", str(tmp_path), "--agent-entry", "agent_mod_ci:run_agent"]
        )
        assert result.exit_code == 0
        assert "1/1 passed" in result.output
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("agent_mod_ci", None)
