"""Tests for cassette export/import (ZIP) and the new CLI commands."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from click.testing import CliRunner

from agentreplay import Cassette, Recorder
from agentreplay.cli import cli


class _StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        return self.responses.pop(0)


def _make_cassette(path: Path) -> Path:
    """Create a tiny cassette for testing."""
    stub = _StubLLM([{"text": "hello", "usage": {}}])
    with Recorder.create(path, framework="raw", agent_name="test") as rec:
        client = rec.wrap_custom_client(stub)
        client.complete(messages=[{"role": "user", "content": "hi"}], model="stub")
    return path


# ---------------------------------------------------------------------- #
# Cassette export/import
# ---------------------------------------------------------------------- #
def test_cassette_export_zip(tmp_path: Path):
    """export_zip should create a ZIP containing cassette.json, events.jsonl, blobs/."""
    cassette = _make_cassette(tmp_path / "cass")
    zip_path = tmp_path / "export.zip"

    c = Cassette.open(cassette, readonly=True)
    result_path = c.export_zip(zip_path)

    assert result_path == zip_path
    assert zip_path.exists()
    assert zip_path.stat().st_size > 0

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "cassette.json" in names
        assert "events.jsonl" in names
        # At least one blob file
        blob_files = [n for n in names if n.startswith("blobs/")]
        assert len(blob_files) > 0


def test_cassette_import_zip(tmp_path: Path):
    """import_zip should reconstruct the cassette from a ZIP."""
    cassette = _make_cassette(tmp_path / "cass")
    zip_path = tmp_path / "export.zip"
    target = tmp_path / "imported"

    c = Cassette.open(cassette, readonly=True)
    c.export_zip(zip_path)

    imported = Cassette.import_zip(zip_path, target)
    assert imported.meta.agent_name == "test"
    assert len(imported.events) == 1
    # The imported cassette should have the same blobs
    assert len(imported.blobs) == len(c.blobs)


def test_cassette_export_import_roundtrip(tmp_path: Path):
    """Export → import should produce an equivalent cassette."""
    cassette = _make_cassette(tmp_path / "cass")
    zip_path = tmp_path / "export.zip"
    target = tmp_path / "imported"

    original = Cassette.open(cassette, readonly=True)
    original.export_zip(zip_path)
    imported = Cassette.import_zip(zip_path, target)

    # Same metadata (except id and created_at which may differ)
    assert imported.meta.framework == original.meta.framework
    assert imported.meta.agent_name == original.meta.agent_name
    assert imported.meta.num_events == original.meta.num_events

    # Same events
    orig_events = list(original.events)
    imported_events = list(imported.events)
    assert len(orig_events) == len(imported_events)
    assert orig_events[0].call_id == imported_events[0].call_id

    # Same blob count
    assert len(imported.blobs) == len(original.blobs)


def test_import_zip_rejects_nonempty_target(tmp_path: Path):
    """import_zip should refuse to overwrite a non-empty directory."""
    cassette = _make_cassette(tmp_path / "cass")
    zip_path = tmp_path / "export.zip"
    target = tmp_path / "target"
    target.mkdir()
    (target / "existing.txt").write_text("x")

    c = Cassette.open(cassette, readonly=True)
    c.export_zip(zip_path)

    from agentreplay.errors import CassetteError
    with __import__("pytest").raises(CassetteError, match="not empty"):
        Cassette.import_zip(zip_path, target)


# ---------------------------------------------------------------------- #
# CLI: info
# ---------------------------------------------------------------------- #
def test_cli_info():
    runner = CliRunner()
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "AgentReplay" in result.output
    assert "Python:" in result.output
    assert "Optional dependencies:" in result.output


# ---------------------------------------------------------------------- #
# CLI: export / import
# ---------------------------------------------------------------------- #
def test_cli_export(tmp_path: Path):
    cassette = _make_cassette(tmp_path / "cass")
    zip_path = str(tmp_path / "out.zip")
    runner = CliRunner()
    result = runner.invoke(cli, ["export", str(cassette), zip_path])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["zip"] == zip_path
    assert payload["size_bytes"] > 0


def test_cli_import(tmp_path: Path):
    cassette = _make_cassette(tmp_path / "cass")
    zip_path = str(tmp_path / "out.zip")
    target = str(tmp_path / "imported")
    runner = CliRunner()
    runner.invoke(cli, ["export", str(cassette), zip_path])
    result = runner.invoke(cli, ["import", zip_path, target])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "ok"
    assert payload["num_events"] == 1


# ---------------------------------------------------------------------- #
# CLI: clean (dry-run by default)
# ---------------------------------------------------------------------- #
def test_cli_clean_dry_run(tmp_path: Path):
    """clean should default to dry-run and not actually delete anything."""
    cassette = _make_cassette(tmp_path / "cass")
    runner = CliRunner()
    result = runner.invoke(cli, ["clean", str(tmp_path), "--older-than", "1d"])
    assert result.exit_code == 0
    # The cassette should still exist (dry-run)
    assert cassette.exists()


def test_cli_clean_no_dry_run(tmp_path: Path):
    """clean --no-dry-run should actually remove cassettes."""
    _make_cassette(tmp_path / "cass")
    runner = CliRunner()
    result = runner.invoke(cli, ["clean", str(tmp_path), "--older-than", "1d", "--no-dry-run"])
    assert result.exit_code == 0
    assert "removed" in result.output.lower()


def test_cli_clean_keep_outcome(tmp_path: Path):
    """clean --keep-outcome should preserve cassettes with that outcome."""
    cassette = _make_cassette(tmp_path / "cass")
    # Mark the cassette as 'fail'
    c = Cassette.open(cassette, readonly=False)
    c.meta.outcome = "fail"
    c.save()

    runner = CliRunner()
    result = runner.invoke(cli, ["clean", str(tmp_path), "--older-than", "1d", "--keep-outcome", "fail"])
    assert result.exit_code == 0
    assert cassette.exists()  # Should be kept because outcome=fail


# ---------------------------------------------------------------------- #
# CLI: --verbose
# ---------------------------------------------------------------------- #
def test_cli_verbose_flag():
    """--verbose should not crash and should enable debug logging."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--verbose", "info"])
    assert result.exit_code == 0
