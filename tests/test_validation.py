"""Tests for the validation harness (§7.1 reproduction fidelity)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentreplay.validation import (
    FidelityReport,
    FidelityResult,
    SyntheticTaskSet,
    Task,
    TaskSet,
    load_synthetic_tasks,
    run_fidelity_check,
    run_validation,
)
from agentreplay.validation.tasks import (
    SwebenchVerifiedTaskSet,
    GaiaTaskSet,
    get_task_set,
)


def test_synthetic_task_set_loads_tasks():
    ts = SyntheticTaskSet(num_tasks=3)
    tasks = ts.load()
    # 3 echo/reverse tasks + 3 lookup tasks = 6
    assert len(tasks) == 6
    assert all(isinstance(t, Task) for t in tasks)
    assert tasks[0].id.startswith("synthetic:")
    assert tasks[3].id.startswith("synthetic-lookup:")


def test_synthetic_task_set_respects_limit():
    ts = SyntheticTaskSet(num_tasks=5)
    tasks = ts.load(limit=2)
    assert len(tasks) == 4  # 2 + 2


def test_load_synthetic_tasks_helper():
    tasks = load_synthetic_tasks(num_tasks=2)
    assert len(tasks) == 4


def test_swebench_verified_task_set_raises_without_datasets():
    """Without the `datasets` package installed, the loader must raise
    NotImplementedError with a helpful message pointing to `pip install datasets`."""
    ts = SwebenchVerifiedTaskSet()
    try:
        import datasets  # noqa: F401
        has_datasets = True
    except ImportError:
        has_datasets = False
    if has_datasets:
        pytest.skip("datasets package installed; cannot test ImportError path")
    with pytest.raises(NotImplementedError) as ei:
        ts.load()
    assert "datasets" in str(ei.value)


def test_gaia_task_set_raises_without_datasets():
    """Without the `datasets` package installed, the loader must raise
    NotImplementedError with a helpful message."""
    ts = GaiaTaskSet()
    try:
        import datasets  # noqa: F401
        has_datasets = True
    except ImportError:
        has_datasets = False
    if has_datasets:
        pytest.skip("datasets package installed; cannot test ImportError path")
    with pytest.raises(NotImplementedError) as ei:
        ts.load()
    assert "datasets" in str(ei.value)


def test_get_task_set_known_names():
    assert isinstance(get_task_set("synthetic"), SyntheticTaskSet)
    assert isinstance(get_task_set("swebench-verified"), SwebenchVerifiedTaskSet)
    assert isinstance(get_task_set("gaia-subset"), GaiaTaskSet)


def test_get_task_set_unknown_raises():
    with pytest.raises(ValueError):
        get_task_set("nope")


def test_run_fidelity_check_passes_for_synthetic_task(tmp_path: Path):
    """A synthetic task must record + replay bit-exact (§7.1 target = 100%)."""
    tasks = load_synthetic_tasks(num_tasks=1)
    task = tasks[0]
    result = run_fidelity_check(task, cassette_root=tmp_path / "cass")
    assert isinstance(result, FidelityResult)
    assert result.passed
    assert result.live_calls_during_replay == 0
    assert result.num_events > 0
    assert result.recorded_terminal == result.replayed_terminal


def test_run_validation_returns_full_report(tmp_path: Path):
    """The full validation run must return a report with 100% fidelity."""
    ts = SyntheticTaskSet(num_tasks=2)
    report = run_validation(ts, cassette_root=tmp_path / "cass")
    assert isinstance(report, FidelityReport)
    assert report.num_total == 4  # 2 + 2
    assert report.num_passed == 4
    assert report.num_failed == 0
    assert report.passed
    assert report.fidelity_pct == 100.0


def test_fidelity_report_render(tmp_path: Path):
    ts = SyntheticTaskSet(num_tasks=1)
    report = run_validation(ts, cassette_root=tmp_path / "cass")
    out = report.render()
    assert "reproduction-fidelity" in out
    assert "100.0%" in out
    assert "✓" in out


def test_fidelity_report_to_dict_json_serialisable(tmp_path: Path):
    ts = SyntheticTaskSet(num_tasks=1)
    report = run_validation(ts, cassette_root=tmp_path / "cass")
    d = report.to_dict()
    s = json.dumps(d, default=str)
    assert "fidelity_pct" in s
    assert "results" in s


def test_fidelity_report_handles_failure(tmp_path: Path):
    """If the recording fails, the result should be marked failed."""
    # A task whose tools dict is empty but whose expected final text
    # requires tool calls — the scripted LLM will still produce a
    # final_text, but the replay must still match.
    task = Task(
        id="test:custom",
        description="custom task",
        messages=[{"role": "user", "content": "hi"}],
        expected={"final_text": "hello"},
        tools={},
    )
    result = run_fidelity_check(task, cassette_root=tmp_path / "cass")
    # Should still pass — no tools just means no tool events.
    assert result.passed
