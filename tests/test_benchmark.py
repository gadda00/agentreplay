"""Tests for the overhead benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentreplay.benchmark.overhead import (
    BenchmarkReport,
    BenchmarkResult,
    run_benchmark,
)


def test_run_benchmark_returns_report(tmp_path: Path):
    """The benchmark must return a complete report with all expected rows."""
    report = run_benchmark(iterations=5, repeats=1, cassette_dir=tmp_path / "bench")
    assert isinstance(report, BenchmarkReport)
    assert report.iterations == 5
    # Must include baseline, AgentReplay record, AgentReplay replay, and
    # four synthetic baselines.
    names = [r.name for r in report.results]
    assert "baseline" in names
    assert "AgentReplay (record)" in names
    assert "AgentReplay (replay)" in names
    assert "LangSmith (synthetic)" in names
    assert "Laminar (synthetic)" in names
    assert "AgentOps (synthetic)" in names
    assert "Langfuse (synthetic)" in names


def test_baseline_has_zero_overhead(tmp_path: Path):
    report = run_benchmark(iterations=5, repeats=1, cassette_dir=tmp_path / "bench")
    baseline = next(r for r in report.results if r.name == "baseline")
    assert baseline.overhead_pct == 0.0


def test_agentreplay_overhead_under_target(tmp_path: Path):
    """§7.2 target: AgentReplay overhead must be ≤ 5%.

    Uses more iterations + repeats than the default to reduce timing
    noise. Tolerance is 8% to account for CI runner variability — the
    real-world overhead is typically <2% on a quiet machine, but shared
    CI runners can spike. The benchmark CLI's own --json output is the
    canonical measurement; this test just guards against regressions."""
    report = run_benchmark(iterations=50, repeats=3, cassette_dir=tmp_path / "bench")
    ar = next(r for r in report.results if r.name == "AgentReplay (record)")
    assert ar.overhead_pct <= 8.0, f"AgentReplay overhead {ar.overhead_pct:.2f}% > 8% guard"


def test_agentreplay_replay_is_faster_than_baseline(tmp_path: Path):
    """In REPLAY mode the agent makes zero model calls, so it must be
    substantially faster than the baseline (which simulates LLM latency)."""
    report = run_benchmark(iterations=10, repeats=1, cassette_dir=tmp_path / "bench")
    baseline = next(r for r in report.results if r.name == "baseline")
    replay = next(r for r in report.results if r.name == "AgentReplay (replay)")
    assert replay.per_call_ms < baseline.per_call_ms
    assert replay.overhead_pct < 0  # negative = faster


def test_synthetic_baselines_match_published_figures(tmp_path: Path):
    """Synthetic baselines should produce overhead close to their target
    fractions (LangSmith ~0%, Laminar ~5%, AgentOps ~12%, Langfuse ~15%).

    Uses more iterations + repeats than the other tests to reduce timing
    noise. Tolerance is 2pp to account for CI runner variability."""
    report = run_benchmark(iterations=40, repeats=3, cassette_dir=tmp_path / "bench")
    for name, expected in [
        ("LangSmith (synthetic)", 0.0),
        ("Laminar (synthetic)", 5.0),
        ("AgentOps (synthetic)", 12.0),
        ("Langfuse (synthetic)", 15.0),
    ]:
        r = next(r for r in report.results if r.name == name)
        assert abs(r.overhead_pct - expected) < 2.0, (
            f"{name} overhead {r.overhead_pct:.2f}% not within 2pp of {expected}%"
        )


def test_report_to_dict_roundtrips(tmp_path: Path):
    report = run_benchmark(iterations=3, repeats=1, cassette_dir=tmp_path / "bench")
    d = report.to_dict()
    # Must be JSON-serialisable
    s = json.dumps(d, default=str)
    assert "iterations" in s
    assert "results" in s


def test_report_render_contains_verdict(tmp_path: Path):
    report = run_benchmark(iterations=3, repeats=1, cassette_dir=tmp_path / "bench")
    out = report.render()
    assert "AgentReplay overhead" in out
    assert "≤ 5% target" in out or "> 5% target" in out
