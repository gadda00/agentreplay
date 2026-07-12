"""Overhead benchmark.

Measures AgentReplay's recording-layer latency overhead against an
uninstrumented baseline, using the same methodology as the independent
2026 four-platform benchmark referenced in §2.5 / §7.2 of the product
proposal: percentage latency increase versus an uninstrumented run,
on an identical repeated workload.

The benchmark also runs synthetic baselines that mimic the published
overhead figures for LangSmith (~0%), Laminar (~5%), AgentOps (~12%),
and Langfuse (~15%) — these are *not* measurements of those tools
(which would require running them with their full SDK + backend), but
simulated baselines that let the report put AgentReplay's number in
context against the published figures.

Usage::

    python -m agentreplay.benchmark.overhead --iterations 200 --report report.json
    agentreplay benchmark-overhead --iterations 200

Target (per §7.2): comparable to or better than Laminar's ~5% figure,
since AgentReplay's interceptors sit at the same client-wrapping layer
rather than the more decoupled callback-based integration pattern
associated with the higher-overhead tools in that benchmark.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentreplay import Cassette, Recorder
from agentreplay.constants import Mode


# ---------------------------------------------------------------------- #
# Workload
# ---------------------------------------------------------------------- #
# Simulated LLM latency. Real LLM calls take 500-5000ms; we use a
# conservative 50ms lower bound so the benchmark runs in reasonable
# time but the file I/O overhead doesn't dominate. The §7.2 methodology
# measures "percentage latency increase versus an uninstrumented run"
# — the percentage is only meaningful when the baseline call has
# realistic latency, otherwise I/O dwarfs everything.
SIMULATED_LLM_LATENCY_S = 0.050


def _stub_response(i: int) -> Dict[str, Any]:
    return {
        "text": f"response {i}",
        "tool_calls": [],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        "finish_reason": "stop",
    }


class _StubLLM:
    """Deterministic stub LLM that simulates realistic call latency.

    The benchmark measures interceptor overhead *as a fraction of total
    call latency*. A real LLM call takes 500-5000ms; we simulate 50ms
    (a conservative lower bound) so the file I/O overhead doesn't
    dominate the percentage.
    """

    def __init__(self, n: int, *, latency_s: float = SIMULATED_LLM_LATENCY_S) -> None:
        self.n = n
        self.i = 0
        self.latency_s = latency_s

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        # Simulate LLM latency
        end = time.perf_counter() + self.latency_s
        while time.perf_counter() < end:
            pass
        r = _stub_response(self.i)
        self.i += 1
        return r


def _run_workload(client: Any, n: int) -> int:
    """Make n LLM calls through `client`. Returns n for assertion."""
    for _ in range(n):
        client.complete(
            messages=[{"role": "user", "content": "ping"}],
            model="stub",
        )
    return n


# ---------------------------------------------------------------------- #
# Synthetic baselines (mimic published 2026 figures)
# ---------------------------------------------------------------------- #
class _SyntheticOverheadClient:
    """A client that simulates a real LLM call plus a fixed percentage of
    instrumentation overhead.

    Used to simulate the published overhead figures for LangSmith (~0%),
    Laminar (~5%), AgentOps (~12%), Langfuse (~15%). The point is *not*
    to measure those tools — it is to put AgentReplay's number in
    context against the published figures using the same workload.

    The baseline latency is the same :data:`SIMULATED_LLM_LATENCY_S`
    used by :class:`_StubLLM`; the overhead is applied on top, matching
    the §7.2 methodology of "percentage latency increase versus an
    uninstrumented run".
    """

    def __init__(self, overhead_fraction: float, *, latency_s: float = SIMULATED_LLM_LATENCY_S) -> None:
        self.overhead_fraction = overhead_fraction
        self.latency_s = latency_s
        self.i = 0

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        # Simulate the baseline LLM latency...
        end = time.perf_counter() + self.latency_s
        while time.perf_counter() < end:
            pass
        # ...plus the platform's instrumentation overhead.
        overhead_s = self.overhead_fraction * self.latency_s
        if overhead_s > 0:
            end = time.perf_counter() + overhead_s
            while time.perf_counter() < end:
                pass
        r = _stub_response(self.i)
        self.i += 1
        return r


# ---------------------------------------------------------------------- #
# Benchmark
# ---------------------------------------------------------------------- #
@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_seconds: float
    per_call_ms: float
    overhead_pct: float  # relative to baseline; 0.0 for baseline itself
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkReport:
    iterations: int
    baseline_per_call_ms: float
    results: List[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations,
            "baseline_per_call_ms": self.baseline_per_call_ms,
            "results": [r.to_dict() for r in self.results],
        }

    def render(self) -> str:
        lines: list[str] = []
        lines.append(f"AgentReplay overhead benchmark — {self.iterations} iterations")
        lines.append(f"Baseline (uninstrumented): {self.baseline_per_call_ms:.4f} ms/call")
        lines.append("")
        lines.append(f"{'Tool':<28s} {'ms/call':>10s} {'overhead%':>10s}")
        lines.append("-" * 50)
        for r in self.results:
            lines.append(f"{r.name:<28s} {r.per_call_ms:>10.4f} {r.overhead_pct:>10.2f}")
        lines.append("")
        # Verdict
        agentreplay_result = next((r for r in self.results if r.name == "AgentReplay (record)"), None)
        if agentreplay_result is not None:
            target = 5.0  # §7.2: target ≤ Laminar's ~5%
            if agentreplay_result.overhead_pct <= target:
                lines.append(
                    f"✓ AgentReplay overhead = {agentreplay_result.overhead_pct:.2f}% "
                    f"(≤ 5% target from §7.2)"
                )
            else:
                lines.append(
                    f"✗ AgentReplay overhead = {agentreplay_result.overhead_pct:.2f}% "
                    f"(> 5% target from §7.2) — investigate interceptor hot path"
                )
        return "\n".join(lines)


def _time_run(fn: Callable[[], Any], iterations: int, repeats: int = 3) -> float:
    """Run `fn` `repeats` times and return the median total seconds."""
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def run_benchmark(
    iterations: int = 200,
    *,
    repeats: int = 3,
    cassette_dir: Optional[Path] = None,
) -> BenchmarkReport:
    """Run the overhead benchmark.

    Measures the per-call latency of:

      1. Uninstrumented baseline (raw stub client)
      2. AgentReplay in RECORD mode
      3. AgentReplay in REPLAY mode
      4. Synthetic LangSmith (~0% overhead)
      5. Synthetic Laminar (~5% overhead)
      6. Synthetic AgentOps (~12% overhead)
      7. Synthetic Langfuse (~15% overhead)
    """
    if cassette_dir is None:
        cassette_dir = Path("/tmp/agentreplay-bench")
    if cassette_dir.exists():
        import shutil
        shutil.rmtree(cassette_dir)
    cassette_dir.mkdir(parents=True, exist_ok=True)

    # --- Baseline: raw stub client, no instrumentation ----------------
    stub = _StubLLM(iterations)
    baseline_total = _time_run(
        lambda: _run_workload(stub, iterations), iterations, repeats=repeats
    )
    baseline_per_call_ms = (baseline_total / iterations) * 1000.0

    report = BenchmarkReport(
        iterations=iterations,
        baseline_per_call_ms=baseline_per_call_ms,
        results=[
            BenchmarkResult(
                name="baseline",
                iterations=iterations,
                total_seconds=baseline_total,
                per_call_ms=baseline_per_call_ms,
                overhead_pct=0.0,
                extra={"kind": "uninstrumented"},
            )
        ],
    )

    # --- AgentReplay RECORD mode --------------------------------------
    rec_cassette = cassette_dir / "record"
    stub = _StubLLM(iterations)
    with Recorder.create(rec_cassette, framework="raw", agent_name="bench") as rec:
        client = rec.wrap_custom_client(stub)
        record_total = _time_run(
            lambda: _run_workload(client, iterations), iterations, repeats=repeats
        )
    record_per_call_ms = (record_total / iterations) * 1000.0
    record_overhead = ((record_total - baseline_total) / baseline_total) * 100.0
    report.results.append(
        BenchmarkResult(
            name="AgentReplay (record)",
            iterations=iterations,
            total_seconds=record_total,
            per_call_ms=record_per_call_ms,
            overhead_pct=record_overhead,
            extra={"kind": "agentreplay", "mode": "record"},
        )
    )

    # --- AgentReplay REPLAY mode (should be FASTER than baseline,
    #     because the stub client never runs) ---------------------------
    from agentreplay import Replayer
    stub = _StubLLM(iterations)  # never called in pure replay
    with Replayer.open(rec_cassette, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(stub)
        replay_total = _time_run(
            lambda: _run_workload(client, iterations), iterations, repeats=repeats
        )
    replay_per_call_ms = (replay_total / iterations) * 1000.0
    replay_overhead = ((replay_total - baseline_total) / baseline_total) * 100.0
    report.results.append(
        BenchmarkResult(
            name="AgentReplay (replay)",
            iterations=iterations,
            total_seconds=replay_total,
            per_call_ms=replay_per_call_ms,
            overhead_pct=replay_overhead,
            extra={"kind": "agentreplay", "mode": "replay"},
        )
    )

    # --- Synthetic baselines (mimic published 2026 figures) -----------
    for name, overhead_fraction in [
        ("LangSmith (synthetic)", 0.00),
        ("Laminar (synthetic)", 0.05),
        ("AgentOps (synthetic)", 0.12),
        ("Langfuse (synthetic)", 0.15),
    ]:
        client = _SyntheticOverheadClient(overhead_fraction)
        total = _time_run(
            lambda: _run_workload(client, iterations), iterations, repeats=repeats
        )
        per_call_ms = (total / iterations) * 1000.0
        overhead_pct = ((total - baseline_total) / baseline_total) * 100.0
        report.results.append(
            BenchmarkResult(
                name=name,
                iterations=iterations,
                total_seconds=total,
                per_call_ms=per_call_ms,
                overhead_pct=overhead_pct,
                extra={
                    "kind": "synthetic_baseline",
                    "target_overhead_fraction": overhead_fraction,
                },
            )
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AgentReplay overhead benchmark (per §7.2 of the product proposal)"
    )
    parser.add_argument("--iterations", type=int, default=200,
                        help="Number of LLM calls per measurement (default: 200)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of times to repeat each measurement and take the median (default: 3)")
    parser.add_argument("--report", type=str, default=None,
                        help="Write JSON report to this path")
    parser.add_argument("--json", action="store_true",
                        help="Print JSON report to stdout instead of human-readable")
    args = parser.parse_args()

    report = run_benchmark(iterations=args.iterations, repeats=args.repeats)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())

    if args.report:
        Path(args.report).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nReport written to {args.report}", file=sys.stderr)

    # Exit 0 if AgentReplay meets the ≤5% target, 1 otherwise.
    ar = next((r for r in report.results if r.name == "AgentReplay (record)"), None)
    if ar is None:
        return 1
    return 0 if ar.overhead_pct <= 5.0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
