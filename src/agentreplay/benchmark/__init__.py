"""Benchmark suite for AgentReplay.

Currently includes:
    * :mod:`agentreplay.benchmark.overhead` — recording-layer latency
      overhead vs. uninstrumented baseline and synthetic platform
      baselines (per §7.2 of the product proposal).
"""
from agentreplay.benchmark.overhead import (
    BenchmarkReport,
    BenchmarkResult,
    run_benchmark,
)

__all__ = ["BenchmarkReport", "BenchmarkResult", "run_benchmark"]
