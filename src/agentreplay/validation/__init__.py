"""Validation harness for SWE-bench Verified and GAIA.

Implements the evaluation methodology from §7 of the product proposal:

    * Reproduction fidelity (§7.1): replay every cassette in pure-replay
      mode and assert bit-exact reproduction of the originally recorded
      terminal state. Target: 100%.

    * Overhead (§7.2): measured by the standalone benchmark module
      (:mod:`agentreplay.benchmark.overhead`); not duplicated here.

    * Cost impact (§7.3): compute the marginal API cost of investigating
      each failure twice — once via a traditional live re-run, once via
      AgentReplay's pure replay — and report the delta. Because pure
      replay makes zero model calls, the expected result is a reduction
      approaching 100% of the *investigation* cost specifically.

This module provides the *harness* — the framework for running the
validation. The actual SWE-bench / GAIA task sets require either:

    1. Network access to download the public task corpora (SWE-bench
       Verified from GitHub, GAIA from HuggingFace), plus
    2. API keys for a real LLM (OpenAI/Anthropic) to record the initial
       cassettes.

Because neither is available in CI, the harness ships with a *synthetic*
task set that exercises the same code paths and lets the validation run
end-to-end without external dependencies. The :func:`run_validation`
function accepts a custom task loader so teams with API access can plug
in the real SWE-bench / GAIA task sets.

Usage::

    # Synthetic validation (no API key needed) — runs in CI
    python -m agentreplay.validation.swebench --tasks synthetic --out report.json

    # Real SWE-bench Verified (requires OPENAI_API_KEY + task download)
    python -m agentreplay.validation.swebench --tasks swebench-verified --limit 20

    # Real GAIA subset (requires OPENAI_API_KEY + task download)
    python -m agentreplay.validation.gaia --tasks gaia-subset --limit 20
"""
from agentreplay.validation.fidelity import (
    FidelityResult,
    FidelityReport,
    run_fidelity_check,
    run_validation,
)
from agentreplay.validation.tasks import (
    Task,
    TaskSet,
    SyntheticTaskSet,
    load_synthetic_tasks,
)

__all__ = [
    "FidelityResult",
    "FidelityReport",
    "run_fidelity_check",
    "run_validation",
    "Task",
    "TaskSet",
    "SyntheticTaskSet",
    "load_synthetic_tasks",
]
