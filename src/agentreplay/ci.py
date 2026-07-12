"""CI regression runner.

Replays a corpus of cassettes against a configurable agent entry point
and fails the build if any cassette diverges from its recording. Because
pure replay makes zero model calls, an arbitrarily large corpus costs
nothing in inference spend to run on every PR — this is the direct
mechanism from §5.7 of the product proposal for turning captured
failures into a permanent, zero-cost regression suite.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from agentreplay.constants import Mode
from agentreplay.cassette import Cassette
from agentreplay.diff import Diff, diff_structural, render_diff
from agentreplay.errors import AgentReplayError, DivergenceError
from agentreplay.replayer import Replayer


@dataclass
class RegressionResult:
    """Outcome of replaying one cassette through the agent."""

    cassette_id: str
    cassette_path: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    diff: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cassette_id": self.cassette_id,
            "cassette_path": self.cassette_path,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "diff": self.diff,
        }


@dataclass
class RegressionReport:
    """Aggregate outcome for a whole corpus run."""

    results: List[RegressionResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "num_total": len(self.results),
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "results": [r.to_dict() for r in self.results],
        }

    def render(self) -> str:
        lines: list[str] = []
        lines.append(f"AgentReplay regression: {self.num_passed}/{len(self.results)} passed")
        for r in self.results:
            mark = "✓" if r.passed else "✗"
            lines.append(f"  {mark} {r.cassette_id} ({r.duration_ms:.1f} ms)")
            if not r.passed:
                if r.error:
                    lines.append(f"      error: {r.error}")
                if r.diff:
                    lines.append(f"      summary: {r.diff}")
        return "\n".join(lines)


def discover_cassettes(root: Union[str, os.PathLike]) -> List[Path]:
    """Find every cassette directory under ``root``.

    A cassette is any directory containing a ``cassette.json`` file.
    """
    root = Path(root)
    if not root.exists():
        return []
    found: list[Path] = []
    if (root / Cassette.META_FILE).exists():
        found.append(root)
        return found
    for path in root.rglob(Cassette.META_FILE):
        found.append(path.parent)
    return sorted(found)


def run_corpus(
    corpus_root: Union[str, os.PathLike],
    agent_run: Callable[[Replayer], Any],
    *,
    stop_on_first_failure: bool = False,
    tag_filter: Optional[str] = None,
    outcome_filter: Optional[str] = None,
    live_client: Any = None,
    live_http: Any = None,
) -> RegressionReport:
    """Replay every cassette in ``corpus_root`` through ``agent_run``.

    ``agent_run`` is the same callable shape used by
    :func:`agentreplay.mutate.mutate_and_replay` — it receives a
    :class:`Replayer` and is expected to run the agent's code using
    the replayer's ``wrap_*`` interceptors.

    Each cassette is replayed in pure REPLAY mode; a divergence counts
    as a failure for that cassette. The returned :class:`RegressionReport`
    can be rendered to stdout or serialised to JSON for the CI log.
    """
    report = RegressionReport()
    for cassette_path in discover_cassettes(corpus_root):
        try:
            cassette = Cassette.open(cassette_path, readonly=True)
        except Exception as exc:  # pragma: no cover
            report.results.append(
                RegressionResult(
                    cassette_id=str(cassette_path),
                    cassette_path=str(cassette_path),
                    passed=False,
                    duration_ms=0.0,
                    error=f"failed to open: {exc!r}",
                )
            )
            if stop_on_first_failure:
                return report
            continue

        if tag_filter and tag_filter not in cassette.meta.tags:
            continue
        if outcome_filter and cassette.meta.outcome != outcome_filter:
            continue

        started = time.time()
        try:
            replayer = Replayer.open(
                cassette_path,
                mode=Mode.REPLAY,
                live_client=live_client,
                live_http=live_http,
            )
            agent_run(replayer)
            duration_ms = (time.time() - started) * 1000.0
            report.results.append(
                RegressionResult(
                    cassette_id=cassette.meta.id,
                    cassette_path=str(cassette_path),
                    passed=True,
                    duration_ms=duration_ms,
                )
            )
        except DivergenceError as exc:
            duration_ms = (time.time() - started) * 1000.0
            report.results.append(
                RegressionResult(
                    cassette_id=cassette.meta.id,
                    cassette_path=str(cassette_path),
                    passed=False,
                    duration_ms=duration_ms,
                    error=str(exc),
                    diff={
                        "step_id": exc.step_id,
                        "call_type": exc.call_type,
                        "recorded_call_id": exc.expected_call_id,
                        "actual_call_id": exc.actual_call_id,
                    },
                )
            )
            if stop_on_first_failure:
                return report
        except AgentReplayError as exc:
            duration_ms = (time.time() - started) * 1000.0
            report.results.append(
                RegressionResult(
                    cassette_id=cassette.meta.id,
                    cassette_path=str(cassette_path),
                    passed=False,
                    duration_ms=duration_ms,
                    error=str(exc),
                )
            )
            if stop_on_first_failure:
                return report
    return report


def run_corpus_and_exit(
    corpus_root: Union[str, os.PathLike],
    agent_run: Callable[[Replayer], Any],
    *,
    json_output: bool = False,
    **kwargs: Any,
) -> None:
    """Run :func:`run_corpus` and ``sys.exit`` with the right status code.

    Designed to be called from a project's CI entry point. Passes
    through any extra kwargs to :func:`run_corpus`.
    """
    report = run_corpus(corpus_root, agent_run, **kwargs)
    if json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())
    sys.exit(0 if report.passed else 1)
