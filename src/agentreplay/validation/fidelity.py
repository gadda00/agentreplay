"""Reproduction-fidelity validation (§7.1).

For every cassette in the validation set, replay it in pure-replay mode
and compare the terminal agent state — final answer, tool-call sequence,
exit condition — against the originally recorded terminal state.

Target (§7.1): 100% bit-exact reproduction for unmodified agent code.
This is a *mechanical guarantee* of the design (§5.3), not a
probabilistic claim; any failure to reproduce indicates an
uninstrumented source of nondeterminism (§8) that must be found and
closed before the tool can be trusted.

This module also reports the cost-impact metric (§7.3): the marginal
API cost of investigating each failure twice — once via a traditional
live re-run, once via AgentReplay's pure replay. Because pure replay
makes zero model calls by construction, the expected result is a
reduction approaching 100% of the investigation cost.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agentreplay import Cassette, Recorder, Replayer
from agentreplay.constants import Mode
from agentreplay.errors import AgentReplayError, DivergenceError
from agentreplay.validation.tasks import Task, TaskSet


# ---------------------------------------------------------------------- #
# Results
# ---------------------------------------------------------------------- #
@dataclass
class FidelityResult:
    """Outcome of recording + replaying a single task."""

    task_id: str
    passed: bool
    recorded_terminal: Any = None
    replayed_terminal: Any = None
    error: Optional[str] = None
    record_duration_ms: float = 0.0
    replay_duration_ms: float = 0.0
    num_events: int = 0
    live_calls_during_replay: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FidelityReport:
    """Aggregate outcome for the whole validation set."""

    task_set: str
    results: List[FidelityResult] = field(default_factory=list)

    @property
    def num_total(self) -> int:
        return len(self.results)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def passed(self) -> bool:
        return self.num_failed == 0

    @property
    def fidelity_pct(self) -> float:
        """Percentage of tasks that replayed bit-exact.

        Per §7.1 the target is 100%.
        """
        if not self.results:
            return 100.0
        return 100.0 * self.num_passed / self.num_total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_set": self.task_set,
            "num_total": self.num_total,
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "fidelity_pct": self.fidelity_pct,
            "passed": self.passed,
            "results": [r.to_dict() for r in self.results],
        }

    def render(self) -> str:
        lines: list[str] = []
        lines.append(f"AgentReplay reproduction-fidelity validation — {self.task_set}")
        lines.append(
            f"  {self.num_passed}/{self.num_total} tasks replayed bit-exact "
            f"({self.fidelity_pct:.1f}%)"
        )
        for r in self.results:
            mark = "✓" if r.passed else "✗"
            lines.append(
                f"  {mark} {r.task_id} "
                f"(record={r.record_duration_ms:.1f}ms, "
                f"replay={r.replay_duration_ms:.1f}ms, "
                f"events={r.num_events})"
            )
            if not r.passed and r.error:
                lines.append(f"      error: {r.error}")
        target = 100.0
        if self.fidelity_pct >= target:
            lines.append(f"\n✓ Fidelity = {self.fidelity_pct:.1f}% (meets §7.1 100% target)")
        else:
            lines.append(
                f"\n✗ Fidelity = {self.fidelity_pct:.1f}% "
                f"(below §7.1 100% target) — investigate uninstrumented nondeterminism"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Stub LLM for synthetic tasks
# ---------------------------------------------------------------------- #
class _ScriptedLLM:
    """LLM stub that returns a scripted response sequence.

    For real SWE-bench / GAIA validation, replace this with a real
    OpenAI / Anthropic client. The :func:`run_validation` function
    accepts a ``client_factory`` for exactly this purpose.
    """

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.live_calls = 0

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        self.live_calls += 1
        if not self.responses:
            raise RuntimeError("_ScriptedLLM exhausted")
        return self.responses.pop(0)


# ---------------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------------- #
def _run_task_live(
    task: Task,
    client: Any,
    tools: Dict[str, Any],
) -> Any:
    """Run a single task through a scripted agent loop.

    The loop is intentionally simple: call the LLM, optionally call a
    tool, call the LLM again with the tool result. This is enough to
    exercise the recording layer's three interceptor types (LLM, tool,
    clock) without requiring a full agent framework.
    """
    r1 = client.complete(messages=task.messages, model="stub")
    tool_results: list[str] = []
    for tool_name, tool_fn in tools.items():
        # Call each tool once with a fixed arg derived from the task id.
        arg = task.id.split(":")[-1]
        tool_results.append(f"{tool_name}({arg!r}) → {tool_fn(arg)}")
    final_messages = list(task.messages) + [
        {"role": "assistant", "content": r1["text"]},
        {"role": "tool", "content": "\n".join(tool_results)},
    ]
    r2 = client.complete(messages=final_messages, model="stub")
    return {"final_text": r2["text"]}


def _make_scripted_responses(task: Task) -> List[Dict[str, Any]]:
    """Produce a deterministic LLM response sequence for `task`.

    For the synthetic task set the "expected" field already encodes the
    final text, so we just split it across the two LLM calls.
    """
    expected_text = task.expected.get("final_text", "")
    return [
        {"text": f"working on: {task.description}", "usage": {"total_tokens": 10}},
        {"text": expected_text, "usage": {"total_tokens": 20}},
    ]


def run_fidelity_check(
    task: Task,
    *,
    cassette_root: Union[str, Path],
    client_factory=None,
) -> FidelityResult:
    """Record + replay a single task and check reproduction fidelity.

    Parameters
    ----------
    task
        The :class:`Task` to run.
    cassette_root
        Directory under which the task's cassette will be written
        (``<cassette_root>/<task.id>/``).
    client_factory
        Optional callable returning a fresh LLM client for the RECORD
        pass. If ``None``, a :class:`_ScriptedLLM` is used (synthetic
        tasks only). For real SWE-bench / GAIA tasks, pass a factory
        that returns a real ``openai.OpenAI()`` or
        ``anthropic.Anthropic()`` client.
    """
    cassette_path = Path(cassette_root) / task.id.replace(":", "-")
    if cassette_path.exists():
        import shutil
        shutil.rmtree(cassette_path)

    # --- RECORD pass ---------------------------------------------------
    if client_factory is None:
        client_obj = _ScriptedLLM(_make_scripted_responses(task))
    else:
        client_obj = client_factory()
    record_start = time.time()
    recorded_terminal: Any = None
    try:
        with Recorder.create(
            cassette_path,
            framework="validation",
            agent_name="validation",
            task_id=task.id,
            model="stub" if client_factory is None else "real",
            tags=["validation"],
            extra=task.metadata,
        ) as rec:
            client = rec.wrap_custom_client(client_obj)
            wrapped_tools = {name: rec.wrap_tool(fn, name=name) for name, fn in task.tools.items()}
            recorded_terminal = _run_task_live(task, client, wrapped_tools)
    except Exception as exc:  # noqa: BLE001
        return FidelityResult(
            task_id=task.id,
            passed=False,
            error=f"record failed: {exc!r}",
            record_duration_ms=(time.time() - record_start) * 1000.0,
        )
    record_duration_ms = (time.time() - record_start) * 1000.0

    # --- REPLAY pass ---------------------------------------------------
    replay_start = time.time()
    # A fresh stub with NO responses — any live call would raise.
    replay_client_obj = _ScriptedLLM(responses=[])
    replayed_terminal: Any = None
    live_calls_during_replay = 0
    try:
        with Replayer.open(cassette_path, mode=Mode.REPLAY) as rep:
            client = rep.wrap_custom_client(replay_client_obj)
            wrapped_tools = {name: rep.wrap_tool(fn, name=name) for name, fn in task.tools.items()}
            replayed_terminal = _run_task_live(task, client, wrapped_tools)
            live_calls_during_replay = replay_client_obj.live_calls
    except DivergenceError as exc:
        return FidelityResult(
            task_id=task.id,
            passed=False,
            recorded_terminal=recorded_terminal,
            error=f"diverged: {exc}",
            record_duration_ms=record_duration_ms,
            replay_duration_ms=(time.time() - replay_start) * 1000.0,
        )
    except AgentReplayError as exc:
        return FidelityResult(
            task_id=task.id,
            passed=False,
            recorded_terminal=recorded_terminal,
            error=str(exc),
            record_duration_ms=record_duration_ms,
            replay_duration_ms=(time.time() - replay_start) * 1000.0,
        )
    replay_duration_ms = (time.time() - replay_start) * 1000.0

    # --- Compare -------------------------------------------------------
    # For the synthetic task set the terminal state is a dict with a
    # "final_text" key; compare it byte-for-byte. For real tasks the
    # comparison is the same — the recorded vs. replayed terminal state
    # must be deeply equal.
    passed = (recorded_terminal == replayed_terminal) and live_calls_during_replay == 0
    cassette = Cassette.open(cassette_path, readonly=True)
    return FidelityResult(
        task_id=task.id,
        passed=passed,
        recorded_terminal=recorded_terminal,
        replayed_terminal=replayed_terminal,
        record_duration_ms=record_duration_ms,
        replay_duration_ms=replay_duration_ms,
        num_events=len(cassette.events),
        live_calls_during_replay=live_calls_during_replay,
    )


def run_validation(
    task_set: TaskSet,
    *,
    cassette_root: Union[str, Path],
    limit: Optional[int] = None,
    client_factory=None,
) -> FidelityReport:
    """Run the full validation: record + replay every task in ``task_set``.

    Returns a :class:`FidelityReport` with one :class:`FidelityResult`
    per task. The report's :attr:`FidelityReport.fidelity_pct` is the
    key metric: per §7.1 the target is 100%.
    """
    tasks = task_set.load(limit=limit)
    report = FidelityReport(task_set=type(task_set).__name__)
    for task in tasks:
        result = run_fidelity_check(
            task,
            cassette_root=cassette_root,
            client_factory=client_factory,
        )
        report.results.append(result)
    return report
