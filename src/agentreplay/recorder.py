"""Recorder — orchestrates a single recording session.

The Recorder owns the cassette and exposes the interceptors that the
agent's code should use in place of the real client / clock / random /
http objects. Use it as a context manager so the cassette header is
always flushed, even on exception::

    with Recorder.create("cassettes/run-001", framework="langgraph") as rec:
        client = rec.wrap_openai(openai.OpenAI())
        clock  = rec.clock
        agent.run(client=client, clock=clock)
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from agentreplay.cassette import Cassette
from agentreplay.constants import Mode
from agentreplay.interceptors import (
    RecordingClient,
    RecordingClock,
    RecordingHTTP,
    RecordingRandom,
    RecordingTool,
)


def _git_commit(cwd: Optional[str] = None) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=cwd or os.getcwd()
        ).decode().strip()
        return out
    except Exception:  # pragma: no cover - non-fatal
        return ""


class Recorder:
    """Owns a cassette in RECORD mode and exposes wrapped interceptors."""

    def __init__(self, cassette: Cassette, *, step_id_provider: Optional[Callable[[], str]] = None) -> None:
        self.cassette = cassette
        # Always use a StepContext so enter_step can mutate the shared
        # state that all interceptors reference. If a custom provider
        # is passed, we still create a StepContext but it won't be used
        # unless someone calls enter_step.
        self._step_context = StepContext()
        self._step_id_provider: Callable[[], str] = step_id_provider or self._step_context
        self._closed = False
        self._start_time = time.time()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        root: Union[str, os.PathLike],
        *,
        framework: str = "raw",
        agent_name: str = "",
        task_id: str = "",
        model: str = "",
        outcome: str = "",
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        git_commit: Optional[str] = None,
        step_id_provider: Optional[Callable[[], str]] = None,
    ) -> "Recorder":
        cassette = Cassette.create(
            root,
            framework=framework,
            agent_name=agent_name,
            task_id=task_id,
            git_commit=git_commit if git_commit is not None else _git_commit(),
            model=model,
            outcome=outcome,
            tags=tags,
            extra=extra,
        )
        return cls(cassette, step_id_provider=step_id_provider)

    @classmethod
    def open_for_record(cls, root: Union[str, os.PathLike], **kwargs: Any) -> "Recorder":
        """Re-open an existing cassette to append more events.

        Rarely used in practice — most workflows record once and replay
        many times — but useful for resuming an interrupted capture.
        """
        cassette = Cassette.open(root, readonly=False)
        for k, v in kwargs.items():
            setattr(cassette.meta, k, v)
        return cls(cassette)

    # ------------------------------------------------------------------ #
    # Wrappers
    # ------------------------------------------------------------------ #
    def wrap_openai(self, client: Any, **kwargs: Any) -> RecordingClient:
        return RecordingClient(
            client,
            self.cassette,
            mode=Mode.RECORD,
            call_type="openai",
            step_id_provider=self._step_context,
            **kwargs,
        )

    def wrap_anthropic(self, client: Any, **kwargs: Any) -> RecordingClient:
        return RecordingClient(
            client,
            self.cassette,
            mode=Mode.RECORD,
            call_type="anthropic",
            step_id_provider=self._step_context,
            **kwargs,
        )

    def wrap_custom_client(self, client: Any, **kwargs: Any) -> RecordingClient:
        return RecordingClient(
            client,
            self.cassette,
            mode=Mode.RECORD,
            call_type="custom",
            step_id_provider=self._step_context,
            **kwargs,
        )

    def wrap_http(self, client: Any, dialect: str = "httpx") -> RecordingHTTP:
        return RecordingHTTP(
            client,
            self.cassette,
            mode=Mode.RECORD,
            step_id_provider=self._step_context,
            dialect=dialect,
        )

    def wrap_tool(self, func: Callable[..., Any], name: Optional[str] = None) -> RecordingTool:
        return RecordingTool(
            func,
            name,
            self.cassette,
            mode=Mode.RECORD,
            step_id_provider=self._step_context,
        )

    @property
    def clock(self) -> RecordingClock:
        if not hasattr(self, "_clock"):
            self._clock = RecordingClock(
                self.cassette, mode=Mode.RECORD, step_id_provider=self._step_context
            )
        return self._clock

    @property
    def random(self) -> RecordingRandom:
        if not hasattr(self, "_rng"):
            self._rng = RecordingRandom(
                self.cassette, mode=Mode.RECORD, step_id_provider=self._step_context
            )
        return self._rng

    # ------------------------------------------------------------------ #
    # Step management
    # ------------------------------------------------------------------ #
    def enter_step(self, step_id: str) -> None:
        """Tell the recorder that the agent has entered a new step.

        Optional but recommended: lets the call-site IDs incorporate the
        framework's own notion of a step (LangGraph node name, CrewAI
        task ID, ...) instead of a monotonic counter.

        This mutates the shared :class:`StepContext` that all
        interceptors reference, so the new step ID takes effect
        immediately for every interceptor created via ``wrap_*``.
        """
        self._step_context.set_static(step_id)
        # Also update the step_id_provider in case it was a custom one —
        # the StepContext is always the source of truth.
        self._step_id_provider = self._step_context

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self, *, outcome: Optional[str] = None) -> None:
        if self._closed:
            return
        self._closed = True
        if outcome is not None:
            self.cassette.meta.outcome = outcome
        self.cassette.meta.duration_ms = (time.time() - self._start_time) * 1000.0
        self.cassette.save()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc: Any) -> None:
        # Only auto-derive outcome if the user has not already set one
        # explicitly (e.g. via Recorder.create(outcome="fail") for a
        # pre-classified regression cassette).
        existing = self.cassette.meta.outcome
        if existing in ("", None):
            outcome = "fail" if exc[1] is not None else "pass"
        else:
            outcome = existing
        self.close(outcome=outcome)

    def __repr__(self) -> str:
        return f"<Recorder cassette={self.cassette.meta.id!r} events={len(self.cassette.events)}>"


# ---------------------------------------------------------------------- #
# Step-ID providers
# ---------------------------------------------------------------------- #
class StepContext:
    """Mutable shared step-ID state.

    All interceptors hold a reference to the SAME ``StepContext``
    instance, so when ``enter_step`` updates the current step, every
    interceptor immediately sees the new value. This is critical for
    framework adapters like LangGraph's ``bind_graph`` which call
    ``enter_step`` from a node wrapper while the LLM interceptor is
    making calls inside the node.

    The context has two modes:

      - **counter** mode (default): returns ``step:N`` with a monotonic
        counter. Used when no framework adapter is plugged in.
      - **static** mode: returns the step ID set by ``enter_step``.
        Used when a framework adapter (LangGraph, CrewAI, ...) is
        providing per-node step IDs.
    """

    def __init__(self) -> None:
        self._counter = 0
        self._static: Optional[str] = None

    def __call__(self) -> str:
        if self._static is not None:
            return self._static
        s = f"step:{self._counter}"
        self._counter += 1
        return s

    def set_static(self, step_id: str) -> None:
        """Pin the context to a fixed step ID (framework adapter mode)."""
        self._static = step_id

    def clear_static(self) -> None:
        """Return to counter mode."""
        self._static = None


# Backwards-compatible aliases
_DefaultStepCounter = StepContext  # kept for backwards compat; use StepContext
