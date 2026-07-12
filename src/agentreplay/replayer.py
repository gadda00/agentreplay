"""Replayer — orchestrates a single pure-replay or hybrid-replay session.

The Replayer is the mirror image of :class:`Recorder`: it opens an
existing cassette in REPLAY (or HYBRID) mode and exposes the same
interceptor surface, so the agent's code can be run unchanged with the
only difference being that every external call is served from the
cassette instead of hitting the real network or model API.

In REPLAY mode, the first time the agent asks for a call whose
canonicalized input does not match any recorded call-site ID, a
:class:`DivergenceError` is raised — this is the *divergence detector*
from §5.3 of the product proposal. The CLI catches it and renders a
structural diff.

In HYBRID mode, divergence is not fatal: the engine falls through to a
live call so the developer can see where the new trajectory goes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from agentreplay.cassette import Cassette
from agentreplay.constants import Mode
from agentreplay.errors import CassetteNotFoundError
from agentreplay.interceptors import (
    RecordingClient,
    RecordingClock,
    RecordingHTTP,
    RecordingRandom,
    RecordingTool,
)


class Replayer:
    """Opens a cassette in REPLAY or HYBRID mode and exposes wrapped interceptors."""

    def __init__(
        self,
        cassette: Cassette,
        *,
        mode: Mode = Mode.REPLAY,
        live_client: Any = None,        # for HYBRID fallback
        live_http: Any = None,
        step_id_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        if mode not in (Mode.REPLAY, Mode.HYBRID):
            raise ValueError(f"Replayer mode must be REPLAY or HYBRID, got {mode}")
        self.cassette = cassette
        self.mode = mode
        self.live_client = live_client
        self.live_http = live_http
        # Mirror the Recorder's StepContext design so enter_step mutates
        # shared state that all interceptors reference.
        from agentreplay.recorder import StepContext
        self._step_context = StepContext()
        self._step_id_provider: Callable[[], str] = step_id_provider or self._step_context
        self._divergences: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def open(
        cls,
        root: Union[str, os.PathLike],
        *,
        mode: Mode = Mode.REPLAY,
        **kwargs: Any,
    ) -> "Replayer":
        path = Path(root)
        if not (path / Cassette.META_FILE).exists():
            raise CassetteNotFoundError(f"no cassette at {root!s}")
        cassette = Cassette.open(path, readonly=True)
        return cls(cassette, mode=mode, **kwargs)

    # ------------------------------------------------------------------ #
    # Wrappers — same surface as Recorder, but in REPLAY mode.
    # ------------------------------------------------------------------ #
    def wrap_openai(self, client: Any = None, **kwargs: Any) -> RecordingClient:
        return RecordingClient(
            client or self.live_client,
            self.cassette,
            mode=self.mode,
            call_type="openai",
            step_id_provider=self._step_context,
            **kwargs,
        )

    def wrap_anthropic(self, client: Any = None, **kwargs: Any) -> RecordingClient:
        return RecordingClient(
            client or self.live_client,
            self.cassette,
            mode=self.mode,
            call_type="anthropic",
            step_id_provider=self._step_context,
            **kwargs,
        )

    def wrap_custom_client(self, client: Any = None, **kwargs: Any) -> RecordingClient:
        return RecordingClient(
            client or self.live_client,
            self.cassette,
            mode=self.mode,
            call_type="custom",
            step_id_provider=self._step_context,
            **kwargs,
        )

    def wrap_http(self, client: Any = None, dialect: str = "httpx") -> RecordingHTTP:
        return RecordingHTTP(
            client or self.live_http,
            self.cassette,
            mode=self.mode,
            step_id_provider=self._step_context,
            dialect=dialect,
        )

    def wrap_tool(self, func: Callable[..., Any], name: Optional[str] = None) -> RecordingTool:
        return RecordingTool(
            func,
            name,
            self.cassette,
            mode=self.mode,
            step_id_provider=self._step_context,
        )

    @property
    def clock(self) -> RecordingClock:
        if not hasattr(self, "_clock"):
            self._clock = RecordingClock(
                self.cassette, mode=self.mode, step_id_provider=self._step_context
            )
        return self._clock

    @property
    def random(self) -> RecordingRandom:
        if not hasattr(self, "_rng"):
            self._rng = RecordingRandom(
                self.cassette, mode=self.mode, step_id_provider=self._step_context
            )
        return self._rng

    # ------------------------------------------------------------------ #
    # Step management + divergence tracking
    # ------------------------------------------------------------------ #
    def enter_step(self, step_id: str) -> None:
        """Pin the replayer's step context to a fixed step ID.

        Mirrors :meth:`agentreplay.Recorder.enter_step` so the same
        agent code (calling ``enter_step`` before each node) works in
        both RECORD and REPLAY mode.
        """
        self._step_context.set_static(step_id)
        self._step_id_provider = self._step_context

    def record_divergence(self, info: Dict[str, Any]) -> None:
        self._divergences.append(info)

    @property
    def divergences(self) -> List[Dict[str, Any]]:
        return list(self._divergences)

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "Replayer":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


class _ReplayStepProvider:
    """Default step-ID provider for replay — mirrors the recorder's
    monotonic counter so step IDs line up across record/replay runs."""

    def __init__(self) -> None:
        self._n = 0

    def __call__(self) -> str:
        s = f"step:{self._n}"
        self._n += 1
        return s


class _StaticStepProvider:
    def __init__(self, step_id: str) -> None:
        self.step_id = step_id

    def __call__(self) -> str:
        return self.step_id
