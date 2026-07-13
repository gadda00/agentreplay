"""Session — thin convenience layer.

A :class:`Session` bundles a Recorder or Replayer and exposes a uniform
``wrap_*`` API so agent code can be written once and run in any of the
four modes (live / record / replay / hybrid) by swapping one line::

    with Session.replay("cassettes/run-001") as s:
        client = s.wrap_openai(openai_client)
        agent.run(client=client)
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Union

from agentreplay.constants import Mode
from agentreplay.cassette import Cassette
from agentreplay.recorder import Recorder
from agentreplay.replayer import Replayer


class Session:
    """Uniform front-door over :class:`Recorder` and :class:`Replayer`."""

    def __init__(self, inner: Union[Recorder, Replayer]) -> None:
        self.inner = inner
        self.cassette = inner.cassette
        self.mode = getattr(inner, "mode", Mode.RECORD)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def record(
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
    ) -> "Session":
        return cls(
            Recorder.create(
                root,
                framework=framework,
                agent_name=agent_name,
                task_id=task_id,
                model=model,
                outcome=outcome,
                tags=tags,
                extra=extra,
                git_commit=git_commit,
                step_id_provider=step_id_provider,
            )
        )

    @classmethod
    def replay(
        cls,
        root: Union[str, os.PathLike],
        *,
        mode: Mode = Mode.REPLAY,
        live_client: Any = None,
        live_http: Any = None,
    ) -> "Session":
        return cls(Replayer.open(root, mode=mode, live_client=live_client, live_http=live_http))

    @classmethod
    def live(cls) -> "Session":
        """A no-op session that passes everything through unchanged."""
        return cls(_LiveSession())

    # ------------------------------------------------------------------ #
    # Wrappers — delegate to inner
    # ------------------------------------------------------------------ #
    def wrap_openai(self, client: Any = None, **kwargs: Any) -> Any:
        return self.inner.wrap_openai(client, **kwargs)

    def wrap_anthropic(self, client: Any = None, **kwargs: Any) -> Any:
        return self.inner.wrap_anthropic(client, **kwargs)

    def wrap_custom_client(self, client: Any = None, **kwargs: Any) -> Any:
        return self.inner.wrap_custom_client(client, **kwargs)

    def wrap_http(self, client: Any = None, dialect: str = "httpx") -> Any:
        return self.inner.wrap_http(client, dialect=dialect)

    def wrap_tool(self, func: Callable[..., Any], name: Optional[str] = None) -> Any:
        return self.inner.wrap_tool(func, name=name)

    @property
    def clock(self) -> Any:
        return self.inner.clock

    @property
    def random(self) -> Any:
        return self.inner.random

    def enter_step(self, step_id: str) -> None:
        if hasattr(self.inner, "enter_step"):
            self.inner.enter_step(step_id)

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "Session":
        if isinstance(self.inner, Recorder):
            self.inner.__enter__()
        elif hasattr(self.inner, "__enter__"):
            self.inner.__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        if isinstance(self.inner, Recorder):
            self.inner.__exit__(*exc)
        elif hasattr(self.inner, "__exit__"):
            self.inner.__exit__(*exc)

    def __repr__(self) -> str:
        return f"<Session mode={self.mode!r} cassette={self.cassette!r}>"


class _LiveSession:
    """A pass-through "session" used when the agent should run normally
    without any recording or replay."""

    mode = Mode.LIVE
    cassette = None  # type: ignore[assignment]

    def wrap_openai(self, client: Any, **_: Any) -> Any:
        return client

    def wrap_anthropic(self, client: Any, **_: Any) -> Any:
        return client

    def wrap_custom_client(self, client: Any, **_: Any) -> Any:
        return client

    def wrap_http(self, client: Any, **_: Any) -> Any:
        return client

    def wrap_tool(self, func: Callable[..., Any], name: Optional[str] = None) -> Any:
        return func

    @property
    def clock(self) -> Any:
        import time as _t
        return _t

    @property
    def random(self) -> Any:
        import random
        return random

    def enter_step(self, step_id: str) -> None:
        pass
