"""Auto-init helpers.

When the CLI's ``agentreplay record`` subcommand execs a child process,
it sets environment variables describing the desired recording session.
``agentreplay.auto.init()`` reads those env vars and returns a ready-to-
use :class:`Session`, so the child process can pick up recording without
having to repeat the path/framework/task configuration.
"""
from __future__ import annotations

import os
from typing import Optional

from agentreplay.constants import Mode
from agentreplay.session import Session


def init() -> Optional[Session]:
    """Inspect the environment and return a configured :class:`Session`.

    Returns ``None`` if no AgentReplay env vars are set (i.e. the
    process was not launched via ``agentreplay record``).
    """
    mode = os.environ.get("AGENTREPLAY_MODE", "").lower()
    cassette = os.environ.get("AGENTREPLAY_CASSETTE")
    if not mode or not cassette:
        return None

    if mode == Mode.RECORD.value:
        return Session.record(
            cassette,
            framework=os.environ.get("AGENTREPLAY_FRAMEWORK", "raw"),
            agent_name=os.environ.get("AGENTREPLAY_AGENT_NAME", ""),
            task_id=os.environ.get("AGENTREPLAY_TASK_ID", ""),
            model=os.environ.get("AGENTREPLAY_MODEL", ""),
            outcome=os.environ.get("AGENTREPLAY_OUTCOME", ""),
            tags=[t for t in os.environ.get("AGENTREPLAY_TAGS", "").split(",") if t],
        )
    if mode in (Mode.REPLAY.value, Mode.HYBRID.value):
        return Session.replay(cassette, mode=Mode(mode))
    if mode == Mode.LIVE.value:
        return Session.live()
    return None
