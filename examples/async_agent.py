"""Example: async agent with AgentReplay.

Demonstrates recording and replaying async LLM calls using
``RecordingClient.acomplete()`` — the async counterpart to ``complete()``.

Run::

    python examples/async_agent.py record
    python examples/async_agent.py replay
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

from agentreplay import Recorder, Replayer
from agentreplay.constants import Mode

CASSETTE = Path(__file__).parent.parent / "cassettes" / "async-demo"


class AsyncStubLLM:
    """Async stub LLM with an acomplete coroutine method."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.live_calls = 0

    async def acomplete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        self.live_calls += 1
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


async def run_agent(client: Any) -> str:
    """Simulate an async agent loop."""
    r1 = await client.acomplete(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        model="stub",
    )
    r2 = await client.acomplete(
        messages=[
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": r1["text"]},
            {"role": "user", "content": "Now multiply by 3"},
        ],
        model="stub",
    )
    return r2["text"]


async def record() -> None:
    stub = AsyncStubLLM([
        {"text": "2+2 = 4.", "usage": {"total_tokens": 10}},
        {"text": "4 × 3 = 12.", "usage": {"total_tokens": 12}},
    ])
    with Recorder.create(CASSETTE, framework="raw", agent_name="async-demo") as rec:
        client = rec.wrap_custom_client(stub)
        result = await run_agent(client)
        print(f"Agent said: {result!r}")
    print(f"Recorded async cassette to {CASSETTE}")


async def replay() -> None:
    with Replayer.open(CASSETTE, mode=Mode.REPLAY) as rep:
        # Empty stub — should never be called during pure replay
        client = rep.wrap_custom_client(AsyncStubLLM([]))
        result = await run_agent(client)
        print(f"Agent said: {result!r}")
    print(f"Replayed async cassette from {CASSETTE} (zero model calls)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "record"
    if mode == "record":
        asyncio.run(record())
    elif mode == "replay":
        asyncio.run(replay())
    else:
        print(f"usage: {sys.argv[0]} [record|replay]")
        sys.exit(1)
