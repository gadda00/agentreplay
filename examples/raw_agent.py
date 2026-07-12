"""Example: record and replay a raw (framework-less) agent loop.

This is the simplest possible use of AgentReplay — no LangGraph, no
CrewAI, just a hand-rolled loop that calls the model and a tool. The
recording/replay boundary is the LLM client and the tool callable.

Run::

    # Record (uses the stub client; no API key needed)
    python examples/raw_agent.py record

    # Replay (zero model calls)
    python examples/raw_agent.py replay

    # Inspect the cassette
    agentreplay show cassettes/raw-demo --events
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from agentreplay import Recorder, Replayer, Session
from agentreplay.constants import Mode

CASSETTE = Path(__file__).parent.parent / "cassettes" / "raw-demo"


# ---------------------------------------------------------------------- #
# A tiny stub LLM client so the example runs without an API key.
# ---------------------------------------------------------------------- #
class StubLLM:
    """Replaces OpenAI/Anthropic for the demo. Returns canned responses."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        if not self.responses:
            raise RuntimeError("StubLLM exhausted")
        return self.responses.pop(0)


# ---------------------------------------------------------------------- #
# A tool the agent can call.
# ---------------------------------------------------------------------- #
def search(query: str) -> str:
    """A toy search tool. In a real agent this would hit a search API."""
    return f"<results for '{query}'>"


# ---------------------------------------------------------------------- #
# The agent itself — a 3-step loop: ask the model, call the tool, ask again.
# ---------------------------------------------------------------------- #
def run_agent(client: Any, tool: Any) -> str:
    r1 = client.complete(messages=[{"role": "user", "content": "What's the weather?"}], model="stub")
    tool_result = tool(query="weather")
    r2 = client.complete(
        messages=[
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": r1["text"]},
            {"role": "tool", "content": tool_result},
        ],
        model="stub",
    )
    return r2["text"]


# ---------------------------------------------------------------------- #
# Entry points
# ---------------------------------------------------------------------- #
def record() -> None:
    stub = StubLLM(
        responses=[
            {"text": "Let me search for that.", "usage": {"total_tokens": 12}},
            {"text": "Based on the search, it's sunny.", "usage": {"total_tokens": 18}},
        ]
    )
    with Recorder.create(CASSETTE, framework="raw", agent_name="demo", model="stub") as rec:
        client = rec.wrap_custom_client(stub)
        tool = rec.wrap_tool(search, name="search")
        result = run_agent(client, tool)
        print(f"Agent said: {result!r}")
    print(f"Recorded cassette to {CASSETTE}")


def replay() -> None:
    # Note: no API key, no live client — pure replay.
    with Replayer.open(CASSETTE, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(StubLLM(responses=[]))  # empty: live calls would raise
        tool = rep.wrap_tool(search, name="search")
        result = run_agent(client, tool)
        print(f"Agent said: {result!r}")
    print(f"Replayed cassette from {CASSETTE} (zero model calls)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "record"
    if mode == "record":
        record()
    elif mode == "replay":
        replay()
    else:
        print(f"usage: {sys.argv[0]} [record|replay]")
        sys.exit(1)
