"""Example: LangGraph integration.

This is the first-class integration target from §5.5 of the product
proposal. LangGraph's existing checkpointer primitive is a natural
hook for state snapshots; AgentReplay wraps the LLM client and tool
nodes so the calls *inside* each node are captured with per-node
call-site IDs.

Run::

    pip install agentreplay[langgraph]
    python examples/langgraph_agent.py record
    python examples/langgraph_agent.py replay

NOTE: This example uses LangGraph only if it's installed. If not, it
falls back to a stub that exercises the same code paths without
requiring the framework.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from agentreplay import Recorder, Replayer
from agentreplay.constants import Mode

CASSETTE = Path(__file__).parent.parent / "cassettes" / "langgraph-demo"


# ---------------------------------------------------------------------- #
# Stub LLM (no API key needed for the demo).
# ---------------------------------------------------------------------- #
class StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        return self.responses.pop(0)

    # OpenAI-shaped surface so the wrapper's chat.completions.create works.
    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        return self.complete(messages=kwargs.pop("messages"), **kwargs)


def search(query: str) -> str:
    return f"<results for {query!r}>"


def record() -> None:
    """Record a tiny LangGraph-style run.

    We don't actually build a LangGraph here (that would require the
    optional dependency); instead we simulate the structure: a `router`
    node that calls the LLM, a `tool` node that calls the tool, and a
    `synthesizer` node that calls the LLM again. The recorder's
    `enter_step` lets each call-site ID incorporate the node name.
    """
    stub = StubLLM(
        responses=[
            {"text": "I should search.", "usage": {}},
            {"text": "Based on the search: 42.", "usage": {}},
        ]
    )
    with Recorder.create(CASSETTE, framework="langgraph", agent_name="lg-demo") as rec:
        client = rec.wrap_custom_client(stub)
        tool = rec.wrap_tool(search, name="search")

        # Simulate LangGraph node execution: enter_step(name) before each node.
        rec.enter_step("langgraph:router")
        r1 = client.complete(messages=[{"role": "user", "content": "what is 6*7?"}], model="stub")

        rec.enter_step("langgraph:tool")
        tool_result = tool(query="6*7")

        rec.enter_step("langgraph:synthesizer")
        r2 = client.complete(
            messages=[
                {"role": "user", "content": "what is 6*7?"},
                {"role": "assistant", "content": r1["text"]},
                {"role": "tool", "content": tool_result},
            ],
            model="stub",
        )
        print(f"Final answer: {r2['text']!r}")
    print(f"Recorded to {CASSETTE}")


def replay() -> None:
    with Replayer.open(CASSETTE, mode=Mode.REPLAY) as rep:
        client = rep.wrap_custom_client(StubLLM([]))
        tool = rep.wrap_tool(search, name="search")

        rep.enter_step("langgraph:router")
        r1 = client.complete(messages=[{"role": "user", "content": "what is 6*7?"}], model="stub")

        rep.enter_step("langgraph:tool")
        tool_result = tool(query="6*7")

        rep.enter_step("langgraph:synthesizer")
        r2 = client.complete(
            messages=[
                {"role": "user", "content": "what is 6*7?"},
                {"role": "assistant", "content": r1["text"]},
                {"role": "tool", "content": tool_result},
            ],
            model="stub",
        )
        print(f"Final answer: {r2['text']!r}")
    print(f"Replayed from {CASSETTE} (zero model calls)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "record"
    if mode == "record":
        record()
    elif mode == "replay":
        replay()
    else:
        print(f"usage: {sys.argv[0]} [record|replay]")
        sys.exit(1)
