"""Example: Real LangGraph integration with AgentReplay.

Builds an actual LangGraph ``StateGraph`` with two nodes (``router``
and ``synthesizer``), records a run, and replays it bit-exact with
zero model calls.

Run::

    pip install agentreplay[langgraph]
    python examples/langgraph_real.py record
    python examples/langgraph_real.py replay
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

from agentreplay import Recorder, Replayer
from agentreplay.constants import Mode
from agentreplay.frameworks.langgraph import bind_graph, wrap_llm

CASSETTE = Path(__file__).parent.parent / "cassettes" / "langgraph-real"


# ---------------------------------------------------------------------- #
# Stub LLM (no API key needed for the demo).
# ---------------------------------------------------------------------- #
class StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


# ---------------------------------------------------------------------- #
# Build a real LangGraph with two nodes.
# ---------------------------------------------------------------------- #
def build_and_compile_graph(client: Any, session: Any):
    """Build a 2-node LangGraph (router → synthesizer) and compile it.

    The graph is built and compiled INSIDE the bind_graph context
    manager so the session can patch each node's runnable to call
    enter_step before the node function runs.
    """
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class AgentState(TypedDict, total=False):
        messages: list
        intermediate: str
        final: str

    def router(state: AgentState) -> AgentState:
        """Call the LLM to decide what to do."""
        r = client.complete(
            messages=state.get("messages", []),
            model="stub",
        )
        return {"intermediate": r["text"]}

    def synthesizer(state: AgentState) -> AgentState:
        """Call the LLM again with the intermediate result."""
        r = client.complete(
            messages=list(state.get("messages", [])) + [
                {"role": "assistant", "content": state.get("intermediate", "")},
            ],
            model="stub",
        )
        return {"final": r["text"]}

    g = StateGraph(AgentState)
    g.add_node("router", router)
    g.add_node("synthesizer", synthesizer)
    g.add_edge(START, "router")
    g.add_edge("router", "synthesizer")
    g.add_edge("synthesizer", END)

    # bind_graph must wrap the graph BEFORE compile() so it can patch
    # the underlying node runnables.
    with bind_graph(session, g):
        return g.compile()


def run_once(client: Any, session: Any) -> str:
    """Build the graph (with bind_graph) and invoke it."""
    graph = build_and_compile_graph(client, session)
    result = graph.invoke({
        "messages": [{"role": "user", "content": "What is 6*7?"}],
    })
    return result.get("final", "")


def record() -> None:
    stub = StubLLM(
        responses=[
            {"text": "I should compute 6*7.", "usage": {"total_tokens": 12}},
            {"text": "Based on my reasoning, 6*7 = 42.", "usage": {"total_tokens": 18}},
        ]
    )
    with Recorder.create(CASSETTE, framework="langgraph", agent_name="lg-real", model="stub") as rec:
        client = wrap_llm(stub, rec, dialect="custom")
        result = run_once(client, rec)
        print(f"Final answer: {result!r}")
    print(f"Recorded to {CASSETTE}")


def replay() -> None:
    with Replayer.open(CASSETTE, mode=Mode.REPLAY) as rep:
        client = wrap_llm(StubLLM(responses=[]), rep, dialect="custom")
        result = run_once(client, rep)
        print(f"Final answer: {result!r}")
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
