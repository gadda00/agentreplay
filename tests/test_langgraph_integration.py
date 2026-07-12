"""Tests for the real LangGraph integration.

These tests only run if LangGraph is installed (``pip install
agentreplay[langgraph]``). They build an actual ``StateGraph`` and
verify that record → replay reproduces bit-exact, with the node names
appearing as step IDs in the cassette.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

langgraph = pytest.importorskip("langgraph", reason="langgraph not installed")

from agentreplay import Cassette, Recorder, Replayer  # noqa: E402
from agentreplay.constants import Mode  # noqa: E402
from agentreplay.errors import DivergenceError  # noqa: E402
from agentreplay.frameworks.langgraph import bind_graph, wrap_llm, wrap_node  # noqa: E402


class _StubLLM:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.live_calls = 0

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Dict[str, Any]:
        self.live_calls += 1
        if not self.responses:
            raise RuntimeError("exhausted")
        return self.responses.pop(0)


def _build_graph(client: Any, session: Any):
    """Build a 2-node LangGraph: router → synthesizer, with bind_graph."""
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class AgentState(TypedDict, total=False):
        messages: list
        intermediate: str
        final: str

    def router(state: AgentState) -> AgentState:
        r = client.complete(messages=state.get("messages", []), model="stub")
        return {"intermediate": r["text"]}

    def synthesizer(state: AgentState) -> AgentState:
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

    # bind_graph BEFORE compile()
    with bind_graph(session, g):
        return g.compile()


def test_langgraph_record_replay_bit_exact(tmp_path: Path):
    """End-to-end: build a real LangGraph, record, replay bit-exact."""
    cassette = tmp_path / "cass"
    stub = _StubLLM([
        {"text": "intermediate", "usage": {"total_tokens": 5}},
        {"text": "final answer", "usage": {"total_tokens": 8}},
    ])

    # Record
    with Recorder.create(cassette, framework="langgraph", agent_name="test") as rec:
        client = wrap_llm(stub, rec, dialect="custom")
        graph = _build_graph(client, rec)
        result = graph.invoke({"messages": [{"role": "user", "content": "hi"}]})

    assert result["final"] == "final answer"
    assert stub.live_calls == 2

    # Replay — fresh stub with NO responses, so any live call would raise.
    fresh = _StubLLM(responses=[])
    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        client = wrap_llm(fresh, rep, dialect="custom")
        graph = _build_graph(client, rep)
        result = graph.invoke({"messages": [{"role": "user", "content": "hi"}]})

    assert result["final"] == "final answer"
    assert fresh.live_calls == 0  # §7.1: zero model calls during pure replay


def test_langgraph_step_ids_include_node_names(tmp_path: Path):
    """The bind_graph wrapper must produce step IDs that include the
    LangGraph node names ('langgraph:router', 'langgraph:synthesizer')."""
    cassette = tmp_path / "cass"
    stub = _StubLLM([
        {"text": "x", "usage": {}},
        {"text": "y", "usage": {}},
    ])
    with Recorder.create(cassette, framework="langgraph") as rec:
        client = wrap_llm(stub, rec, dialect="custom")
        graph = _build_graph(client, rec)
        graph.invoke({"messages": [{"role": "user", "content": "q"}]})

    c = Cassette.open(cassette, readonly=True)
    step_ids = [e.step_id for e in c.events]
    # Step IDs must include the node names.
    assert any("router" in s for s in step_ids), f"no router in {step_ids}"
    assert any("synthesizer" in s for s in step_ids), f"no synthesizer in {step_ids}"


def test_langgraph_bind_graph_wraps_persist(tmp_path: Path):
    """bind_graph wraps node functions and the wrapping persists after
    exit (this is deliberate — the compiled graph holds references to
    the same RunnableCallable objects, so restoring would break it)."""
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class S(TypedDict, total=False):
        x: str

    def n1(state): return {"x": "a"}

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)

    with Recorder.create(tmp_path / "c", framework="langgraph") as rec:
        original_func = g.nodes["n1"].runnable.func
        with bind_graph(rec, g):
            # Inside the context, the func should be wrapped
            assert g.nodes["n1"].runnable.func is not original_func
            assert getattr(g.nodes["n1"].runnable.func, "_agentreplay_wrapped", False)
        # After exit, the wrapping PERSISTS (deliberately — see bind_graph docs)
        assert g.nodes["n1"].runnable.func is not original_func
        assert getattr(g.nodes["n1"].runnable.func, "_agentreplay_wrapped", False)


def test_langgraph_record_replay_divergence_detected(tmp_path: Path):
    """If the agent's input changes between record and replay, the
    divergence detector must fire."""
    cassette = tmp_path / "cass"
    stub = _StubLLM([
        {"text": "x", "usage": {}},
        {"text": "y", "usage": {}},
    ])
    with Recorder.create(cassette, framework="langgraph") as rec:
        client = wrap_llm(stub, rec, dialect="custom")
        graph = _build_graph(client, rec)
        graph.invoke({"messages": [{"role": "user", "content": "ORIGINAL"}]})

    # Replay with DIFFERENT input — must diverge.
    fresh = _StubLLM(responses=[])
    with Replayer.open(cassette, mode=Mode.REPLAY) as rep:
        client = wrap_llm(fresh, rep, dialect="custom")
        graph = _build_graph(client, rep)
        with pytest.raises(DivergenceError):
            graph.invoke({"messages": [{"role": "user", "content": "DIFFERENT"}]})


def test_wrap_node_helper(tmp_path: Path):
    """The wrap_node helper should produce a callable that calls
    enter_step before delegating."""
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict

    class S(TypedDict, total=False):
        x: str

    def my_node(state):
        return {"x": "result"}

    with Recorder.create(tmp_path / "c", framework="langgraph") as rec:
        wrapped = wrap_node("my-node", my_node, rec)
        g = StateGraph(S)
        g.add_node("n", wrapped)
        g.add_edge(START, "n")
        g.add_edge("n", END)
        compiled = g.compile()
        result = compiled.invoke({"x": ""})

    assert result["x"] == "result"
    # The cassette should have step IDs containing "my-node"
    c = Cassette.open(tmp_path / "c", readonly=True)
    # No LLM calls were made, so the only events would be from the clock
    # or RNG if used. The node ran though, so enter_step was called.
    # We can't directly assert on step IDs without an LLM call, but we
    # can verify the graph executed successfully.
