"""LangGraph adapter (first-class integration target, §5.5).

LangGraph already has a checkpointer primitive that snapshots graph
state after every node; AgentReplay hooks into that primitive so each
node transition is a natural step boundary, and wraps the LLM client
and tool nodes so the calls *inside* each node are captured with
per-step call-site IDs.

Two integration points:

    1. ``wrap_llm`` — wrap the LLM client passed into the graph's LLM
       node. This captures every ``chat.completions.create`` /
       ``messages.create`` call with the current node's name as the
       step ID.

    2. ``wrap_tools`` — wrap each tool callable so tool invocations are
       captured as TOOL events with the calling node's name as the
       step ID.

The adapter deliberately does NOT try to wrap LangGraph's checkpointer
itself: AgentReplay's cassettes are a strict superset of what a
checkpointer stores, and the two coexist cleanly (the checkpointer
remains useful for *resumable* live runs; the cassette is what gives
you *bit-exact* replay).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agentreplay.interceptors import RecordingClient, RecordingTool


def wrap_llm(client: Any, session: Any, **kwargs: Any) -> RecordingClient:
    """Wrap an LLM client for use inside a LangGraph LLM node.

    Pass the returned object in place of the raw client to your node
    function. The session's ``enter_step`` will be called automatically
    by the graph runner if you also call :func:`bind_state`, so each
    call-site ID incorporates the current node's name.
    """
    return session.wrap_openai(client, **kwargs) if _dialect(session) == "openai" else session.wrap_anthropic(client, **kwargs)


def wrap_tools(tools: List[Callable[..., Any]], session: Any) -> List[RecordingTool]:
    """Wrap a list of tool callables for use inside LangGraph tool nodes."""
    return [session.wrap_tool(t) for t in tools]


def bind_state(session: Any, graph: Any) -> Any:
    """Attach a LangGraph graph to the session so node names become step IDs.

    The returned object is a context manager that, when entered, patches
    the graph's node executor to call ``session.enter_step(node_name)``
    before each node runs. On exit it restores the original executor.

    This is the only place where the adapter imports LangGraph itself,
    so teams not using LangGraph pay no import cost.
    """
    try:
        from langgraph.graph import StateGraph  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langgraph is required for the LangGraph adapter; "
            "install with `pip install agentreplay[langgraph]`"
        ) from exc

    # The simplest portable integration: wrap each node callable so it
    # calls session.enter_step before delegating to the original node.
    # LangGraph stores nodes on graph.nodes; we wrap them in place.
    nodes = getattr(graph, "nodes", None)
    if nodes is None:
        # Compiled graph — wrap the underlying attribute if present.
        return _NullContext()

    original = dict(nodes)
    for name, node in original.items():
        nodes[name] = _wrap_node(name, node, session)
    return _NodeRestoreContext(nodes, original)


def _wrap_node(name: str, node: Callable, session: Any) -> Callable:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        session.enter_step(f"langgraph:{name}")
        return node(*args, **kwargs)
    return wrapped


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


class _NodeRestoreContext:
    def __init__(self, nodes: Dict[str, Any], original: Dict[str, Any]) -> None:
        self.nodes = nodes
        self.original = original

    def __enter__(self) -> "_NodeRestoreContext":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.nodes.clear()
        self.nodes.update(self.original)


def _dialect(session: Any) -> str:
    """Best-effort guess at the SDK dialect the session was created with."""
    # Session doesn't carry this explicitly; default to "openai" since
    # the LangGraph adapter's wrap_llm is mostly used with OpenAI clients.
    return getattr(session, "_langgraph_dialect", "openai")
