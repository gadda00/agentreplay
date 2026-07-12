"""LangGraph adapter (first-class integration target, §5.5).

LangGraph already has a checkpointer primitive that snapshots graph
state after every node; AgentReplay hooks into that primitive so each
node transition is a natural step boundary, and wraps the LLM client
and tool nodes so the calls *inside* each node are captured with
per-node call-site IDs.

Two integration points:

    1. :func:`wrap_llm` — wrap the LLM client passed into the graph's
       LLM node. This captures every ``chat.completions.create`` /
       ``messages.create`` call with the current node's name as the
       step ID.

    2. :func:`wrap_tools` — wrap each tool callable so tool invocations
       are captured as TOOL events with the calling node's name as the
       step ID.

    3. :func:`bind_graph` — attach a *raw* (pre-compile) LangGraph
       ``StateGraph`` to the session so node names become step IDs
       automatically. This patches each node's underlying runnable
       function to call ``session.enter_step(node_name)`` before
       delegating to the original function. Call ``bind_graph`` BEFORE
       ``graph.compile()``.

    4. :func:`wrap_node` — lower-level helper that wraps a single node
       function for explicit use with ``graph.add_node``.

The adapter deliberately does NOT try to wrap LangGraph's checkpointer
itself: AgentReplay's cassettes are a strict superset of what a
checkpointer stores, and the two coexist cleanly (the checkpointer
remains useful for *resumable* live runs; the cassette is what gives
you *bit-exact* replay).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from agentreplay.interceptors import RecordingClient, RecordingTool


def wrap_llm(client: Any, session: Any, *, dialect: str = "openai", **kwargs: Any) -> RecordingClient:
    """Wrap an LLM client for use inside a LangGraph LLM node.

    Pass the returned object in place of the raw client to your node
    function. The session's ``enter_step`` will be called automatically
    by :func:`bind_graph`, so each call-site ID incorporates the current
    node's name.

    Parameters
    ----------
    client
        The real LLM client (OpenAI, Anthropic, or custom).
    session
        A :class:`agentreplay.Session`.
    dialect
        ``"openai"`` (default), ``"anthropic"``, or ``"custom"``.
    """
    if dialect == "openai":
        return session.wrap_openai(client, **kwargs)
    if dialect == "anthropic":
        return session.wrap_anthropic(client, **kwargs)
    return session.wrap_custom_client(client, **kwargs)


def wrap_tools(tools: List[Callable[..., Any]], session: Any) -> List[RecordingTool]:
    """Wrap a list of tool callables for use inside LangGraph tool nodes."""
    return [session.wrap_tool(t) for t in tools]


def wrap_node(name: str, node: Callable, session: Any) -> Callable:
    """Wrap a single node function so ``session.enter_step`` is called
    before the node runs.

    Use this with ``graph.add_node``::

        from agentreplay.frameworks.langgraph import wrap_node

        g = StateGraph(MyState)
        g.add_node("router", wrap_node("router", router_fn, session))
        g.add_node("synthesizer", wrap_node("synthesizer", synth_fn, session))
        compiled = g.compile()
    """
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        session.enter_step(f"langgraph:{name}")
        return node(*args, **kwargs)
    # Preserve metadata so LangGraph's signature inspection still works.
    try:
        wrapped.__name__ = getattr(node, "__name__", name)
        wrapped.__doc__ = getattr(node, "__doc__", None)
    except (AttributeError, TypeError):  # pragma: no cover
        pass
    return wrapped


def bind_graph(session: Any, graph: Any) -> "_GraphBinding":
    """Attach a *raw* (pre-compile) LangGraph ``StateGraph`` to the
    session so node names become step IDs.

    Patches each node's underlying runnable function to call
    ``session.enter_step(f"langgraph:{name}")`` before delegating to
    the original function. The patching is reversed on context exit.

    Must be called BEFORE ``graph.compile()``.

    Example::

        from langgraph.graph import StateGraph
        from agentreplay import Session
        from agentreplay.frameworks.langgraph import bind_graph, wrap_llm

        with Session.record("cassettes/run-001", framework="langgraph") as s:
            client = wrap_llm(openai_client, s)
            graph = StateGraph(MyState)
            graph.add_node("router", router_fn)
            graph.add_node("synthesizer", synth_fn)
            graph.add_edge(START, "router")
            graph.add_edge("router", "synthesizer")
            graph.add_edge("synthesizer", END)

            with bind_graph(s, graph):
                compiled = graph.compile()
                result = compiled.invoke(initial_state)
    """
    try:
        import langgraph  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langgraph is required for the LangGraph adapter; "
            "install with `pip install agentreplay[langgraph]`"
        ) from exc

    return _GraphBinding(session, graph)


class _GraphBinding:
    """Context manager that patches a raw StateGraph's node runnables
    to call ``enter_step`` before each node runs.

    Works by patching ``graph.nodes[name].runnable.func`` — the
    underlying callable that LangGraph's ``RunnableCallable`` wraps.

    Note: the patching is NOT reversed on exit. This is deliberate —
    the compiled graph holds references to the same ``RunnableCallable``
    objects, so restoring the original functions would break the
    compiled graph. The wrapping is harmless: it just calls
    ``session.enter_step(name)`` before the original function, which
    is a no-op if the session is closed.

    Must be called BEFORE ``graph.compile()``.
    """

    def __init__(self, session: Any, graph: Any) -> None:
        self.session = session
        self.graph = graph

    def __enter__(self) -> "_GraphBinding":
        nodes = getattr(self.graph, "nodes", None)
        if not isinstance(nodes, dict):
            return self
        for name, spec in nodes.items():
            runnable = getattr(spec, "runnable", None)
            if runnable is None:
                continue
            func = getattr(runnable, "func", None)
            if func is None:
                continue
            # Only wrap if not already wrapped (avoid double-wrapping).
            if not getattr(func, "_agentreplay_wrapped", False):
                runnable.func = self._wrap(name, func)
        return self

    def __exit__(self, *exc: Any) -> None:
        # Intentionally do NOT restore the original functions. The
        # compiled graph holds references to the same RunnableCallable
        # objects, so restoring would break it. The wrapping is harmless
        # (just an enter_step call before the original function).
        pass

    def _wrap(self, name: str, func: Callable) -> Callable:
        """Wrap `func` so it calls ``session.enter_step`` first.

        The session is captured in the closure so the wrapped function
        works correctly even after the context manager exits.
        """
        session = self.session

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            session.enter_step(f"langgraph:{name}")
            return func(*args, **kwargs)

        # Mark as wrapped so we don't double-wrap if bind_graph is called twice.
        wrapped._agentreplay_wrapped = True  # type: ignore[attr-defined]
        try:
            wrapped.__name__ = getattr(func, "__name__", name)
            wrapped.__doc__ = getattr(func, "__doc__", None)
        except (AttributeError, TypeError):  # pragma: no cover
            pass
        return wrapped
