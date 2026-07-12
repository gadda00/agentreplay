"""CrewAI adapter.

CrewAI agents use an ``LLM`` wrapper class that internally calls the
OpenAI or Anthropic SDK. The cleanest integration point is the
``crewai.LLM`` class itself — we wrap its ``call`` method so every
model invocation is captured.

Usage::

    from crewai import Agent, Task, Crew
    from agentreplay import Recorder
    from agentreplay.frameworks.crewai import wrap_crewai_llm

    with Recorder.create("cassettes/run-001", framework="crewai") as rec:
        llm = wrap_crewai_llm(crewai.LLM(model="gpt-4o"), rec)
        agent = Agent(role="Analyst", llm=llm, ...)
        crew = Crew(agents=[agent], tasks=[...])
        result = crew.kickoff()

If you construct your CrewAI ``LLM`` objects inside the recorder
context manager and use ``wrap_crewai_llm``, every model call the
crew makes will be captured with per-task step IDs.
"""
from __future__ import annotations

from typing import Any

from agentreplay.interceptors import RecordingClient


def wrap_crewai_llm(llm: Any, session: Any, **kwargs: Any) -> Any:
    """Wrap a CrewAI ``LLM`` object for recording/replay.

    CrewAI's ``LLM`` class exposes a ``call()`` method (and ``aincall()``
    for async) that internally dispatches to the OpenAI or Anthropic SDK.
    Rather than wrapping the underlying SDK (which CrewAI may swap), we
    wrap the ``LLM`` object's ``call`` method directly so every model
    invocation goes through the AgentReplay recording layer.

    The wrapped object is the same ``llm`` instance — we patch its
    ``call`` method in place so CrewAI's internal references still work.
    """
    # Wrap the underlying client as a custom RecordingClient. CrewAI's
    # LLM.call() signature varies across versions, so we use the
    # "custom" dialect which expects a .complete() method.
    recording_client = session.wrap_custom_client(_CrewAIShim(llm), **kwargs)

    # Patch the LLM's call method to delegate through the recording client.
    original_call = llm.call

    def patched_call(prompt: str, *args: Any, **call_kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = recording_client.complete(messages=messages, model=getattr(llm, "model", "unknown"))
        # CrewAI's LLM.call returns a string; the recording client returns
        # whatever the underlying client returned. Normalize to string.
        if isinstance(response, dict):
            return response.get("text", str(response))
        return str(response)

    llm.call = patched_call
    # Preserve the original for restoration if needed.
    if not hasattr(llm, "_agentreplay_original_call"):
        llm._agentreplay_original_call = original_call
    return llm


def restore_crewai_llm(llm: Any) -> None:
    """Restore a CrewAI ``LLM`` to its original (un-wrapped) state."""
    if hasattr(llm, "_agentreplay_original_call"):
        llm.call = llm._agentreplay_original_call
        del llm._agentreplay_original_call


class _CrewAIShim:
    """A thin adapter that wraps a CrewAI LLM and exposes the ``complete``
    method signature that ``RecordingClient`` expects."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Any:
        # Extract the last user message and call the LLM.
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
        # Call the ORIGINAL llm.call (not the patched one) to avoid infinite recursion.
        original = getattr(self.llm, "_agentreplay_original_call", None)
        if original is not None:
            text = original(last_user)
        else:
            text = self.llm.call(last_user)
        return {"text": text, "usage": {}}
