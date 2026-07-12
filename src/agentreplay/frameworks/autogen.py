"""AutoGen adapter.

AutoGen (Microsoft) agents use a ``ConversableAgent`` class that
holds an ``llm_config`` dict pointing at an OpenAI-compatible endpoint.
The cleanest integration point is the ``openai_client`` that AutoGen
constructs internally — we wrap it before the agent makes any calls.

Usage::

    import autogen
    from agentreplay import Recorder
    from agentreplay.frameworks.autogen import wrap_autogen_client

    with Recorder.create("cassettes/run-001", framework="autogen") as rec:
        config_list = [{"model": "gpt-4o", "api_key": "..."}]
        agent = autogen.ConversableAgent(
            name="analyst",
            llm_config={"config_list": config_list},
        )
        # Wrap the agent's OpenAI client AFTER construction
        wrap_autogen_client(agent, rec)
        result = agent.generate_reply(messages=[{"role": "user", "content": "hi"}])

AutoGen v0.4+ (the "autogen-agentchat" rewrite) uses a different API —
see :func:`wrap_autogen_v4_agent` for that version.
"""
from __future__ import annotations

from typing import Any

from agentreplay.interceptors import RecordingClient


def wrap_autogen_client(agent: Any, session: Any, **kwargs: Any) -> Any:
    """Wrap an AutoGen v0.2 ``ConversableAgent``'s internal OpenAI client.

    AutoGen v0.2 constructs an ``openai.OpenAI`` client internally and
    stores it on the agent. This function wraps that client in place so
    every ``chat.completions.create`` call the agent makes is captured.

    Must be called AFTER the agent is constructed (so the internal
    client exists) and BEFORE the agent is used.
    """
    # AutoGen v0.2 stores the client on `agent.client` (for
    # ConversableAgent with llm_config set) or on
    # `agent.llm_config.client`. The exact attribute varies by version.
    inner_client = getattr(agent, "client", None)
    if inner_client is None:
        # Try the OpenAIWrapper path
        llm_config = getattr(agent, "llm_config", None)
        if llm_config is not None and hasattr(llm_config, "client"):
            inner_client = llm_config.client
    if inner_client is None:
        raise AttributeError(
            "Could not find an OpenAI client on the AutoGen agent. "
            "Ensure the agent was constructed with an llm_config and "
            "that AutoGen v0.2 (not v0.4+) is installed. For AutoGen "
            "v0.4+, use wrap_autogen_v4_agent instead."
        )

    # AutoGen v0.2's client is an OpenAI client — use the OpenAI dialect.
    recording_client = session.wrap_openai(inner_client, **kwargs)

    # Replace the client on the agent. The exact attribute varies, so
    # we set it on every plausible location.
    agent.client = recording_client
    llm_config = getattr(agent, "llm_config", None)
    if llm_config is not None and hasattr(llm_config, "client"):
        llm_config.client = recording_client
    return agent


def wrap_autogen_v4_agent(agent: Any, session: Any, **kwargs: Any) -> Any:
    """Wrap an AutoGen v0.4+ (autogen-agentchat) agent's model client.

    AutoGen v0.4+ uses a different architecture: agents hold a
    ``_model_client`` that implements ``create_stream`` / ``create``.
    This function wraps that client in place.

    Must be called AFTER the agent is constructed and BEFORE it is used.
    """
    inner_client = getattr(agent, "_model_client", None) or getattr(agent, "model_client", None)
    if inner_client is None:
        raise AttributeError(
            "Could not find a model client on the AutoGen v0.4 agent. "
            "Ensure the agent was constructed with a model_client. "
            "For AutoGen v0.2, use wrap_autogen_client instead."
        )

    # AutoGen v0.4's model client is framework-agnostic. We wrap it as
    # a custom client and patch the `create` method.
    recording_client = session.wrap_custom_client(_AutoGenV4Shim(inner_client), **kwargs)

    original_create = inner_client.create

    def patched_create(*args: Any, **create_kwargs: Any) -> Any:
        messages = create_kwargs.get("messages", [])
        response = recording_client.complete(messages=messages, model="autogen-v4")
        return response

    inner_client.create = patched_create
    if not hasattr(inner_client, "_agentreplay_original_create"):
        inner_client._agentreplay_original_create = original_create
    return agent


def restore_autogen_v4_agent(agent: Any) -> None:
    """Restore an AutoGen v0.4 agent's model client to its original state."""
    inner_client = getattr(agent, "_model_client", None) or getattr(agent, "model_client", None)
    if inner_client is not None and hasattr(inner_client, "_agentreplay_original_create"):
        inner_client.create = inner_client._agentreplay_original_create
        del inner_client._agentreplay_original_create


class _AutoGenV4Shim:
    """Adapter that exposes a ``complete`` method for AutoGen v0.4 clients."""

    def __init__(self, inner_client: Any) -> None:
        self.inner_client = inner_client

    def complete(self, *, messages: Any, tools: Any = None, **params: Any) -> Any:
        original = getattr(self.inner_client, "_agentreplay_original_create", None)
        if original is not None:
            return original(messages=messages)
        return self.inner_client.create(messages=messages)
