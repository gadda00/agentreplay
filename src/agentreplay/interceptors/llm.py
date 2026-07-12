"""LLM client interceptor.

Wraps an LLM client (OpenAI, Anthropic, or any object exposing a
``complete``/``chat.completions.create``/``messages.create`` method)
and intercepts every call. In RECORD mode the request/response pair is
written to the cassette; in REPLAY mode the call is served from the
cassette and the real client is never touched.

The wrapper is intentionally agnostic about the underlying client shape
so it can support the OpenAI SDK, the Anthropic SDK, LangChain's
``BaseChatModel``, or a custom agent's raw client object — see
§5.5 ("Framework integration") of the product proposal.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from agentreplay.constants import CallType, Mode
from agentreplay.cassette import Cassette
from agentreplay.errors import DivergenceError
from agentreplay.hashing import hash_call_site


class RecordingClient:
    """Wrap an LLM client object and intercept its completion calls.

    Parameters
    ----------
    real_client
        The underlying client. Must expose one of:
            - ``chat.completions.create(**kwargs)`` (OpenAI shape)
            - ``messages.create(**kwargs)``          (Anthropic shape)
            - ``complete(messages=..., tools=..., **params)`` (custom shape)
    cassette
        The cassette to record into / replay from.
    mode
        One of :class:`Mode`. Defaults to RECORD.
    call_type
        The SDK dialect to invoke. ``"openai"``, ``"anthropic"`` or
        ``"custom"``.
    step_id_provider
        Optional callable returning the current step ID. Framework
        adapters set this so each call gets a unique, deterministic
        step ID even when the agent invokes the model multiple times
        in a single graph node.
    """

    def __init__(
        self,
        real_client: Any,
        cassette: Cassette,
        *,
        mode: Mode = Mode.RECORD,
        call_type: str = "custom",
        step_id_provider: Optional[Callable[[], str]] = None,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        self.real_client = real_client
        self.cassette = cassette
        self.mode = mode
        self.call_type = call_type
        self._step_id_provider = step_id_provider or (lambda: "llm")
        self.agent_id = agent_id
        self.thread_id = thread_id
        # Counter within the current step — distinguishes multiple model
        # calls from the same node without forcing the framework adapter
        # to assign unique step IDs itself.
        self._call_counter = 0

    # ------------------------------------------------------------------ #
    # Public surface: a single canonical entry point.
    # ------------------------------------------------------------------ #
    def complete(
        self,
        *,
        messages: Any,
        tools: Any = None,
        step_id: Optional[str] = None,
        **params: Any,
    ) -> Any:
        """Intercept a single LLM completion call.

        ``messages`` and ``tools`` are the canonical request payload —
        whatever the agent passes in. ``**params`` carries model,
        temperature, max_tokens, etc. The return value is whatever the
        real client returned (in RECORD/LIVE mode) or what was recorded
        (in REPLAY mode).
        """
        sid = step_id or f"{self._step_id_provider()}:{self._call_counter}"
        self._call_counter += 1

        request: Dict[str, Any] = {"messages": messages, "tools": tools, **params}
        call_id = hash_call_site(
            sid,
            request,
            call_type=CallType.LLM.value,
            agent_id=self.agent_id,
            thread_id=self.thread_id,
        )

        # ----- REPLAY path -------------------------------------------------
        if self.mode in (Mode.REPLAY, Mode.HYBRID):
            event = self.cassette.lookup_call(call_id)
            if event is not None:
                return self.cassette.resolve_response(event)
            # No match — diverged.
            if self.mode == Mode.REPLAY:
                # Raise so the Replayer can surface a structured diff.
                raise DivergenceError(
                    step_id=sid,
                    call_type=CallType.LLM.value,
                    expected_call_id=None,
                    actual_call_id=call_id,
                    actual_request=request,
                )
            # HYBRID: fall through to live call.
            response = self._invoke_real(messages=messages, tools=tools, **params)
            # Do NOT record the live response into this cassette — the
            # cassette is the *reference* run; hybrid calls are exploratory.
            return response

        # ----- LIVE / RECORD path -----------------------------------------
        started = time.time()
        response = self._invoke_real(messages=messages, tools=tools, **params)
        duration_ms = (time.time() - started) * 1000.0

        if self.mode == Mode.RECORD:
            self.cassette.write_event(
                step_id=sid,
                call_type=CallType.LLM,
                call_id=call_id,
                request=request,
                response=response,
                started_at=started,
                duration_ms=duration_ms,
                metadata={"call_type": self.call_type, "model": params.get("model", "")},
            )
        return response

    # ------------------------------------------------------------------ #
    # SDK-dialect dispatch
    # ------------------------------------------------------------------ #
    def _invoke_real(self, *, messages: Any, tools: Any = None, **params: Any) -> Any:
        if self.call_type == "openai":
            kwargs: Dict[str, Any] = {"messages": messages, **params}
            if tools is not None:
                kwargs["tools"] = tools
            return self.real_client.chat.completions.create(**kwargs)
        if self.call_type == "anthropic":
            kwargs = {"messages": messages, **params}
            if tools is not None:
                kwargs["tools"] = tools
            return self.real_client.messages.create(**kwargs)
        if self.call_type == "custom":
            # Custom clients are expected to expose ``complete(**kwargs)``.
            return self.real_client.complete(messages=messages, tools=tools, **params)
        raise ValueError(f"unknown call_type {self.call_type!r}")

    # ------------------------------------------------------------------ #
    # SDK-specific convenience methods so the wrapper can be used as a
    # drop-in replacement for the raw client object.
    # ------------------------------------------------------------------ #
    @property
    def chat(self) -> "_OpenAIChatShim":
        return _OpenAIChatShim(self)

    @property
    def messages(self) -> "_AnthropicMessagesShim":
        return _AnthropicMessagesShim(self)


class _OpenAIChatShim:
    """Mimics ``openai.OpenAI().chat`` so the wrapper can be passed in
    place of the raw client to code that expects the OpenAI shape."""

    def __init__(self, parent: RecordingClient) -> None:
        self.parent = parent
        self.completions = _OpenAICompletionsShim(parent)

    # Pass-through for any attribute the SDK might expose that we don't
    # explicitly model (e.g. ``.create`` for chat.completions.create on
    # very old SDKs). Recording only happens via .complete above.


class _OpenAICompletionsShim:
    def __init__(self, parent: RecordingClient) -> None:
        self.parent = parent

    def create(self, **kwargs: Any) -> Any:
        # Make sure the parent uses the openai dialect for this call.
        original = self.parent.call_type
        self.parent.call_type = "openai"
        try:
            messages = kwargs.pop("messages")
            tools = kwargs.pop("tools", None)
            return self.parent.complete(messages=messages, tools=tools, **kwargs)
        finally:
            self.parent.call_type = original


class _AnthropicMessagesShim:
    """Mimics ``anthropic.Anthropic().messages`` for drop-in use."""

    def __init__(self, parent: RecordingClient) -> None:
        self.parent = parent

    def create(self, **kwargs: Any) -> Any:
        original = self.parent.call_type
        self.parent.call_type = "anthropic"
        try:
            messages = kwargs.pop("messages")
            tools = kwargs.pop("tools", None)
            return self.parent.complete(messages=messages, tools=tools, **kwargs)
        finally:
            self.parent.call_type = original
