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
from agentreplay.interceptors.streaming import (
    RecordingStream,
    ReplayStream,
    is_streamed_response,
    make_streamed_response,
)
from agentreplay.logging import get_logger

logger = get_logger(__name__)


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

        is_stream = params.get("stream", False)
        logger.debug("complete: sid=%s call_id=%s mode=%s stream=%s", sid, call_id[:8], self.mode, is_stream)

        # ----- REPLAY path -------------------------------------------------
        if self.mode in (Mode.REPLAY, Mode.HYBRID):
            event = self.cassette.lookup_call(call_id)
            if event is not None:
                response = self.cassette.resolve_response(event)
                # If the recorded response was streamed, wrap it in a
                # ReplayStream so the agent code can iterate over chunks.
                if is_stream and is_streamed_response(response):
                    logger.debug("complete: replaying %d streamed chunks", len(response.get("chunks", [])))
                    return ReplayStream(response["chunks"])
                return response
            # No match — diverged.
            if self.mode == Mode.REPLAY:
                raise DivergenceError(
                    step_id=sid,
                    call_type=CallType.LLM.value,
                    expected_call_id=None,
                    actual_call_id=call_id,
                    actual_request=request,
                )
            # HYBRID: fall through to live call.
            response = self._invoke_real(messages=messages, tools=tools, **params)
            return response

        # ----- LIVE / RECORD path -----------------------------------------
        started = time.time()

        if is_stream:
            # Streaming: wrap the real stream in a RecordingStream that
            # captures chunks as they're consumed, then write them to the
            # cassette as a single event when the stream is exhausted.
            real_stream = self._invoke_real(messages=messages, tools=tools, **params)

            def _on_complete(chunks: list) -> None:
                duration_ms = (time.time() - started) * 1000.0
                if self.mode == Mode.RECORD:
                    streamed_response = make_streamed_response(chunks)
                    self.cassette.write_event(
                        step_id=sid,
                        call_type=CallType.LLM,
                        call_id=call_id,
                        request=request,
                        response=streamed_response,
                        started_at=started,
                        duration_ms=duration_ms,
                        metadata={
                            "call_type": self.call_type,
                            "model": params.get("model", ""),
                            "streamed": True,
                            "num_chunks": len(chunks),
                        },
                    )
                    logger.debug("complete: recorded %d streamed chunks", len(chunks))

            return RecordingStream(real_stream, on_complete=_on_complete)

        # Non-streaming: call and record as before.
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
            logger.debug("complete: recorded response (%.1fms)", duration_ms)
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
    # Async support — for AsyncOpenAI / AsyncAnthropic clients.
    # ------------------------------------------------------------------ #
    async def acomplete(
        self,
        *,
        messages: Any,
        tools: Any = None,
        step_id: Optional[str] = None,
        **params: Any,
    ) -> Any:
        """Async version of :meth:`complete`.

        If the underlying real client supports async (``AsyncOpenAI``,
        ``AsyncAnthropic``, or a custom async client with an
        ``acomplete`` method), the call is awaited. Otherwise the
        sync ``complete`` is called in a thread via ``asyncio.to_thread``.

        For streaming (``stream=True``), returns a :class:`RecordingStream`
        (RECORD) or :class:`ReplayStream` (REPLAY) that can be iterated
        asynchronously. The caller should use ``async for chunk in stream``.
        """
        import asyncio
        import inspect

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

        is_stream = params.get("stream", False)
        logger.debug("acomplete: sid=%s call_id=%s mode=%s stream=%s", sid, call_id[:8], self.mode, is_stream)

        # ----- REPLAY path -------------------------------------------------
        if self.mode in (Mode.REPLAY, Mode.HYBRID):
            event = self.cassette.lookup_call(call_id)
            if event is not None:
                response = self.cassette.resolve_response(event)
                if is_stream and is_streamed_response(response):
                    return ReplayStream(response["chunks"])
                return response
            if self.mode == Mode.REPLAY:
                raise DivergenceError(
                    step_id=sid,
                    call_type=CallType.LLM.value,
                    expected_call_id=None,
                    actual_call_id=call_id,
                    actual_request=request,
                )
            # HYBRID fallthrough — call the real async client.
            real_acomplete = getattr(self.real_client, "acomplete", None)
            if real_acomplete is not None and inspect.iscoroutinefunction(real_acomplete):
                return await real_acomplete(messages=messages, tools=tools, **params)
            return await asyncio.to_thread(self._invoke_real, messages=messages, tools=tools, **params)

        # ----- LIVE / RECORD path -----------------------------------------
        started = time.time()

        if is_stream:
            # Streaming async — get the real async stream and wrap it.
            real_acomplete = getattr(self.real_client, "acomplete", None)
            if real_acomplete is not None and inspect.iscoroutinefunction(real_acomplete):
                real_stream = await real_acomplete(messages=messages, tools=tools, **params)
            else:
                real_stream = await asyncio.to_thread(self._invoke_real, messages=messages, tools=tools, **params)

            def _on_complete(chunks: list) -> None:
                duration_ms = (time.time() - started) * 1000.0
                if self.mode == Mode.RECORD:
                    streamed_response = make_streamed_response(chunks)
                    self.cassette.write_event(
                        step_id=sid,
                        call_type=CallType.LLM,
                        call_id=call_id,
                        request=request,
                        response=streamed_response,
                        started_at=started,
                        duration_ms=duration_ms,
                        metadata={
                            "call_type": self.call_type,
                            "model": params.get("model", ""),
                            "streamed": True,
                            "async": True,
                            "num_chunks": len(chunks),
                        },
                    )

            return RecordingStream(real_stream, on_complete=_on_complete)

        # Non-streaming async.
        real_acomplete = getattr(self.real_client, "acomplete", None)
        if real_acomplete is not None and inspect.iscoroutinefunction(real_acomplete):
            response = await real_acomplete(messages=messages, tools=tools, **params)
        else:
            response = await asyncio.to_thread(self._invoke_real, messages=messages, tools=tools, **params)

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
                metadata={
                    "call_type": self.call_type,
                    "model": params.get("model", ""),
                    "async": True,
                },
            )
            logger.debug("acomplete: recorded response (%.1fms)", duration_ms)
        return response

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
