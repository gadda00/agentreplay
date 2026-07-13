"""HTTP & tool interceptors.

These wrap (a) the underlying HTTP transport so even tool logic that was
not explicitly instrumented is still captured at the network boundary,
and (b) individual tool callables so framework-registered tools can be
intercepted without touching the framework itself.

The HTTP interceptor supports ``httpx.Client`` (used by both OpenAI and
Anthropic SDKs under the hood) and ``requests.Session`` — the two
libraries the frameworks we target actually use. ``urllib`` is supported
through a top-level ``urlopen`` patch because some legacy tools still
use it.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from agentreplay.constants import CallType, Mode
from agentreplay.cassette import Cassette
from agentreplay.errors import DivergenceError
from agentreplay.hashing import hash_call_site


class _BaseCallInterceptor:
    """Common machinery for HTTP/tool interceptors."""

    def __init__(
        self,
        cassette: Cassette,
        *,
        mode: Mode = Mode.RECORD,
        step_id_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        self.cassette = cassette
        self.mode = mode
        self._step_id_provider = step_id_provider or (lambda: "tool")
        self._counter = 0

    def _next_step_id(self, op: str) -> str:
        sid = f"{self._step_id_provider()}:{op}:{self._counter}"
        self._counter += 1
        return sid

    def _lookup_or_raise(self, sid: str, call_id: str, request: Any, call_type: CallType) -> Any:
        event = self.cassette.lookup_call(call_id)
        if event is not None:
            return self.cassette.resolve_response(event)
        if self.mode == Mode.REPLAY:
            raise DivergenceError(
                step_id=sid,
                call_type=call_type.value,
                expected_call_id=None,
                actual_call_id=call_id,
                actual_request=request,
            )
        return None  # HYBRID: caller decides what to do


class RecordingTool(_BaseCallInterceptor):
    """Wrap a single tool callable.

    A *tool* in this context is any function the agent can call: a
    LangGraph ``@tool``-decorated function, a CrewAI tool, an OpenAI
    function-calling tool, or a raw Python callable. The wrapper
    captures the positional/keyword arguments and the return value
    (or exception) and keys them by the canonicalized arg tuple.

    The wrapped function is invoked through ``__call__`` so the wrapper
    is a drop-in replacement for the original callable.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str],
        cassette: Cassette,
        *,
        mode: Mode = Mode.RECORD,
        step_id_provider: Optional[Callable[[], str]] = None,
    ) -> None:
        super().__init__(cassette, mode=mode, step_id_provider=step_id_provider)
        self.func = func
        self.name = name or getattr(func, "__name__", "tool")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        sid = self._next_step_id(f"tool:{self.name}")
        request: Dict[str, Any] = {"name": self.name, "args": list(args), "kwargs": dict(kwargs)}
        call_id = hash_call_site(sid, request, call_type=CallType.TOOL.value)

        if self.mode in (Mode.REPLAY, Mode.HYBRID):
            cached = self._lookup_or_raise(sid, call_id, request, CallType.TOOL)
            if cached is not None:
                return cached["value"]
            # HYBRID fallthrough:
            return self._invoke_and_record(sid, call_id, request, args, kwargs)

        # LIVE / RECORD
        return self._invoke_and_record(sid, call_id, request, args, kwargs)

    def _invoke_and_record(
        self,
        sid: str,
        call_id: str,
        request: Dict[str, Any],
        args: tuple,
        kwargs: Dict[str, Any],
    ) -> Any:
        started = time.time()
        try:
            value = self.func(*args, **kwargs)
            duration_ms = (time.time() - started) * 1000.0
            response: Dict[str, Any] = {"value": value, "error": None}
        except Exception as exc:  # noqa: BLE001 - we record and re-raise
            duration_ms = (time.time() - started) * 1000.0
            response = {"value": None, "error": f"{type(exc).__name__}: {exc}"}
            if self.mode == Mode.RECORD:
                self.cassette.write_event(
                    step_id=sid,
                    call_type=CallType.TOOL,
                    call_id=call_id,
                    request=request,
                    response=response,
                    started_at=started,
                    duration_ms=duration_ms,
                    metadata={"tool": self.name, "raised": True},
                )
            raise
        if self.mode == Mode.RECORD:
            self.cassette.write_event(
                step_id=sid,
                call_type=CallType.TOOL,
                call_id=call_id,
                request=request,
                response=response,
                started_at=started,
                duration_ms=duration_ms,
                metadata={"tool": self.name},
            )
        return value


class RecordingHTTP(_BaseCallInterceptor):
    """Wrap an ``httpx.Client`` (or ``requests.Session``) transport.

    The interceptor is *transparent*: the wrapped client behaves
    exactly like the original, except each request/response pair is
    recorded to the cassette. Replay returns the recorded response
    without touching the network.
    """

    def __init__(
        self,
        real_client: Any,
        cassette: Cassette,
        *,
        mode: Mode = Mode.RECORD,
        step_id_provider: Optional[Callable[[], str]] = None,
        dialect: str = "httpx",  # "httpx" | "requests"
    ) -> None:
        super().__init__(cassette, mode=mode, step_id_provider=step_id_provider)
        self.real_client = real_client
        self.dialect = dialect

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        sid = self._next_step_id("http")
        request: Dict[str, Any] = {
            "method": method.upper(),
            "url": url,
            "headers": dict(kwargs.get("headers") or {}),
            "params": kwargs.get("params"),
            "body": kwargs.get("content") or kwargs.get("json") or kwargs.get("data"),
        }
        call_id = hash_call_site(sid, request, call_type=CallType.HTTP.value)

        if self.mode in (Mode.REPLAY, Mode.HYBRID):
            cached = self._lookup_or_raise(sid, call_id, request, CallType.HTTP)
            if cached is not None:
                return _ReplayResponse(cached)
            # HYBRID fallthrough:

        started = time.time()
        try:
            if self.dialect == "httpx":
                raw = self.real_client.request(method, url, **kwargs)
                response = {
                    "status": raw.status_code,
                    "headers": dict(raw.headers),
                    "body": raw.text,
                    "url": str(raw.url) if hasattr(raw, "url") else url,
                    "encoding": getattr(raw, "encoding", "utf-8") or "utf-8",
                    "elapsed": getattr(raw, "elapsed", 0.0),
                }
            else:  # requests
                raw = self.real_client.request(method, url, **kwargs)
                response = {
                    "status": raw.status_code,
                    "headers": dict(raw.headers),
                    "body": raw.text,
                    "url": str(raw.url) if hasattr(raw, "url") else url,
                    "encoding": getattr(raw, "encoding", "utf-8") or "utf-8",
                    "elapsed": getattr(raw, "elapsed", 0.0),
                    "reason": getattr(raw, "reason", ""),
                }
            duration_ms = (time.time() - started) * 1000.0
        except Exception as exc:
            # Record the exception as the response so replay can reproduce it.
            duration_ms = (time.time() - started) * 1000.0
            response = {
                "status": 0,
                "headers": {},
                "body": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if self.mode == Mode.RECORD:
                self.cassette.write_event(
                    step_id=sid,
                    call_type=CallType.HTTP,
                    call_id=call_id,
                    request=request,
                    response=response,
                    started_at=started,
                    duration_ms=duration_ms,
                    metadata={"dialect": self.dialect, "raised": True, "error": str(exc)},
                )
            raise

        if self.mode == Mode.RECORD:
            self.cassette.write_event(
                step_id=sid,
                call_type=CallType.HTTP,
                call_id=call_id,
                request=request,
                response=response,
                started_at=started,
                duration_ms=duration_ms,
                metadata={"dialect": self.dialect},
            )
        return raw

    # Convenience methods matching the httpx/requests public surface.
    def get(self, url: str, **kwargs: Any) -> Any:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Any:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> Any:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> Any:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> Any:
        return self.request("DELETE", url, **kwargs)


class _ReplayResponse:
    """Mimics enough of an ``httpx.Response`` / ``requests.Response``
    for the wrapped client's caller to keep working in REPLAY mode.

    Implements the common subset of attributes/methods from both
    httpx.Response and requests.Response so agent code that uses either
    library's API surface works transparently during replay.
    """

    def __init__(self, cached: Dict[str, Any]) -> None:
        self.status_code: int = cached.get("status", 200)
        self.headers: Dict[str, str] = dict(cached.get("headers", {}))
        self.text: str = cached.get("body", "") or ""
        self._json: Any = None
        self._content: Optional[bytes] = None
        # httpx/requests compatibility attrs
        self.url: str = cached.get("url", "")
        self.encoding: str = cached.get("encoding", "utf-8")
        self.elapsed: float = cached.get("elapsed", 0.0)
        self.reason: str = cached.get("reason", "")

    # --- Content accessors ---

    @property
    def content(self) -> bytes:
        """Raw response body as bytes (httpx + requests compatible)."""
        if self._content is None:
            self._content = self.text.encode("utf-8") if self.text else b""
        return self._content

    def json(self, **kwargs: Any) -> Any:
        """Parse the response body as JSON (httpx + requests compatible)."""
        import json as _json

        if self._json is None:
            try:
                self._json = _json.loads(self.text or "null")
            except _json.JSONDecodeError:
                self._json = None
        return self._json

    # --- Status helpers (httpx + requests compatible) ---

    @property
    def ok(self) -> bool:
        """True if status code is < 400 (requests) / is_success (httpx)."""
        return self.status_code < 400

    @property
    def is_success(self) -> bool:
        """httpx: True if 200 <= status < 300."""
        return 200 <= self.status_code < 300

    @property
    def is_redirect(self) -> bool:
        """httpx: True if 300 <= status < 400."""
        return 300 <= self.status_code < 400

    @property
    def is_client_error(self) -> bool:
        """httpx: True if 400 <= status < 500."""
        return 400 <= self.status_code < 500

    @property
    def is_server_error(self) -> bool:
        """httpx: True if status >= 500."""
        return self.status_code >= 500

    @property
    def is_error(self) -> bool:
        """httpx: True if status >= 400."""
        return self.status_code >= 400

    def raise_for_status(self) -> None:
        """Raise an exception if status >= 400 (httpx + requests compatible)."""
        if self.status_code >= 400:
            raise RuntimeError(f"replayed HTTP {self.status_code}: {self.reason or 'Error'}")

    # --- Cookies (basic compatibility) ---

    @property
    def cookies(self) -> Dict[str, str]:
        """Parse Set-Cookie headers into a simple dict. Not a full CookieJar."""
        result: Dict[str, str] = {}
        for key, value in self.headers.items():
            if key.lower() == "set-cookie":
                # Very basic parsing — doesn't handle all cookie attributes
                for cookie in value.split(";"):
                    if "=" in cookie:
                        name, _, val = cookie.strip().partition("=")
                        result[name] = val
        return result

    # --- Iteration (for streaming responses, though we replay as complete) ---

    def iter_bytes(self, chunk_size: int = 1024) -> Any:
        """httpx: iterate over bytes in chunks."""
        yield self.content

    def iter_content(self, chunk_size: int = 1024, decode_unicode: bool = False) -> Any:
        """requests: iterate over content in chunks."""
        if decode_unicode:
            yield self.text
        else:
            yield self.content

    def close(self) -> None:
        """No-op — compatibility with httpx/requests stream close()."""
        pass

    def __repr__(self) -> str:
        return f"<_ReplayResponse [{self.status_code}]>"
