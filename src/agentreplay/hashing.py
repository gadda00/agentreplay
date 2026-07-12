"""Canonicalization & call-site hashing.

The single most important property of the whole library is implemented here:
*the same logical call must produce the same call-site ID*, so that the
replay engine can match "what the agent is asking for right now" against
"what was recorded for that exact ask".

We borrow the pattern from HTTP-mocking libraries (VCR, betamax, Polly.JS)
and deterministic-replay debuggers (Mozilla ``rr``, ``revive``): hash the
*canonicalized* input, not the raw input, so that cosmetic differences
(dict key ordering, trailing whitespace, non-deterministic UUIDs in
request metadata) do not cause spurious divergence.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, Optional

# Keys that are intentionally non-deterministic and must be stripped
# before hashing. Adding to this list is backwards-compatible: existing
# call-site IDs do not change because we only strip keys that were not
# part of the canonical hash before.
_NON_DETERMINISTIC_KEYS: frozenset[str] = frozenset(
    {
        "request_id",        # OpenAI / Anthropic assign a random UUID per request
        "id",                # response.id assigned by the server
        "created",           # OpenAI: epoch seconds the row was created
        "system_fingerprint",  # server-side fingerprint
        "x_request_id",      # common header
        "x_trace_id",
        "x_span_id",
        "session_id",        # client-side bookkeeping
        "seed",              # we capture seed separately; do not hash it
        "user-agent",        # headers drift across client versions
        "User-Agent",
    }
)

# Regex for redacting UUID-shaped values inside string fields so two
# semantically identical requests do not diverge because a client
# generated a fresh idempotency key.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# Regex for ISO-8601 timestamps — replaced with a placeholder so time-
# stamped payloads (e.g. `{"now": "2026-07-12T13:00:00Z"}`) hash
# consistently. Note: the recording layer *also* intercepts the clock
# directly, so timestamps the agent reads via `time.time()` are already
# replayed verbatim; this regex only catches timestamps embedded in
# arbitrary JSON payloads.
_ISO8601_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)


def _redact_string(s: str) -> str:
    s = _UUID_RE.sub("<uuid>", s)
    s = _ISO8601_RE.sub("<iso8601>", s)
    return s


def canonicalize(value: Any) -> Any:
    """Recursively canonicalize a JSON-serialisable value.

    - Dicts are sorted by key; non-deterministic keys are dropped.
    - Strings have UUIDs and ISO-8601 timestamps redacted.
    - Lists and tuples are canonicalized element-wise.
    - Everything else is returned as-is.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, dict):
        return {
            k: canonicalize(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if k not in _NON_DETERMINISTIC_KEYS
        }
    # Fall back to repr for exotic types (Path, Decimal, datetime, ...).
    # The goal is "same object → same canonical form", not "round-trippable".
    return repr(value)


def canonical_json(value: Any) -> str:
    """Canonical JSON encoding: sorted keys, no whitespace, ensure_ascii off."""
    return json.dumps(canonicalize(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def hash_call_site(
    step_id: str,
    request: Any,
    *,
    call_type: Optional[str] = None,
    agent_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute the SHA-256 call-site ID for an intercepted call.

    Per §5.2 of the product proposal, the ID is a hash of
    ``(agent_id, thread_id, node_id, step_index, canonicalized_input)``.
    In practice the agent_id / thread_id are usually encoded *inside* the
    ``step_id`` (which is namespace-prefixed), so we fold them in
    explicitly only when provided.

    The returned digest is a 64-char lowercase hex string.
    """
    parts: list[Any] = [str(step_id)]
    if agent_id is not None:
        parts.append(("agent", agent_id))
    if thread_id is not None:
        parts.append(("thread", thread_id))
    if call_type is not None:
        parts.append(("call_type", call_type))
    if extra:
        for k, v in sorted(extra.items()):
            parts.append((k, v))
    parts.append(canonicalize(request))

    blob = canonical_json(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def hash_payload(value: Any) -> str:
    """SHA-256 hex of a payload's canonical form — used as the blob-store key."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def diff_keys(a: Any, b: Any) -> Iterable[str]:
    """Yield dotted-path keys where ``a`` and ``b`` differ after canonicalization.

    Used by the CLI's structural-diff renderer to highlight *what* changed
    at a divergence point, not just *that* it changed.
    """
    ca = canonicalize(a)
    cb = canonicalize(b)
    yield from _diff_keys(ca, cb, prefix="")


def _diff_keys(a: Any, b: Any, prefix: str) -> Iterable[str]:
    if type(a) is not type(b):
        yield prefix or "<root>"
        return
    if isinstance(a, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys):
            sub = f"{prefix}.{k}" if prefix else str(k)
            if k not in a:
                yield sub
            elif k not in b:
                yield sub
            else:
                yield from _diff_keys(a[k], b[k], sub)
        return
    if isinstance(a, list):
        for i in range(max(len(a), len(b))):
            sub = f"{prefix}[{i}]"
            if i >= len(a) or i >= len(b):
                yield sub
            else:
                yield from _diff_keys(a[i], b[i], sub)
        return
    if a != b:
        yield prefix or "<root>"
