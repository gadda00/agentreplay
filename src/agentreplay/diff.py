"""Structural diff between cassettes and individual payloads.

The point of a structural diff (as opposed to a textual one) is that
LLM payloads are nested JSON with arbitrary key ordering and frequent
repeated segments (system prompts, tool schemas). A textual diff would
drown the developer in noise; a structural diff highlights only the
paths where the two payloads actually diverge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agentreplay.cassette import Cassette
from agentreplay.hashing import canonicalize, diff_keys


@dataclass
class FieldDiff:
    """A single field-level divergence."""

    path: str
    recorded: Any
    actual: Any


@dataclass
class StepDiff:
    """Per-step divergence summary."""

    seq: int
    step_id: str
    call_type: str
    recorded_call_id: Optional[str]
    actual_call_id: Optional[str]
    field_diffs: List[FieldDiff] = field(default_factory=list)

    @property
    def kind(self) -> str:
        if self.recorded_call_id is None and self.actual_call_id is not None:
            return "extra_actual"
        if self.recorded_call_id is not None and self.actual_call_id is None:
            return "extra_recorded"
        if self.recorded_call_id != self.actual_call_id:
            return "diverged"
        # Same call_id, but content (request or response) may still differ.
        if self.field_diffs:
            return "diverged"
        return "matching"


@dataclass
class Diff:
    """Full diff between two cassettes (or between a cassette and a live run)."""

    source: str
    target: str
    steps: List[StepDiff] = field(default_factory=list)

    @property
    def has_divergence(self) -> bool:
        return any(s.kind != "matching" for s in self.steps)

    @property
    def first_divergence(self) -> Optional[StepDiff]:
        for s in self.steps:
            if s.kind != "matching":
                return s
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "total_steps": len(self.steps),
            "matching": sum(1 for s in self.steps if s.kind == "matching"),
            "diverged": sum(1 for s in self.steps if s.kind == "diverged"),
            "extra_actual": sum(1 for s in self.steps if s.kind == "extra_actual"),
            "extra_recorded": sum(1 for s in self.steps if s.kind == "extra_recorded"),
            "has_divergence": self.has_divergence,
        }


def diff_payloads(recorded: Any, actual: Any) -> List[FieldDiff]:
    """Return per-field differences between two payloads (canonicalized)."""
    diffs: List[FieldDiff] = []
    for path in diff_keys(recorded, actual):
        r = _extract(recorded, path)
        a = _extract(actual, path)
        diffs.append(FieldDiff(path=path, recorded=r, actual=a))
    return diffs


def diff_events(
    recorded_req: Any,
    actual_req: Any,
    recorded_resp: Any,
    actual_resp: Any,
    *,
    seq: int,
    step_id: str,
    call_type: str,
    recorded_call_id: Optional[str],
    actual_call_id: Optional[str],
) -> StepDiff:
    """Compare two events at the same sequence position.

    Compares BOTH request and response payloads so that a divergent
    *response* (same call_id, different content) is flagged in addition
    to a divergent *request* (different call_id).
    """
    fd: List[FieldDiff] = []
    if recorded_req is not None and actual_req is not None:
        fd.extend(diff_payloads(canonicalize(recorded_req), canonicalize(actual_req)))
    if recorded_resp is not None and actual_resp is not None:
        # Prefix response diffs with "response." so the developer can tell
        # whether the divergence was in what the agent asked or what it got.
        for d in diff_payloads(canonicalize(recorded_resp), canonicalize(actual_resp)):
            fd.append(FieldDiff(path=f"response.{d.path}", recorded=d.recorded, actual=d.actual))
    return StepDiff(
        seq=seq,
        step_id=step_id,
        call_type=call_type,
        recorded_call_id=recorded_call_id,
        actual_call_id=actual_call_id,
        field_diffs=fd,
    )


def diff_structural(source: Cassette, target: Cassette) -> Diff:
    """Compare two cassettes step-by-step.

    The diff is *positional*: events are matched by ``seq`` rather than by
    call-site ID, because the whole point of a counterfactual mutation
    is to let the call-site IDs diverge while keeping the run comparable.
    The ``recorded_call_id`` / ``actual_call_id`` fields preserve the
    call-site IDs so the developer can see *where* they diverged.
    """
    diff = Diff(source=source.meta.id, target=target.meta.id)
    src_records = list(source.iter_records())
    tgt_records = list(target.iter_records())
    n = max(len(src_records), len(tgt_records))
    for i in range(n):
        r = src_records[i] if i < len(src_records) else None
        t = tgt_records[i] if i < len(tgt_records) else None
        if r is None:
            diff.steps.append(
                StepDiff(
                    seq=i,
                    step_id=t.event.step_id,  # type: ignore[union-attr]
                    call_type=t.event.call_type,  # type: ignore[union-attr]
                    recorded_call_id=None,
                    actual_call_id=t.event.call_id,  # type: ignore[union-attr]
                    field_diffs=[],
                )
            )
            continue
        if t is None:
            diff.steps.append(
                StepDiff(
                    seq=i,
                    step_id=r.event.step_id,
                    call_type=r.event.call_type,
                    recorded_call_id=r.event.call_id,
                    actual_call_id=None,
                    field_diffs=[],
                )
            )
            continue
        diff.steps.append(
            diff_events(
                r.request,
                t.request,
                r.response,
                t.response,
                seq=i,
                step_id=r.event.step_id,
                call_type=r.event.call_type,
                recorded_call_id=r.event.call_id,
                actual_call_id=t.event.call_id,
            )
        )
    return diff


def render_diff(diff: Diff, *, max_field_diffs: int = 8) -> str:
    """Render a :class:`Diff` as a human-readable string for the CLI."""
    lines: list[str] = []
    lines.append(f"Diff: {diff.source} → {diff.target}")
    s = diff.summary()
    lines.append(
        f"  {s['total_steps']} steps | "
        f"matching={s['matching']} diverged={s['diverged']} "
        f"extra_actual={s['extra_actual']} extra_recorded={s['extra_recorded']}"
    )
    if not diff.has_divergence:
        lines.append("  ✓ bit-exact match")
        return "\n".join(lines)
    first = diff.first_divergence
    if first is None:
        lines.append("  (no divergence details available)")
        return "\n".join(lines)
    lines.append(f"  ✗ first divergence at step {first.seq} ({first.call_type}, step_id={first.step_id!r})")
    lines.append(f"    recorded call_id: {first.recorded_call_id}")
    lines.append(f"    actual   call_id: {first.actual_call_id}")
    for i, fd in enumerate(first.field_diffs[:max_field_diffs]):
        lines.append(f"    · {fd.path}")
        lines.append(f"        recorded: {_short(fd.recorded)!r}")
        lines.append(f"        actual  : {_short(fd.actual)!r}")
    if len(first.field_diffs) > max_field_diffs:
        lines.append(f"    ... ({len(first.field_diffs) - max_field_diffs} more)")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Internals
# ---------------------------------------------------------------------- #
def _extract(value: Any, path: str) -> Any:
    """Extract the value at dotted path ``path`` from canonicalized ``value``."""
    if path in ("", "<root>"):
        return value
    cur: Any = value
    for token in _tokenize_path(path):
        try:
            if isinstance(token, int):
                cur = cur[token]
            elif isinstance(cur, dict):
                cur = cur[token]
            else:
                return None
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def _tokenize_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    cur = ""
    i = 0
    while i < len(path):
        c = path[i]
        if c == ".":
            if cur:
                tokens.append(cur)
                cur = ""
            i += 1
        elif c == "[":
            if cur:
                tokens.append(cur)
                cur = ""
            j = path.index("]", i)
            tokens.append(int(path[i + 1 : j]))
            i = j + 1
        else:
            cur += c
            i += 1
    if cur:
        tokens.append(cur)
    return tokens


def _short(value: Any, *, limit: int = 80) -> str:
    s = repr(value)
    if len(s) > limit:
        return s[: limit - 3] + "..."
    return s
