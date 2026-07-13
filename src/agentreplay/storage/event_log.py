"""Append-only event log (JSONL).

Each row corresponds to exactly one intercepted call and holds the
call-site ID, call type, timestamps, metadata, and references (SHA-256)
to the request and response payloads stored in the blob store.

We use JSONL rather than a single JSON array so the file can be appended
to in O(1) during recording, and so a corrupted tail (e.g. process
crash mid-write) only loses the last line.

Performance: an in-memory index is built lazily on the first read and
kept up-to-date on append. This makes ``by_call_id`` and ``by_step``
O(1) / O(k) instead of O(n) — critical for replay, where every
intercepted call does a lookup.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from agentreplay.types import Event


class EventLog:
    """JSONL append-only event log with in-memory index.

    The log is *monotonic*: rows are appended with a sequence number
    (``seq``) and never edited in place. Counterfactual mutation creates
    a *new* cassette rather than rewriting history, so the original
    recording remains a pristine, reproducible artifact.

    An in-memory index (``_by_call_id``, ``_by_step``) is built lazily
    on the first read and updated on each append. This makes lookups
    O(1) instead of O(n) — critical for replay performance.
    """

    FILENAME = "events.jsonl"

    def __init__(self, root: Union[str, os.PathLike]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILENAME
        self._lock = threading.Lock()
        # Lazy in-memory index — built on first access, kept up-to-date on append.
        self._index_dirty = True
        self._by_call_id: Dict[str, Event] = {}
        self._by_step: Dict[str, List[Event]] = {}
        self._by_seq: Dict[int, Event] = {}
        self._count = 0

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #
    def _rebuild_index(self) -> None:
        """Rebuild the in-memory index from the JSONL file."""
        self._by_call_id.clear()
        self._by_step.clear()
        self._by_seq.clear()
        self._count = 0
        if not self.path.exists():
            self._index_dirty = False
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = Event.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError):
                    # Skip corrupted lines (e.g. partial write from crash)
                    continue
                self._index_event(event)
                self._count += 1
        self._index_dirty = False

    def _index_event(self, event: Event) -> None:
        """Add a single event to the in-memory index."""
        self._by_call_id[event.call_id] = event
        self._by_step.setdefault(event.step_id, []).append(event)
        self._by_seq[event.seq] = event

    def _ensure_index(self) -> None:
        """Build the index if it hasn't been built yet. Thread-safe."""
        if self._index_dirty:
            with self._lock:
                # Double-check under lock — another thread may have built it
                if self._index_dirty:
                    self._rebuild_index()

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def append(self, event: Event) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            # Update the in-memory index so lookups stay O(1).
            # Don't call _ensure_index here — if the index is dirty it
            # would rebuild from the file (which now includes the event
            # we just appended), and then _index_event would double-count.
            # Instead, just add the event to the index directly.
            if self._index_dirty:
                # Index hasn't been built yet — it will be built lazily
                # on the first read, which will pick up this event from
                # the file. No need to do anything here.
                pass
            else:
                # Index is already built — add the new event to it.
                self._index_event(event)
                self._count += 1

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[Event]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Event.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError):
                    continue

    def __len__(self) -> int:
        self._ensure_index()
        return self._count

    def all(self) -> List[Event]:
        return list(self)

    def at(self, seq: int) -> Optional[Event]:
        """O(1) lookup by sequence number."""
        self._ensure_index()
        return self._by_seq.get(seq)

    def by_call_id(self, call_id: str) -> Optional[Event]:
        """O(1) lookup by call-site ID."""
        self._ensure_index()
        return self._by_call_id.get(call_id)

    def by_step(self, step_id: str) -> List[Event]:
        """O(1) lookup by step ID (returns list, may be empty)."""
        self._ensure_index()
        return list(self._by_step.get(step_id, []))

    def rebuild_index(self) -> None:
        """Force a rebuild of the in-memory index. Useful after external
        modifications to the JSONL file (e.g. mutation via ``replace_response``)."""
        self._rebuild_index()
