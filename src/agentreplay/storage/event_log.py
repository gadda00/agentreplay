"""Append-only event log (JSONL).

Each row corresponds to exactly one intercepted call and holds the
call-site ID, call type, timestamps, metadata, and references (SHA-256)
to the request and response payloads stored in the blob store.

We use JSONL rather than a single JSON array so the file can be appended
to in O(1) during recording, and so a corrupted tail (e.g. process
crash mid-write) only loses the last line.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterator, List, Optional, Union

from agentreplay.types import Event


class EventLog:
    """JSONL append-only event log.

    The log is *monotonic*: rows are appended with a sequence number
    (``seq``) and never edited in place. Counterfactual mutation creates
    a *new* cassette rather than rewriting history, so the original
    recording remains a pristine, reproducible artifact.
    """

    FILENAME = "events.jsonl"

    def __init__(self, root: Union[str, os.PathLike]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILENAME
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def append(self, event: Event) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

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
                yield Event.from_dict(json.loads(line))

    def __len__(self) -> int:
        if not self.path.exists():
            return 0
        n = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def all(self) -> List[Event]:
        return list(self)

    def at(self, seq: int) -> Optional[Event]:
        for e in self:
            if e.seq == seq:
                return e
        return None

    def by_call_id(self, call_id: str) -> Optional[Event]:
        for e in self:
            if e.call_id == call_id:
                return e
        return None

    def by_step(self, step_id: str) -> List[Event]:
        return [e for e in self if e.step_id == step_id]
