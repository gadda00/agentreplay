"""SQLite metadata index for cassettes.

Stores per-cassette metadata (task ID, git commit, model, pass/fail outcome)
so a team can run queries like "every failing cassette for benchmark task
214" or "every cassette recorded against commit a1b2c3d" without scanning
the filesystem.

The index is optional: a cassette on disk is *self-describing* (see
``cassette.json``) and can be used directly without a metadata index.
The index exists purely to make multi-cassette queries fast for teams
that accumulate a regression corpus.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cassettes (
    id           TEXT PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    task_id      TEXT,
    git_commit   TEXT,
    model        TEXT,
    framework    TEXT,
    outcome      TEXT,
    created_at   REAL,
    duration_ms  REAL,
    num_events   INTEGER,
    tags         TEXT,            -- JSON array
    extra        TEXT             -- JSON object
);

CREATE INDEX IF NOT EXISTS idx_cassettes_task     ON cassettes(task_id);
CREATE INDEX IF NOT EXISTS idx_cassettes_commit   ON cassettes(git_commit);
CREATE INDEX IF NOT EXISTS idx_cassettes_model    ON cassettes(model);
CREATE INDEX IF NOT EXISTS idx_cassettes_outcome  ON cassettes(outcome);
"""


class MetaIndex:
    """Thin SQLite wrapper around cassette metadata."""

    FILENAME = "meta.db"

    def __init__(self, root: Union[str, os.PathLike]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILENAME
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def upsert(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cassettes
                    (id, path, task_id, git_commit, model, framework,
                     outcome, created_at, duration_ms, num_events, tags, extra)
                VALUES
                    (:id, :path, :task_id, :git_commit, :model, :framework,
                     :outcome, :created_at, :duration_ms, :num_events, :tags, :extra)
                ON CONFLICT(id) DO UPDATE SET
                    path=excluded.path,
                    task_id=excluded.task_id,
                    git_commit=excluded.git_commit,
                    model=excluded.model,
                    framework=excluded.framework,
                    outcome=excluded.outcome,
                    created_at=excluded.created_at,
                    duration_ms=excluded.duration_ms,
                    num_events=excluded.num_events,
                    tags=excluded.tags,
                    extra=excluded.extra
                """,
                {
                    "id": entry["id"],
                    "path": entry["path"],
                    "task_id": entry.get("task_id"),
                    "git_commit": entry.get("git_commit"),
                    "model": entry.get("model"),
                    "framework": entry.get("framework"),
                    "outcome": entry.get("outcome"),
                    "created_at": float(entry.get("created_at", 0.0)),
                    "duration_ms": float(entry.get("duration_ms", 0.0)),
                    "num_events": int(entry.get("num_events", 0)),
                    "tags": json.dumps(entry.get("tags", [])),
                    "extra": json.dumps(entry.get("extra", {})),
                },
            )
            self._conn.commit()

    def delete(self, cassette_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM cassettes WHERE id = ?", (cassette_id,))
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def get(self, cassette_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM cassettes WHERE id = ?", (cassette_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(
        self,
        *,
        task_id: Optional[str] = None,
        git_commit: Optional[str] = None,
        model: Optional[str] = None,
        outcome: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM cassettes WHERE 1=1"
        params: list[Any] = []
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        if git_commit:
            sql += " AND git_commit = ?"
            params.append(git_commit)
        if model:
            sql += " AND model = ?"
            params.append(model)
        if outcome:
            sql += " AND outcome = ?"
            params.append(outcome)
        if tag:
            sql += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MetaIndex":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["extra"] = json.loads(d.get("extra") or "{}")
        return d
