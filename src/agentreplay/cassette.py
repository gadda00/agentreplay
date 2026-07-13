"""The Cassette — a recorded agent run.

A cassette is a directory on disk holding:

    cassette.json   — metadata header
    events.jsonl    — append-only event log
    blobs/          — content-addressed blob store
    meta.db         — optional SQLite index for cross-cassette queries

The :class:`Cassette` class is the single entry point the rest of the
library uses; it owns the :class:`BlobStore`, :class:`EventLog` and
optional :class:`MetaIndex`, and exposes high-level operations:

    * ``write_event`` — record one intercepted call
    * ``lookup_call`` — match a call-site ID against the log (pure-replay)
    * ``resolve`` — fetch a request/response payload by hash
    * ``iter_records`` — iterate events with their payloads attached
    * ``diff_against`` — structural diff against another cassette
"""
from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from agentreplay.constants import CASSETTE_VERSION, CallType
from agentreplay.errors import CassetteError, CassetteNotFoundError
from agentreplay.storage import BlobStore, EventLog, MetaIndex
from agentreplay.types import Event, EventRecord


@dataclass
class CassetteMeta:
    """Header written to ``cassette.json``.

    Stored as plain JSON so a cassette is self-describing and portable
    across machines (no DB needed to inspect one).
    """

    id: str
    schema_version: str = CASSETTE_VERSION
    framework: str = "raw"
    agent_name: str = ""
    task_id: str = ""
    git_commit: str = ""
    model: str = ""
    outcome: str = ""              # "pass" | "fail" | "partial" | ""
    created_at: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    num_events: int = 0
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CassetteMeta":
        return cls(
            id=str(d["id"]),
            schema_version=str(d.get("schema_version", CASSETTE_VERSION)),
            framework=str(d.get("framework", "raw")),
            agent_name=str(d.get("agent_name", "")),
            task_id=str(d.get("task_id", "")),
            git_commit=str(d.get("git_commit", "")),
            model=str(d.get("model", "")),
            outcome=str(d.get("outcome", "")),
            created_at=float(d.get("created_at", 0.0)),
            duration_ms=float(d.get("duration_ms", 0.0)),
            num_events=int(d.get("num_events", 0)),
            tags=list(d.get("tags", [])),
            extra=dict(d.get("extra", {})),
        )


class Cassette:
    """A recorded agent run.

    Use :meth:`create` to start a fresh cassette in RECORD mode, or
    :meth:`open` to load an existing one for REPLAY.
    """

    META_FILE = "cassette.json"

    def __init__(
        self,
        root: Union[str, os.PathLike],
        meta: CassetteMeta,
        *,
        readonly: bool = False,
    ) -> None:
        self.root = Path(root)
        self.meta = meta
        self.readonly = readonly
        self.blobs = BlobStore(self.root)
        self.events = EventLog(self.root)
        # MetaIndex is optional and only created lazily for local-dev.
        self._meta_index: Optional[MetaIndex] = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def create(
        cls,
        root: Union[str, os.PathLike],
        *,
        framework: str = "raw",
        agent_name: str = "",
        task_id: str = "",
        git_commit: str = "",
        model: str = "",
        outcome: str = "",
        tags: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        cassette_id: Optional[str] = None,
    ) -> "Cassette":
        root = Path(root)
        if root.exists() and any(root.iterdir()):
            raise CassetteError(f"cassette root {root!s} is not empty")
        root.mkdir(parents=True, exist_ok=True)
        meta = CassetteMeta(
            id=cassette_id or f"cass-{uuid.uuid4().hex[:12]}",
            framework=framework,
            agent_name=agent_name,
            task_id=task_id,
            git_commit=git_commit,
            model=model,
            outcome=outcome,
            tags=list(tags or []),
            extra=dict(extra or {}),
        )
        c = cls(root, meta, readonly=False)
        c._write_meta()
        return c

    @classmethod
    def open(cls, root: Union[str, os.PathLike], *, readonly: bool = True) -> "Cassette":
        root = Path(root)
        meta_path = root / cls.META_FILE
        if not meta_path.exists():
            raise CassetteNotFoundError(f"no cassette at {root!s}")
        meta = CassetteMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        return cls(root, meta, readonly=readonly)

    # ------------------------------------------------------------------ #
    # Metadata persistence
    # ------------------------------------------------------------------ #
    def _write_meta(self) -> None:
        if self.readonly:
            raise CassetteError(f"cassette {self.meta.id} is readonly")
        self.meta.num_events = len(self.events)
        path = self.root / self.META_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.meta.to_dict(), indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def save(self) -> None:
        """Flush metadata header. Called by :class:`Recorder` on close."""
        self._write_meta()

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def write_event(
        self,
        *,
        step_id: str,
        call_type: Union[CallType, str],
        call_id: str,
        request: Any,
        response: Any,
        started_at: float,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        if self.readonly:
            raise CassetteError(f"cassette {self.meta.id} is readonly")
        req_hash = self.blobs.put(request)
        resp_hash = self.blobs.put(response)
        seq = len(self.events)
        event = Event(
            seq=seq,
            step_id=step_id,
            call_type=call_type.value if isinstance(call_type, CallType) else str(call_type),
            call_id=call_id,
            request_hash=req_hash,
            response_hash=resp_hash,
            started_at=started_at,
            duration_ms=duration_ms,
            metadata=dict(metadata or {}),
        )
        self.events.append(event)
        return event

    # ------------------------------------------------------------------ #
    # Replay
    # ------------------------------------------------------------------ #
    def lookup_call(self, call_id: str) -> Optional[Event]:
        return self.events.by_call_id(call_id)

    def resolve_request(self, event: Event) -> Any:
        return self.blobs.get(event.request_hash)

    def resolve_response(self, event: Event) -> Any:
        return self.blobs.get(event.response_hash)

    def iter_records(self) -> Iterator[EventRecord]:
        for e in self.events:
            yield EventRecord(
                event=e,
                request=self.resolve_request(e),
                response=self.resolve_response(e),
            )

    def records(self) -> List[EventRecord]:
        return list(self.iter_records())

    # ------------------------------------------------------------------ #
    # Mutation (counterfactual)
    # ------------------------------------------------------------------ #
    def fork(self, new_root: Union[str, os.PathLike], *, new_id: Optional[str] = None) -> "Cassette":
        """Copy this cassette to ``new_root`` so it can be mutated.

        The blob store is reused (hardlinked where possible) so a fork
        costs almost nothing on disk even for very large cassettes.
        """
        new_root = Path(new_root)
        new_root.mkdir(parents=True, exist_ok=True)
        # Hardlink blobs (cheap on same filesystem; falls back to copy).
        src_blobs = self.root / BlobStore.BLOB_DIR
        dst_blobs = new_root / BlobStore.BLOB_DIR
        dst_blobs.mkdir(parents=True, exist_ok=True)
        if src_blobs.exists():
            for path in src_blobs.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(src_blobs)
                    target = dst_blobs / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(path, target)
                    except OSError:
                        shutil.copy2(path, target)
        # Copy events log.
        shutil.copy2(self.events.path, new_root / EventLog.FILENAME)
        # New metadata header.
        new_meta = CassetteMeta(
            id=new_id or f"cass-{uuid.uuid4().hex[:12]}",
            schema_version=self.meta.schema_version,
            framework=self.meta.framework,
            agent_name=self.meta.agent_name,
            task_id=self.meta.task_id,
            git_commit=self.meta.git_commit,
            model=self.meta.model,
            outcome=self.meta.outcome,
            created_at=time.time(),
            duration_ms=self.meta.duration_ms,
            num_events=self.meta.num_events,
            tags=list(self.meta.tags),
            extra=dict(self.meta.extra),
        )
        new_c = Cassette(new_root, new_meta, readonly=False)
        new_c._write_meta()
        return new_c

    def replace_response(self, seq: int, new_response: Any) -> Event:
        """Replace the recorded response for event ``seq``.

        Used by the counterfactual-mutation engine to inject "what if the
        tool had returned this instead?" patches. The request hash is
        left untouched so the call-site ID stays matchable.
        """
        if self.readonly:
            raise CassetteError(f"cassette {self.meta.id} is readonly")
        # Rewrite the events log with the patched row.
        all_events = self.events.all()
        if seq < 0 or seq >= len(all_events):
            raise CassetteError(f"seq {seq} out of range (have {len(all_events)} events)")
        old = all_events[seq]
        new_hash = self.blobs.put(new_response)
        patched = Event(
            seq=old.seq,
            step_id=old.step_id,
            call_type=old.call_type,
            call_id=old.call_id,
            request_hash=old.request_hash,
            response_hash=new_hash,
            started_at=old.started_at,
            duration_ms=old.duration_ms,
            metadata={**old.metadata, "mutated": True},
        )
        all_events[seq] = patched
        # Atomically rewrite events.jsonl.
        tmp = self.events.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in all_events:
                f.write(json.dumps(e.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, self.events.path)
        # Force the EventLog to rebuild its in-memory index so subsequent
        # lookups see the patched event.
        self.events.rebuild_index()
        self.meta.num_events = len(all_events)
        self._write_meta()
        return patched

    # ------------------------------------------------------------------ #
    # Diff
    # ------------------------------------------------------------------ #
    def diff_against(self, other: "Cassette") -> "Diff":
        """High-level structural comparison between two cassettes."""
        from agentreplay.diff import Diff, diff_structural

        return diff_structural(self, other)

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        return {
            "id": self.meta.id,
            "framework": self.meta.framework,
            "task_id": self.meta.task_id,
            "outcome": self.meta.outcome,
            "num_events": len(self.events),
            "blobs": self.blobs.stats(),
        }

    # ------------------------------------------------------------------ #
    # Export / Import (ZIP for portability)
    # ------------------------------------------------------------------ #
    def export_zip(self, zip_path: Union[str, os.PathLike]) -> Path:
        """Export this cassette as a ZIP archive for sharing.

        The ZIP contains:
            cassette.json
            events.jsonl
            blobs/<sha256[:2]>/<sha256>
            blobs/...

        Use :meth:`Cassette.import_zip` to reconstruct the cassette
        from the archive.
        """
        import zipfile

        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Metadata
            zf.write(self.root / self.META_FILE, self.META_FILE)
            # Events
            events_path = self.root / EventLog.FILENAME
            if events_path.exists():
                zf.write(events_path, EventLog.FILENAME)
            # Blobs
            blobs_dir = self.root / BlobStore.BLOB_DIR
            if blobs_dir.exists():
                for path in blobs_dir.rglob("*"):
                    if path.is_file():
                        arcname = str(path.relative_to(self.root))
                        zf.write(path, arcname)
        return zip_path

    @classmethod
    def import_zip(
        cls,
        zip_path: Union[str, os.PathLike],
        target_root: Union[str, os.PathLike],
        *,
        readonly: bool = True,
    ) -> "Cassette":
        """Import a cassette from a ZIP archive created by :meth:`export_zip`.

        Extracts the archive to ``target_root`` and opens the resulting
        cassette directory.
        """
        import zipfile

        zip_path = Path(zip_path)
        target_root = Path(target_root)
        if target_root.exists() and any(target_root.iterdir()):
            raise CassetteError(f"target root {target_root!s} is not empty")
        target_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_root)
        return cls.open(target_root, readonly=readonly)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<Cassette id={self.meta.id!r} framework={self.meta.framework!r} "
            f"events={len(self)} blobs={len(self.blobs)}>"
        )
