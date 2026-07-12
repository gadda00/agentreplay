"""Content-addressed blob store.

Blobs are written once and referenced by SHA-256 forever. Deduplication
is automatic: the same system prompt recorded on every step of a 50-step
run is stored exactly once. This is the mechanism the product proposal
(§5.2, §8) calls out for keeping storage growth bounded.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Union

from agentreplay.hashing import canonical_json, hash_payload


class BlobStore:
    """A simple on-disk content-addressed store.

    The store is *block-level* deduplicated: each blob is a JSON-serialised
    canonical payload, written to ``blobs/<sha256>``. Subdirectories are
    sharded by the first two hex characters once the directory grows past
    a small threshold (avoids creating 100k files in one directory on
    large cassettes).
    """

    BLOB_DIR = "blobs"

    def __init__(self, root: Union[str, os.PathLike]) -> None:
        self.root = Path(root)
        self.blob_dir = self.root / self.BLOB_DIR
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #
    def put(self, value: Any) -> str:
        """Store ``value`` and return its SHA-256 hex digest."""
        payload = canonical_json(value).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        path = self._path_for(digest)
        if not path.exists():
            with self._lock:
                if not path.exists():  # double-check under lock
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(payload)
                    os.replace(tmp, path)
        return digest

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #
    def get(self, digest: str) -> Any:
        """Return the canonical payload stored at ``digest``.

        Raises ``KeyError`` if the blob is missing.
        """
        path = self._path_for(digest)
        if not path.exists():
            raise KeyError(f"blob {digest!r} not found in {self.blob_dir}")
        return json.loads(path.read_text(encoding="utf-8"))

    def has(self, digest: str) -> bool:
        return self._path_for(digest).exists()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _path_for(self, digest: str) -> Path:
        # Always shard by the first two hex characters. The decision must
        # be STABLE across writes and reads — a dynamic "shard once we
        # cross N files" rule would cause reads to fail for blobs written
        # before the threshold (they'd be in the flat path but the read
        # would look in the sharded path).
        return self.blob_dir / digest[:2] / digest

    def __len__(self) -> int:
        if not self.blob_dir.exists():
            return 0
        total = 0
        for _root, _dirs, files in os.walk(self.blob_dir):
            total += len(files)
        return total

    def total_bytes(self) -> int:
        if not self.blob_dir.exists():
            return 0
        total = 0
        for root, _dirs, files in os.walk(self.blob_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    def stats(self) -> Dict[str, int]:
        return {"blobs": len(self), "bytes": self.total_bytes()}
