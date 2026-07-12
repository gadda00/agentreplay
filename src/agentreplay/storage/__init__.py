"""Storage primitives.

A *cassette* is a directory on disk with this layout::

    <cassette>/
      cassette.json          # metadata: agent, framework, task, git commit, outcome
      events.jsonl           # append-only event log, one row per intercepted call
      blobs/                 # content-addressed blob store (one file per SHA-256)
        <sha256>
        <sha256>
        ...
      meta.db                # SQLite metadata index (optional, local-dev only)

The split between ``events.jsonl`` (small, indexed, frequently read) and
``blobs/`` (large, content-addressed, deduplicated) is deliberate: it lets
us read the structure of a run in milliseconds without loading megabytes
of payload, and lets us deduplicate across runs for free — a system prompt
recorded once is referenced, not re-stored, on every subsequent call.
"""
from agentreplay.storage.blob import BlobStore
from agentreplay.storage.event_log import EventLog
from agentreplay.storage.meta_index import MetaIndex

__all__ = ["BlobStore", "EventLog", "MetaIndex"]
