# Cassette Format

A cassette is a directory on disk containing four artefacts:

```
<cassette>/
  cassette.json     # metadata header
  events.jsonl      # append-only event log
  blobs/            # content-addressed blob store
    <sha256[:2]>/<sha256>   # sharded by first 2 hex chars
  meta.db           # SQLite metadata index (optional)
```

The format is plain JSON — no pickle, no opaque blobs. A cassette is
self-describing and portable across machines.

## `cassette.json`

Metadata header. Written by `Recorder.save()` on close.

```json
{
  "id": "cass-bf9c3190d4c2",
  "schema_version": "1.0.0",
  "framework": "raw",
  "agent_name": "sample",
  "task_id": "",
  "git_commit": "a1b2c3d...",
  "model": "stub",
  "outcome": "pass",
  "created_at": 1783863793.116,
  "duration_ms": 0.828,
  "num_events": 3,
  "tags": ["sample", "regression"],
  "extra": {}
}
```

| Field | Description |
|---|---|
| `id` | Unique cassette ID (`cass-<random>`) |
| `schema_version` | Cassette format version (semver) |
| `framework` | Framework used (`raw`, `openai`, `anthropic`, `langgraph`) |
| `agent_name` | Human-readable agent name |
| `task_id` | Task identifier (e.g. `swe-bench:214`) |
| `git_commit` | Git commit hash at record time |
| `model` | Model name (e.g. `gpt-4o-mini`) |
| `outcome` | `pass`, `fail`, `partial`, or empty |
| `created_at` | Epoch seconds |
| `duration_ms` | Total recording duration |
| `num_events` | Number of events in `events.jsonl` |
| `tags` | Free-form tags for filtering |
| `extra` | Free-form metadata dict |

## `events.jsonl`

Append-only event log, one JSON object per line. Each row corresponds
to exactly one intercepted call.

```json
{
  "seq": 0,
  "step_id": "step:0:0",
  "call_type": "llm",
  "call_id": "365f9fc8...",
  "request_hash": "035621c2...",
  "response_hash": "d8175ac5...",
  "started_at": 1783863829.766,
  "duration_ms": 0.004,
  "metadata": {"call_type": "custom", "model": "stub"}
}
```

| Field | Description |
|---|---|
| `seq` | 0-indexed position within the cassette |
| `step_id` | Step identifier (e.g. `langgraph:router`) |
| `call_type` | `llm`, `tool`, `http`, `clock`, `rng`, or `other` |
| `call_id` | SHA-256 of `(step_id, canonicalized_input)` |
| `request_hash` | SHA-256 of the canonicalized request payload |
| `response_hash` | SHA-256 of the recorded response payload |
| `started_at` | Epoch seconds when the call started |
| `duration_ms` | Call duration in milliseconds |
| `metadata` | Free-form per-event metadata |

## `blobs/`

Content-addressed blob store. Each blob is a JSON-serialised canonical
payload, stored at `blobs/<sha256[:2]>/<sha256>`.

Blobs are deduplicated: the same system prompt recorded on every step
of a 50-step run is stored exactly once. This is the mechanism from
§5.2 / §8 for keeping storage growth bounded.

Sharding by the first 2 hex characters avoids creating 100k+ files in
a single directory on large cassettes.

## `meta.db` (optional)

SQLite metadata index for cross-cassette queries. Created lazily by
the `MetaIndex` class. Stores the same fields as `cassette.json` so
you can run queries like:

```sql
SELECT id, path FROM cassettes WHERE outcome = 'fail' AND task_id = 'swe-bench:214';
```

The index is optional — a cassette on disk is self-describing via
`cassette.json` and can be used directly without it. The index exists
purely to make multi-cassette queries fast for teams that accumulate
a regression corpus.

## Schema versioning

The `schema_version` field in `cassette.json` follows semver. The
current version is `1.0.0`. Future versions will bump:

- **Patch** (1.0.x): backwards-compatible additions (new optional fields).
- **Minor** (1.x.0): backwards-compatible new features (new call types,
  new metadata fields).
- **Major** (x.0.0): breaking changes (renamed fields, removed fields,
  changed blob format).

The library checks `schema_version` on cassette open and warns if the
cassette was written by a newer version of the library than the one
reading it.
