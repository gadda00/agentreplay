# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Streaming response support** (CRITICAL): `RecordingClient.complete()` and
  `acomplete()` now handle `stream=True` (OpenAI/Anthropic). Chunks are
  captured via `RecordingStream` and stored as a single event with
  `{"chunks": [...], "streamed": True}`. On replay, a `ReplayStream`
  yields the recorded chunks. Without this, agents using streaming
  bypassed the recording layer entirely.
- **Cassette export/import** (HIGH): `Cassette.export_zip()` and
  `Cassette.import_zip()` for sharing cassettes as single ZIP archives.
  CLI: `agentreplay export <cassette> <zip>` and `agentreplay import
  <zip> <target>`.
- **`agentreplay info` command**: shows installed version, Python version,
  install path, optional dependency status, and available framework adapters.
- **`agentreplay clean` command**: removes old/unwanted cassettes from a
  corpus. Supports `--older-than 30d`, `--keep-outcome fail`, `--dry-run`
  (default) / `--no-dry-run`.
- **`--verbose` flag** on the CLI: enables debug logging for all
  interceptors. Also configurable via `AGENTREPLAY_VERBOSE=1` or
  `AGENTREPLAY_LOG_LEVEL=DEBUG` env vars.
- **Logging module** (`agentreplay.logging`): centralised logger with
  `get_logger()` and `set_verbose()`. All interceptors now log at DEBUG
  level, silent by default.
- **28 new tests**: streaming (10), export/import + CLI commands (11),
  EventLog index (7). Total: 133 tests, all passing.

### Fixed
- **CRITICAL: EventLog O(n) lookup** — `by_call_id()` and `by_step()`
  scanned the entire events.jsonl file on EVERY call. During replay, every
  intercepted call did a full file scan. A 1000-event cassette = 500k line
  reads. Now uses an in-memory index (lazily built, kept up-to-date on
  append) for O(1) lookups. `replace_response()` now calls
  `rebuild_index()` after rewriting the file.
- **CRITICAL: Async `acomplete` race condition** — the `_call_counter` was
  incremented at different points in the sync vs async paths, and in REPLAY
  mode the async path delegated to sync which double-incremented. Now both
  paths increment the counter exactly once at the top of the method.
- **HIGH: HTTP interceptor exception handling** — exceptions from the real
  client were not captured. Now they're recorded as the response payload
  (with `error` field) and re-raised, so replay can reproduce connection
  errors, timeouts, etc.
- **MEDIUM: `_ReplayResponse` missing attributes** — added `.ok`,
  `.is_success`, `.is_redirect`, `.is_client_error`, `.is_server_error`,
  `.is_error`, `.url`, `.encoding`, `.elapsed`, `.reason`, `.cookies`,
  `iter_bytes()`, `iter_content()`, `close()`, `__repr__()` for full
  httpx/requests compatibility.

## [0.1.0] — 2026-07-12

### Added
- Initial public release.
- Core: `Cassette`, `BlobStore`, `EventLog`, `MetaIndex`.
- Hashing: `canonicalize`, `hash_call_site`, `diff_keys`.
- Interceptors: `RecordingClient` (LLM), `RecordingTool`, `RecordingHTTP`,
  `RecordingClock`, `RecordingRandom` — all support LIVE/RECORD/REPLAY/HYBRID.
- Orchestrators: `Recorder`, `Replayer`, `Session`.
- Counterfactual mutation engine: `mutate_response`, `mutate_and_replay`,
  `apply_patch_set`.
- Structural diff engine: `diff_structural`, `diff_payloads`, `render_diff`.
- CI regression runner: `run_corpus`, `RegressionReport`.
- CLI: `agentreplay show / list / record / replay / diff / mutate / ci`.
- Framework adapters: OpenAI SDK, Anthropic SDK, LangGraph, raw agent loops.
- Auto-init helper for the `agentreplay record` subprocess wrapper.
- 75-test pytest suite with end-to-end reproduction-fidelity tests.
- Examples: raw agent, counterfactual mutation, LangGraph integration,
  OpenAI SDK integration.
- GitHub Actions regression workflow template.
- Architecture documentation.

[0.1.0]: https://github.com/gadda00/agentreplay/releases/tag/v0.1.0
[Unreleased]: https://github.com/gadda00/agentreplay/compare/v0.1.0...HEAD
