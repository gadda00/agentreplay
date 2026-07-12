# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
