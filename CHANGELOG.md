# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CrewAI framework adapter (`agentreplay.frameworks.crewai.wrap_crewai_llm`).
  Patches the CrewAI `LLM.call` method in place so every model invocation
  is captured.
- AutoGen framework adapter — supports both v0.2 (`wrap_autogen_client`)
  and v0.4+ (`wrap_autogen_v4_agent`).
- Async LLM client support: `RecordingClient.acomplete()` coroutine.
  Works with `AsyncOpenAI`, `AsyncAnthropic`, or any client with an
  `acomplete` method. Falls back to running sync `complete` in a thread
  if the wrapped client has no async path.
- Real SWE-bench Verified task loader: `SwebenchVerifiedTaskSet.load()`
  now downloads the dataset from HuggingFace and converts each task to
  a `Task` object (requires `pip install datasets`).
- Real GAIA task loader: `GaiaTaskSet.load()` — same pattern
  (requires `pip install datasets` + accepting the GAIA license on HF).
- New optional extras: `[crewai]`, `[autogen]`, `[datasets]`.
- `pymdown-extensions` added to `[docs]` extra (mkdocs build was failing
  without it).
- 11 new tests: 8 for CrewAI/AutoGen adapters, 3 for async support.
  Total: 105 tests, all passing.

### Fixed
- **Critical**: YAML syntax error in `.github/workflows/regression.yml` —
  the step names `Overhead benchmark (§7.2 target: ≤5%)` and
  `Reproduction-fidelity validation (§7.1 target: 100%)` contained
  colons that broke the YAML parser, preventing ALL jobs in the workflow
  from running. Renamed to `Overhead benchmark (target ≤5%)` and
  `Reproduction-fidelity validation (target 100%)`.
- **Critical**: `mkdocs build --strict` was failing in the docs workflow
  because `pymdownx.admonition` was removed in pymdown-extensions v11.
  Replaced with `pymdownx.details` and added `pymdown-extensions` to
  the docs install step.
- Broken docs links: `architecture.md` linked to `../README.md` (not
  in docs tree), `contributing.md` linked to `docs/architecture.md`
  (wrong path), `guides/recording.md` linked to a non-existent
  `#risks-and-limitations` anchor. All fixed; the "Risks and limitations"
  section was added to `architecture.md`.

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
