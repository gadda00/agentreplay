# Evaluation Methodology

A tool that claims determinism has to be judged on whether it delivers
determinism, not on vibes. Three measurements anchor the evaluation,
each tied directly to a claim made in the product proposal (§7).

## 1. Reproduction fidelity (§7.1)

**Target**: 100% bit-exact reproduction for unmodified agent code.

**Method**: For every cassette in the validation set, replay it in
pure-replay mode and compare the terminal agent state — final answer,
tool-call sequence, exit condition — against the originally recorded
terminal state. Any failure to reproduce indicates an uninstrumented
source of nondeterminism (§8) that must be found and closed before the
tool can be trusted.

**How to run**:

```bash
# Synthetic validation (CI-friendly, no API key)
agentreplay validate-swebench --tasks synthetic --limit 5
agentreplay validate-gaia --tasks synthetic --limit 5

# Real SWE-bench Verified (requires API key + dataset download)
agentreplay validate-swebench --tasks swebench-verified --limit 20

# Real GAIA subset
agentreplay validate-gaia --tasks gaia-subset --limit 20
```

The test suite in `tests/test_record_replay.py` includes end-to-end
reproduction-fidelity checks: every `record → replay` test asserts that
the replayed responses are byte-equal to the recorded ones, and that
the live client is never invoked.

## 2. Overhead (§7.2)

**Target**: ≤ 5% latency increase vs. uninstrumented run (comparable to
or better than Laminar's ~5% figure).

**Method**: Measure recording-layer latency overhead against a live
baseline, using the same methodology as the independent 2026 four-platform
benchmark (percentage latency increase versus an uninstrumented run, on
an identical repeated workload).

**How to run**:

```bash
agentreplay benchmark-overhead --iterations 200 --report report.json
```

The benchmark also runs synthetic baselines matching the published 2026
figures for LangSmith (~0%), Laminar (~5%), AgentOps (~12%), Langfuse
(~15%). These are *not* measurements of those tools (which would require
running them with their full SDK + backend), but simulated baselines
that let the report put AgentReplay's number in context.

**Current result** (on the development machine):

```
AgentReplay overhead benchmark — 50 iterations
Baseline (uninstrumented): 50.0025 ms/call

Tool                            ms/call  overhead%
--------------------------------------------------
baseline                        50.0025       0.00
AgentReplay (record)            50.3369       0.67
AgentReplay (replay)             0.3415     -99.32
LangSmith (synthetic)           50.0024      -0.00
Laminar (synthetic)             52.5034       5.00
AgentOps (synthetic)            56.0026      12.00
Langfuse (synthetic)            57.5027      15.00

✓ AgentReplay overhead = 0.67% (≤ 5% target from §7.2)
```

AgentReplay's overhead is **0.67%** — well under the 5% target. Replay
mode is **99.32% faster** than the baseline (since it makes zero model
calls).

## 3. Cost impact (§7.3)

**Target**: ~100% reduction in *investigation* cost (not total inference).

**Method**: For the validation set, compute the marginal API cost of
investigating each failure twice — once via a traditional live re-run,
once via AgentReplay's pure replay — and report the delta.

Because pure replay makes zero model calls by construction, the expected
result is a reduction approaching 100% of the *investigation* cost
specifically (not total agent operating cost, which includes production
traffic, not just debugging).

This should be reported honestly as a debugging-cost saving rather than
conflated with the much larger inference-spend figures in §2.4 of the
product proposal, which include production traffic, not just debugging.

## Validation task sets

| Task set | Description | Status |
|---|---|---|
| `synthetic` | Built-in synthetic tasks (CI-friendly, no API key) | ✓ Implemented |
| `swebench-verified` | Real SWE-bench Verified (500 tasks, frozen Docker repos) | ◻ Stub — requires `datasets` + API key |
| `gaia-subset` | Real GAIA subset (20 tasks, live web/search) | ◻ Stub — requires `datasets` + API key |

The synthetic task set exercises the same record/replay code paths as
the real task sets but uses a stub LLM, so it runs in CI without
external dependencies. The harness's ability to detect a divergence IS
meaningful even with a stub LLM; the fidelity *numbers* from the
synthetic set are not meaningful as product claims (a stub LLM is
trivially reproducible).

To plug in the real task sets, subclass `TaskSet` and implement `load()`
— see `agentreplay/validation/tasks.py` for details.
