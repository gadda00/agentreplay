# CLI Reference

The `agentreplay` command-line tool provides subcommands for inspecting,
replaying, diffing, and mutating cassettes, plus a CI regression runner
and an overhead benchmark.

## `agentreplay show`

Print cassette metadata (and optionally every event row).

```bash
agentreplay show cassettes/run-001
agentreplay show cassettes/run-001 --events
```

## `agentreplay list`

List cassettes in a corpus, optionally filtered by metadata.

```bash
agentreplay list cassettes/
agentreplay list cassettes/ --outcome fail
agentreplay list cassettes/ --tag regression
agentreplay list cassettes/ --task swe-bench:214
agentreplay list cassettes/ --json
```

## `agentreplay record`

Run a subprocess with the AgentReplay recorder auto-installed.

```bash
agentreplay record cassettes/run-001 -- python my_agent.py
agentreplay record cassettes/run-001 --framework langgraph --tag regression -- python my_agent.py
```

The child process must call `agentreplay.auto.init()` at startup to
pick up the env vars set by this command.

## `agentreplay replay`

Replay a cassette through an agent entry point.

```bash
# Just verify the cassette parses and print stats
agentreplay replay cassettes/run-001

# Replay through an agent
agentreplay replay cassettes/run-001 --agent-entry my_agent:run

# Hybrid mode (fall through to live on divergence)
agentreplay replay cassettes/run-001 --agent-entry my_agent:run --mode hybrid
```

Exits 0 on success, 2 on divergence.

## `agentreplay diff`

Structural diff between two cassettes.

```bash
agentreplay diff cassettes/baseline cassettes/mutated
agentreplay diff cassettes/baseline cassettes/mutated --json
```

## `agentreplay mutate`

Create a counterfactual cassette by replacing one recorded response.

```bash
agentreplay mutate cassettes/run-001 \
    --seq 3 \
    --response '{"value": null, "error": "PermissionError"}' \
    --out cassettes/run-001-mutated

# Or target by step_id / call_id
agentreplay mutate cassettes/run-001 --step-id langgraph:router --response-file patch.json --out cassettes/mutated
```

## `agentreplay ci`

Replay every cassette in a corpus through an agent entry point.

```bash
agentreplay ci cassettes/ --agent-entry my_project.tests:run_agent
agentreplay ci cassettes/ --agent-entry my_project.tests:run_agent --tag regression
agentreplay ci cassettes/ --agent-entry my_project.tests:run_agent --stop-on-first-failure
agentreplay ci cassettes/ --agent-entry my_project.tests:run_agent --json
```

Exits 0 if all cassettes replayed bit-exact, 1 otherwise.

## `agentreplay benchmark-overhead`

Measure recording-layer latency overhead vs. baseline (§7.2).

```bash
agentreplay benchmark-overhead --iterations 200
agentreplay benchmark-overhead --iterations 200 --report report.json --json
```

Exits 0 if AgentReplay's overhead is ≤ 5% (the §7.2 target), 1 otherwise.

## `agentreplay validate-swebench`

Run reproduction-fidelity validation (§7.1) on a SWE-bench task set.

```bash
# Synthetic (CI-friendly, no API key)
agentreplay validate-swebench --tasks synthetic --limit 5

# Real SWE-bench Verified (requires setup)
agentreplay validate-swebench --tasks swebench-verified --limit 20
```

Exits 0 if fidelity = 100% (§7.1 target), 1 otherwise.

## `agentreplay validate-gaia`

Run reproduction-fidelity validation (§7.1) on a GAIA task set.

```bash
agentreplay validate-gaia --tasks synthetic --limit 5
agentreplay validate-gaia --tasks gaia-subset --limit 20
```

Exits 0 if fidelity = 100% (§7.1 target), 1 otherwise.
