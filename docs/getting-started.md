# Installation

## Requirements

- Python 3.9 or newer
- No required runtime dependencies beyond `click`, `httpx`, and `pydantic`

## Install

```bash
# Core only (smallest install)
pip install agentreplay

# With framework adapters
pip install agentreplay[openai]        # OpenAI SDK adapter
pip install agentreplay[anthropic]     # Anthropic SDK adapter
pip install agentreplay[langgraph]     # LangGraph adapter (first-class)
pip install agentreplay[all]           # everything

# With dev tools (pytest, ruff, mypy)
pip install agentreplay[dev]
```

## Verify

```bash
agentreplay --version
# agentreplay, version 0.1.0
```

## From source

```bash
git clone https://github.com/gadda00/agentreplay.git
cd agentreplay
pip install -e .[dev]
pytest  # should be 99/99 green
```
