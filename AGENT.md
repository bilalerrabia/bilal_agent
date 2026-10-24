# AGENT.md - project memory for bilal_agent

Facts below are injected into the agent's system prompt on every start.
Keep it short, factual and current - the agent can edit this file too.

## Project
- This repository IS bilal_agent: a terminal AI coding agent (Claude Code style).
- Python 3.11+, managed with uv, deps: openai + rich only.

## Commands
- run tests: python3 -m pytest tests/ -q
- run the CLI locally: uv run main.py --help

## Conventions
- Tool functions take ctx first; raise ToolError for model-readable failures.
- Tools always return strings; errors are strings too, never exceptions that escape the loop.
- JSON schemas keep "required" INSIDE the parameters object.
- The ledger format is plain OpenAI dicts; the system message is never stored in it.
