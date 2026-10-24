# bilal_agent

An AI coding agent that lives in your terminal - the Claude Code experience, small enough to read all of it.

It explores your project with tools (read, grep, glob, list), edits files with diffs you approve, runs shell commands, keeps a plan, and compacts its own memory on long sessions. It **remembers** (project journal + resumable sessions), it **delegates** (read-only sub-agents for heavy exploration), it **protects your work** (git checkpoint before every turn, `/undo` to roll back), and it ships a **telemetry TUI**: slim header with session card, `◇ thinking` plan blocks, `◆ agent::tools` call trees, a live `tok / cost / turn` economy line above every prompt, and a full `/stats` dashboard. Any OpenAI-compatible provider works (zenmux, OpenRouter, OpenAI, local ollama).

```
> why does bus_schedule fail on hidden tests?
> read_file(bus_schedule/sol.py)
> grep(pattern="heapq", path=".")
> fix the sorting bug in the solution and rerun the tests
> [diff preview]  allow this action? (y/n/a)  y
> edited sol.py (1 replacement) ... exit code: 0, 23/23 passed
```

## Quickstart

```bash
# 1. install (uv recommended)
cd bilal_agent
uv sync                 # creates .venv, installs openai + rich

# 2. configure keys  (in the AGENT folder, not the target project)
cp .env.example .env    # then put your key in it

# 3. run inside any project
cd ~/your-project
uv run --project ~/path/to/bilal_agent bilal          # as a command
# or from the repo itself:
uv run main.py ~/your-project
```

### Install globally - one command from any terminal

```bash
cd bilal_agent
uv tool install .          # upgrade later with: uv tool install --force .

# then, from ANY directory:
bilal_agent                # current dir is the project root
bilal_agent ~/some/project # or point at a path explicitly
bilal_agent -p "summarize the architecture"   # one-shot
```

`uv tool install` puts the `bilal` / `bilal_agent` commands on your PATH (`~/.local/bin`).
For keys, a tool-installed build reads `~/.config/bilal/.env` (or put `config.toml` next to it).

First useful thing to do: `bilal_agent --init` inside a project creates an `AGENT.md` memory template.

## Configuration

`config.toml` (repo root, safe to commit - it contains no secrets, only env var names):

```toml
active = "zenmux"                    # switch provider here, or: bilal --provider openrouter

[providers.zenmux]
base_url = "https://zenmux.ai/api/v1"
model = "z-ai/glm-5.3-flash"
api_key_env = "ZENMUX_API_KEY"       # the key itself lives in .env
```

Keys go in `.env` (gitignored). `.env` lookup order: **the bilal_agent repo itself, then `~/.config/bilal/.env`, then the current directory, then the target project** - the first file that defines a key wins, and real environment variables always win. Config search order: `--config path` > `<project>/config.toml` > `./config.toml` > `~/.config/bilal/config.toml` > built-in defaults.

Useful limits: `agent.llm_timeout` (seconds per LLM request - raise it for slow providers), `agent.stream` (live token output; set `false` if your proxy mangles streamed tool calls), `agent.max_iterations` (hard stop per request), `context.max_tokens` (compaction trigger), `safety.command_timeout` (per shell command).

## Memory: the agent remembers your project

Two files stop the agent from re-exploring the same codebase every session:

| file | written by | holds |
|---|---|---|
| `.bilal/JOURNAL.md` | **the code**, automatically - one line per applied edit/rewrite/shell command, with timestamps and diff stats | the full history of what the agent changed |
| `AGENT.md` | **the agent** (prompted to keep it current) | durable knowledge: what the project is, commands, conventions, architecture decisions |

Both are injected into the system prompt at startup (journal: last 60 entries), so the agent arrives already knowing the project. The journal never costs you model tokens to write and it cannot lie - it is written by the tool layer, not the model. Add `.bilal/` to `.gitignore` if you do not want it versioned; commit it if you do.

## Usage

```bash
bilal_agent [PATH]              # REPL in project PATH (default: current dir)
bilal_agent -p "summarize the architecture"   # one-shot: single request, print, exit
bilal_agent -y                  # --yes: auto-approve risky actions (careful)
bilal_agent --provider local    # switch provider for this run
bilal_agent --model qwen2.5-coder:14b
```

(`bilal` is a shorter alias for the same command.)

REPL commands: `/help  /model  /todos  /context  /compact  /resume [n]  /undo  /clear  /quit`

### Sessions - stop where you are, continue later

Every turn is auto-saved to `.bilal/sessions/`. Pick up where you left off:

```bash
bilal_agent --continue        # (-c) resume the most recent session in this project
```

or type `/resume` inside the REPL to see the list (timestamp, turns, first request) and pick one. `/clear` starts a new session; the last 20 are kept per project.

### Sub-agents - heavy research, tiny context cost

The `task` tool spawns a **read-only sub-agent with its own fresh context window**. It greps, reads and explores as much as it needs, then returns only a dense report to the main conversation - which grows by exactly one tool result. Sub-agents cannot edit files, run commands, or spawn further sub-agents, and their token usage is rolled into your totals.

### Checkpoints - undo the agent's last turn

Before the first file mutation of every turn, the agent snapshots tracked files (`git stash create` - non-destructive, does not touch your tree or index). Unhappy with a turn's edits?

```
/undo        # tracked files restored to the pre-turn snapshot
```

Files *created* during the turn are left in place and listed. No git repo -> no checkpoint, and `/undo` says so.

### Live token & cost telemetry

- **Modern header** on startup: traffic lights, version + session tag, `● ONLINE | tty`, model chip, pid, and a rounded session card (workdir, policy, memory layers, shortcuts).
- **Terminal window title** updates live while tokens stream: `bilal · ↑12.3k ↓4.5k · ctx 38%`.
- **Economy line above every prompt**: `tok: 23k/60k (38%) · $0.07 · turn 5 · ctx: 4 files · ready`.
- **Message flow**: timestamped `❯ user` anchor, numbered `◇ thinking (1.2s · ~0.3k tokens)` plan blocks between tool rounds, `◆ agent::tools · 2 calls` trees with inline `✓ result · 0.4s` per call (sub-agent work nests inside), `◆ bilal_agent · ready` label before answers.
- **`/stats`** opens the full dashboard: cost card with budget bar, token totals, API call count + avg latency, per-turn consumption table, per-tool invocation table (calls/errors/avg), runtime environment.
- **`/cost`** is the quick money view; **`/reset-budget`** zeroes the session cost counter.

Cost estimation uses a built-in list-price table matched by model name; override it (and set a soft session budget that warns at 80%/100%) in config.toml:

```toml
[agent]
budget_usd = 2.0

[pricing]          # USD per 1M tokens - wins over the built-in table
input_per_m = 3.0
output_per_m = 15.0
```

`ctx %` is an estimate (chars/4) of the ledger vs `context.max_tokens` - when it nears 100%, compaction kicks in automatically.

### Safety model

| tools | behavior |
|---|---|
| read_file, grep, glob, list_dir, todo_write | auto-allowed (read-only) |
| edit_file, write_file, run_command | colored diff / command preview, then `y/n/a` prompt |

`a` = allow that tool for the rest of the session. `--yes` skips all prompts. Extra guardrails: every path is resolved against the project root (escapes refused), files must be read before they can be edited, and every action is printed.

## The tools

| tool | purpose |
|---|---|
| `read_file(path, offset, limit)` | numbered lines, big files come in slices with a continuation hint |
| `grep(pattern, path, file_glob, ignore_case)` | regex search, `path:line:` results, skips binaries/junk dirs |
| `glob(pattern)` | find files by pattern |
| `list_dir(path)` | one directory level |
| `edit_file(path, old_string, new_string, replace_all)` | exact string replacement + diff (never line numbers) |
| `write_file(path, content)` | create/overwrite (existing files must be read first) |
| `run_command(command, timeout)` | shell in the project root, output capped |
| `todo_write(todos)` | the model writes and maintains its own plan |
| `task(prompt, label)` | read-only sub-agent, fresh context, returns a report |

## Architecture (what to read, in order)

```
main.py                     shim -> uv run main.py
bilal_agent/
  cli.py        argv parsing, bootstrap checks, REPL + slash commands + sessions
  agent.py      THE LOOP: chat -> tool_calls? -> policy -> execute -> repeat
                + sub-agent runner + git checkpoints + session save/load + HUD
  policy.py     permission gate (safe vs ask, --yes, session allows)
  context.py    the messages ledger: token estimate + compaction + totals
  llm.py        OpenAI-compatible client, retries, summarize() for compaction
  system_prompt.py  identity + hard rules + environment + AGENT.md + journal
  tools/
    base.py     Tool dataclass, ToolError, AgentContext, path confinement, journal
    read.py     read_file / grep / glob / list_dir
    write.py    edit_file / write_file (+ read-before-edit enforcement)
    shell.py    run_command
    todo.py     todo_write
    task.py     task: read-only sub-agent with a fresh context window
  ui/render.py  rich TUI: modern header + session card, thinking/tools/response blocks,
                economy line, /stats + /cost dashboards, SubAgentUI (nested tool trees)
tests/          offline tests: tools + full loop with a scripted fake LLM
```

The loop in one paragraph: the API is stateless, so the agent's memory is just the `messages` ledger. Each turn: append the user message, call the model with the tool schemas; if the model returns `tool_calls`, run each one through the permission gate, append results as `role: "tool"` messages, and call again; when a response has no `tool_calls`, that content is the final answer. Tools never raise out of the loop - every failure becomes an `error: ...` string the model can read and retry. That's the whole agent.

Design rules worth keeping:

1. **Search, don't dump** - the model pulls context with grep/glob instead of the whole repo being preloaded.
2. **Edits are exact strings, not line numbers** - a failed match is a free self-check.
3. **Read before edit, enforced in code** - not just a prompt suggestion.
4. **Errors are strings** - the model self-corrects; the loop never crashes from a tool.
5. **Truncate with escape hatches** - every cap says how to get more (offset, narrower search).

## Extending: add a tool in ~10 lines

```python
# bilal_agent/tools/read.py - define it
my_tool = Tool(
    name="my_tool", description="What it does for the model.",
    parameters=P({"x": {"type": "string", "description": "..."}}, ["x"]),
    risk="safe",                      # or "ask" to require approval
    fn=lambda ctx, x: f"got {x}",
)
# tools/__init__.py - register it in build_tools()
```

The description and property descriptions are the model's only manual - write them like documentation for a new hire.

## Roadmap (v5 candidates)

- plan mode: read-only until you approve a plan
- background tasks: run dev servers / long tests without blocking
- post-edit lint hook: feed ruff/mypy errors back to the model automatically
- MCP tool servers, web fetch tool, per-role models (cheap summarizer)
- alternate-screen TUI: persistent bottom bar, tabbed views (repl/exec/context/stats)

## Development

```bash
uv run --with pytest pytest tests/ -q    # 75 offline tests, no network or keys needed
```
