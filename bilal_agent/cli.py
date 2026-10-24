"""Command-line entry: argument parsing, bootstrap checks, REPL with slash commands."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_config, load_dotenv, require_api_key
from .context import COMPACT_TAG

HELP = """[bold]commands[/]
  /help          show this help
  /stats         full telemetry dashboard (tokens, cost, latency, tools)
  /cost          quick cost & token summary
  /reset-budget  zero the session cost counter
  /diff          show working-tree changes vs HEAD (git)
  /model         show active provider / model / context size
  /todos         show the current plan
  /context       show estimated ledger size and recent messages
  /compact       force context compaction now
  /resume [n]    list saved sessions and jump back into one
  /undo          roll back this turn's file changes (git checkpoint)
  /banner        the classic big block-letter banner
  /clear         wipe conversation memory (files on disk are untouched)
  /quit          exit (also ctrl+d)"""

AGENT_MD_TEMPLATE = """# AGENT.md - project memory for bilal_agent

Facts below are injected into the agent's system prompt on every start.
Keep it short, factual and current - the agent can edit this file too.
(Auto-recorded edits live in .bilal/JOURNAL.md - no need to duplicate them here.)

## Project
- <what this project is>

## Commands
- run tests: <command>
- run the app: <command>

## Conventions
- <style rules the agent must respect>
"""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="bilal",
        description="bilal_agent - an AI coding agent in your terminal",
    )
    ap.add_argument("root", nargs="?", default=".", help="project root (default: current dir)")
    ap.add_argument("-p", "--prompt", help="one-shot: run this single request and exit")
    ap.add_argument("-y", "--yes", action="store_true", help="auto-approve all risky actions")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true",
                    help="resume the most recent session in this project")
    ap.add_argument("--provider", help="override active provider from config.toml")
    ap.add_argument("--model", help="override model name")
    ap.add_argument("--config", help="path to config.toml")
    ap.add_argument("--max-iterations", type=int, help="override agent.max_iterations")
    ap.add_argument("--init", action="store_true", help="create an AGENT.md template and exit")
    ap.add_argument("--version", action="store_true", help="print version and exit")
    return ap.parse_args(argv)


def _env_candidates(root: Path, repo_dir: Path | None = None) -> list[Path]:
    """Where we look for .env, in priority order (first file defining a key wins).

    The AGENT's own folders come first - the key lives with the agent, not inside
    every project it opens. ~/.config/bilal/.env covers `uv tool install` builds,
    where the repo dir no longer exists next to the code.
    """
    candidates = []
    if repo_dir is not None:
        candidates.append(repo_dir / ".env")
    candidates += [
        Path.home() / ".config" / "bilal" / ".env",
        Path.cwd() / ".env",
        root / ".env",  # the target project - last, not first
    ]
    return candidates


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.version:
        print(f"bilal_agent {__version__}")
        return 0

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"error: project root does not exist: {root}", file=sys.stderr)
        return 1

    # --init works offline, no config needed
    if args.init:
        target = root / "AGENT.md"
        if target.exists():
            print(f"AGENT.md already exists: {target}")
        else:
            target.write_text(AGENT_MD_TEMPLATE, encoding="utf-8")
            print(f"wrote {target}")
        return 0

    from .ui.render import UI  # rich only needed from here on

    ui = UI(auto_yes=args.yes)
    repo_dir = Path(__file__).resolve().parent.parent
    env_candidates = _env_candidates(root, repo_dir)
    for env_path, keys in load_dotenv(env_candidates):
        if keys:
            ui.info(f"env: loaded {env_path} ({', '.join(sorted(keys))})")
    try:
        cfg = load_config(args.config, args.provider, args.model, auto_yes=args.yes, root=root)
    except (FileNotFoundError, ValueError) as exc:
        ui.error(f"config error: {exc}")
        return 1

    missing = require_api_key(cfg)
    if missing:
        searched = "\n  ".join(str(p) for p in env_candidates)
        ui.error(
            f"missing API key: set {missing} in .env or environment (see .env.example)\n"
            f"looked in:\n  {searched}\n"
            f"or export it directly:  export {missing}=<your-key>"
        )
        return 1

    if args.max_iterations:
        cfg.max_iterations = args.max_iterations

    # heavy imports
    from .agent import Agent
    from .llm import LLMClient, LLMError

    try:
        llm = LLMClient(cfg)
    except Exception as exc:  # noqa: BLE001
        ui.error(f"failed to init LLM client ({cfg.provider.base_url}): {exc}")
        return 1

    agent = Agent(cfg, llm, ui, root)
    ui.banner(__version__, cfg.provider.name, cfg.provider.model, root, cfg.auto_yes)

    if args.cont:
        sessions = agent.list_sessions(limit=1)
        if sessions and agent.load_session(sessions[0]["path"]):
            ui.info(f"resumed session {agent.session_id} "
                    f"({len(agent.context.messages)} messages, turn {agent.context.turns})")
        else:
            ui.info("no previous session found - starting fresh")

    # ------------------------------------------------------------- one-shot
    if args.prompt:
        try:
            agent.run(args.prompt)
        except LLMError as exc:
            ui.error(str(exc))
            return 1
        except KeyboardInterrupt:
            ui.warn("\ninterrupted")
            return 130
        finally:
            agent.save_session()
        return 0

    # ----------------------------------------------------------------- REPL
    while True:
        try:
            ui.hud(agent.hud_line())
            user_input = ui.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            agent.save_session()
            ui.info("\nbye (session saved)")
            return 0
        if not user_input:
            continue
        if user_input.startswith("/"):
            cmd, _, rest = user_input.partition(" ")
            if cmd in ("/quit", "/exit", "/q"):
                agent.save_session()
                ui.info("bye (session saved)")
                return 0
            elif cmd == "/help":
                ui.console.print(HELP)
            elif cmd == "/model":
                ui.info(
                    f"provider={cfg.provider.name} model={cfg.provider.model} "
                    f"base_url={cfg.provider.base_url} yes={'on' if cfg.auto_yes else 'off'}"
                )
            elif cmd == "/todos":
                ui.render_todos(agent.ctx.todos)
            elif cmd == "/stats":
                ui.telemetry(agent)
            elif cmd == "/cost":
                ui.cost_card(agent)
            elif cmd == "/reset-budget":
                was = agent.stats["cost"]
                agent.stats["cost"] = 0.0
                agent._budget_stage = 0
                ui.info(f"budget counter zeroed (was ${was:.2f})")
            elif cmd == "/diff":
                msg = agent.diff_summary()
                (ui.warn if msg.startswith("not a git repo") else ui.info)(msg)
            elif cmd == "/banner":
                ui.banner_classic(__version__, cfg.provider.name, cfg.provider.model)
            elif cmd == "/context":
                est = agent.context.estimate()
                recent = sum(1 for m in agent.context.messages if not m.get("content", "").startswith(COMPACT_TAG))
                ui.info(f"messages={len(agent.context.messages)} (~{est} tokens est., compacts at {cfg.max_context_tokens})")
            elif cmd == "/compact":
                summary = agent.context.compact(llm)
                ui.info("compacted:\n" + (summary or "(nothing to compact)"))
            elif cmd == "/resume":
                _resume(agent, ui, rest.strip())
            elif cmd == "/undo":
                msg = agent.undo_checkpoint()
                (ui.warn if msg.startswith("error") or msg.startswith("no checkpoint") else ui.info)(msg)
            elif cmd == "/clear":
                agent.context.clear()
                agent.session_id = None
                agent.last_saved_path = None
                ui.info("conversation memory cleared (next save starts a new session)")
            else:
                ui.warn(f"unknown command {cmd} - {HELP.splitlines()[1].strip()}")
            continue

        try:
            agent.run(user_input)
        except LLMError as exc:
            ui.error(str(exc))
        except KeyboardInterrupt:
            ui.warn("\n(turn interrupted - partial state kept)")
        finally:
            agent.save_session()  # auto-save the ledger after every turn


def _resume(agent, ui, arg: str) -> None:
    """`/resume` lists sessions; `/resume n` jumps straight to entry n (1-based)."""
    sessions = agent.list_sessions()
    if not sessions:
        ui.info("no saved sessions yet - they are written after every turn")
        return
    if arg.isdigit():
        idx = int(arg) - 1
    else:
        for i, s in enumerate(sessions, 1):
            ui.console.print(
                f"[cyan]{i:>2}[/] [dim]{s['saved_at']}[/]  turn {s['turns']:<3}  {s['first']}"
            )
        try:
            raw = ui.console.input("[bold]resume # (enter=cancel)[/]")
        except (EOFError, KeyboardInterrupt):
            return
        idx = int(raw) - 1 if raw.strip().isdigit() else -1
    if 0 <= idx < len(sessions) and agent.load_session(sessions[idx]["path"]):
        ui.info(f"resumed session {agent.session_id} "
                f"({len(agent.context.messages)} messages, turn {agent.context.turns})")
    else:
        ui.warn("nothing resumed")
