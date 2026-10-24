"""Shell tool: run_command. risk="ask" - the full command is shown before running."""
from __future__ import annotations

import subprocess

from .base import P, AgentContext, Tool, ToolError, truncate_output

MAX_SECTION = 8000


def run_command(ctx: AgentContext, command: str, timeout: int | None = None) -> str:
    if not command.strip():
        raise ToolError("empty command")
    timeout = int(timeout or ctx.config.command_timeout)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx.root),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"command timed out after {timeout}s - pass a bigger timeout or fix the command") from exc
    except OSError as exc:
        raise ToolError(f"failed to run command: {exc}") from exc

    sections = [f"exit code: {proc.returncode}"]
    if proc.stdout:
        sections.append("--- stdout ---\n" + truncate_output(proc.stdout.rstrip("\n"), MAX_SECTION, "stdout"))
    if proc.stderr:
        sections.append("--- stderr ---\n" + truncate_output(proc.stderr.rstrip("\n"), MAX_SECTION, "stderr"))
    ctx.log_modification("shell", f"`{command.strip()[:100]}` (exit {proc.returncode})")
    return "\n".join(sections)


def _run_preview(ctx: AgentContext, args: dict) -> str:
    return f"$ {args.get('command', '?')}"


run_command_tool = Tool(
    name="run_command",
    description=(
        "Run a shell command inside the project root (build, tests, git, pip...). "
        "stdout/stderr are capped; long-running commands hit the timeout. "
        "Use git commands to inspect history; never force-push."
    ),
    parameters=P(
        {
            "command": {"type": "string", "description": "The shell command to run."},
            "timeout": {"type": "integer", "description": "Seconds before the command is killed. Default from config (60)."},
        },
        ["command"],
    ),
    risk="ask",
    fn=run_command,
    preview=_run_preview,
)
