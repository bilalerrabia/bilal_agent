"""System prompt builder: identity + hard rules + live environment + project memory."""
from __future__ import annotations

import datetime
import platform
import subprocess
from pathlib import Path


def _git_branch(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "branch", "--show-current"],
            capture_output=True, text=True, timeout=5,
        )
        branch = out.stdout.strip()
        return branch or "(no branch)"
    except Exception:  # noqa: BLE001
        return "(not a git repo)"


def _top_level(root: Path) -> str:
    try:
        entries = sorted(root.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        names = [e.name + ("/" if e.is_dir() else "") for e in entries[:60]]
        return "\n".join(names) if names else "(empty)"
    except Exception:  # noqa: BLE001
        return "(unreadable)"


def _journal_tail(root: Path, max_lines: int = 60) -> str:
    """Last N entry lines of .bilal/JOURNAL.md (the auto-written change log)."""
    p = root / ".bilal" / "JOURNAL.md"
    if not p.is_file():
        return ""
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    body = [ln for ln in lines if not ln.startswith("#")]  # drop the header block
    if not body:
        return ""
    tail = body[-max_lines:]
    note = "" if len(body) <= max_lines else f"... ({len(body) - max_lines} older entries omitted - read the file for more)\n"
    return note + "\n".join(tail)


def build_system_prompt(root: Path, extra_rules: str = "") -> str:
    agent_md = ""
    md_path = root / "AGENT.md"
    if md_path.is_file():
        try:
            agent_md = md_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            agent_md = ""
        if len(agent_md) > 6000:
            agent_md = agent_md[:6000] + "\n... (AGENT.md truncated)"

    journal = _journal_tail(root)

    today = datetime.date.today().isoformat()
    parts = [
        # identity + goal
        "You are bilal_agent, an expert AI pair programmer running in the user's terminal. "
        "You help with the user's codebase: exploring it, explaining it, and modifying it. "
        "You are precise, pragmatic and concise.",

        # environment
        f"<environment>\n"
        f"project_root: {root}\n"
        f"date: {today}\n"
        f"os: {platform.system()} {platform.release()} (python {platform.python_version()})\n"
        f"git_branch: {_git_branch(root)}\n"
        f"top-level entries:\n{_top_level(root)}\n"
        f"</environment>",

        # hard rules - how the agent must behave
        "<rules>\n"
        "1. Use your tools to act. Never claim you read or changed a file unless a tool call "
        "actually did it. NEVER invent tool outputs (no fake Observation lines) - results come "
        "only from the system after your tool calls.\n"
        "2. Reuse memory before you explore: the <project_memory> and <journal> sections below "
        "were written by your previous sessions - they often answer questions without reading "
        "files again. Explore with grep/glob/read_file only for what they do not cover.\n"
        "3. You MUST call read_file on a file before edit_file or write_file on it.\n"
        "4. edit_file old_string must be copied EXACTLY from the file (whitespace matters) and "
        "must be unique - include enough surrounding lines. If a tool returns 'error: ...', "
        "fix your arguments and retry instead of apologizing blindly.\n"
        "5. Keep edits minimal and focused; do not reformat unrelated code.\n"
        "6. For multi-step work, write a plan with todo_write and keep statuses updated.\n"
        "7. Dangerous shell (rm -rf, git push --force, dropping databases) is forbidden.\n"
        "8. Keep memory current: after finishing significant work (new module, command learned, "
        "convention discovered, architecture decision), update AGENT.md with the durable facts. "
        "Every applied edit is auto-recorded in .bilal/JOURNAL.md - do NOT edit the journal, "
        "only AGENT.md.\n"
        "9. Final answers: concise markdown. For code questions show the relevant snippet, "
        "path and line numbers. Explain what you did, then stop.\n"
        "</rules>",

        # tool tips
        "<tool_tips>\n"
        "- read_file: numbered lines; big files come in slices, follow the offset hint.\n"
        "- grep: returns path:line matches; narrow with path= or file_glob=.\n"
        "- edit_file: exact string replacement with diff preview; the user approves it.\n"
        "- run_command: runs in the project root; output is capped.\n"
        "- task: delegate heavy read-only exploration (broad greps, many-file reads) to a "
        "sub-agent with its own context; it returns a dense report. Prefer it when research "
        "would take many tool rounds.\n"
        "</tool_tips>",
    ]

    if agent_md:
        parts.append(f"<project_memory>\n{agent_md}\n</project_memory>")
    if journal:
        parts.append(
            "<journal>\n"
            "(auto-written log of modifications applied to this project by previous "
            "sessions - newest last)\n"
            f"{journal}\n"
            "</journal>"
        )
    if extra_rules:
        parts.append(extra_rules)
    return "\n\n".join(parts)


# Appended to the system prompt of every Task-tool sub-agent.
SUBAGENT_RULES = (
    "<subagent_rules>\n"
    "You are a SUB-AGENT spawned by the main agent for focused, read-only research.\n"
    "- You have ONLY read tools: read_file, grep, glob, list_dir.\n"
    "- Be fast and cheap: grep before reading, read only what you need, skip what is not asked.\n"
    "- NEVER ask the user questions - the user cannot see you. If something is missing, say\n"
    "  what you could not find in the final report.\n"
    "- Your final message is a REPORT for the main agent: dense, factual, under 400 words,\n"
    "  with file paths and line numbers it can act on directly.\n"
    "</subagent_rules>"
)
