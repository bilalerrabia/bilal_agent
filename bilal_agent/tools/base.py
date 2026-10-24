"""Tool primitives shared by every tool module.

- ToolError: raised inside tool functions; dispatch converts it to an "error: ..."
  string so the model can read it and self-correct (never crashes the loop).
- Tool: name + description + JSON schema + risk level + function (+ optional
  preview used by the permission prompt).
- AgentContext: shared state handed to every tool closure (project root, files
  the agent has already read, config, todos) + the modification journal.
"""
from __future__ import annotations

import datetime
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Directories never searched by grep/glob walks.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".env",
    "dist", "build", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    ".idea", ".vscode", "target", ".next", ".bilal",
}

MAX_LINE_LEN = 2000


class ToolError(Exception):
    """Deliberate, model-readable failure."""


@dataclass
class AgentContext:
    root: Path
    config: Any
    ui: Any = None
    read_files: set[str] = field(default_factory=set)  # resolved paths read via read_file
    todos: list[dict] = field(default_factory=list)

    def resolve(self, path: str) -> Path:
        """Resolve `path` inside the project root; refuse anything that escapes it."""
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        p = p.resolve()
        if not (p == self.root or p.is_relative_to(self.root)):
            raise ToolError(f"path '{path}' escapes the project root ({self.root})")
        return p

    # -------------------------------------------------------------- journal
    @property
    def journal_path(self) -> Path:
        return self.root / ".bilal" / "JOURNAL.md"

    def log_modification(self, tool: str, detail: str) -> None:
        """Append one line to .bilal/JOURNAL.md - the persistent record of every
        change the agent ever applied. Written by the CODE (not the model), so it
        never lies and costs zero tokens. Best-effort: never raises."""
        try:
            self.journal_path.parent.mkdir(exist_ok=True)
            if not self.journal_path.exists():
                self.journal_path.write_text(
                    "# Modification journal\n\n"
                    "Auto-written by bilal_agent - one line per action, newest last.\n"
                    "Next sessions read this (plus AGENT.md) instead of re-exploring\n"
                    "the whole project from scratch.\n\n",
                    encoding="utf-8",
                )
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            with self.journal_path.open("a", encoding="utf-8") as f:
                f.write(f"- {stamp} [{tool}] {detail}\n")
        except OSError:
            pass  # journaling must never break a tool


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema - "required" lives INSIDE this dict
    risk: str  # "safe" | "ask"
    fn: Callable[..., str]
    preview: Callable[[dict], str] | None = None  # rendered by the permission prompt

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def unified_diff(old: str, new: str, rel_path: str, max_chars: int = 8000) -> str:
    """Unified diff text between two file versions (rendered colored by the UI)."""
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
    )
    text = "".join(diff)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (diff truncated)"
    return text.rstrip("\n") or "(no changes)"


def truncate_output(text: str, max_chars: int = 10000, label: str = "output") -> str:
    """Head+tail truncation for long outputs, with an explicit note."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n... [{label} truncated: {len(text) - max_chars} chars hidden] ...\n"
        + text[-half:]
    )


def _params(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required}


P = _params  # short alias used by tool definitions
