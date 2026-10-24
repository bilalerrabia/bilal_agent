"""Read-only tools: read_file, list_dir, glob, grep.

These give the model "eyes" it moves around the project. Nothing here mutates
state, so everything is risk="safe" and auto-allowed. All outputs are capped
with explicit hints telling the model how to get more (offset, refine pattern).
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from .base import MAX_LINE_LEN, P, SKIP_DIRS, AgentContext, Tool, ToolError

# ------------------------------------------------------------------ read_file


def read_file(ctx: AgentContext, path: str, offset: int = 1, limit: int = 2000) -> str:
    p = ctx.resolve(path)
    if not p.exists():
        raise ToolError(f"file not found: {path} (cwd is {ctx.root})")
    if p.is_dir():
        raise ToolError(f"'{path}' is a directory - use list_dir on it")
    ctx.read_files.add(str(p))  # marks the file as read (required before edit/write)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"cannot read '{path}': {exc}") from exc

    lines = text.splitlines()
    total = len(lines)
    offset = max(1, int(offset))
    limit = max(1, min(int(limit), 5000))

    if total == 0:
        return "(empty file)"

    start = offset - 1
    if start >= total:
        raise ToolError(f"offset {offset} is beyond end of file ({total} lines) - use offset <= {total}")

    chunk = lines[start : start + limit]
    numbered = [f"{i:>6}\t{ln[:MAX_LINE_LEN]}" for i, ln in enumerate(chunk, start=offset)]
    out = "\n".join(numbered)

    end_shown = offset + len(chunk) - 1
    notes = []
    if offset > 1 or end_shown < total:
        notes.append(f"showing lines {offset}-{end_shown} of {total}")
    if end_shown < total:
        notes.append(f"call again with offset={end_shown + 1} to continue")
    if notes:
        out += f"\n[{' - '.join(notes)}]"
    return out


read_file_tool = Tool(
    name="read_file",
    description=(
        "Read a text file from the project. Returns numbered lines (cat -n format). "
        "Big files are returned in slices: default shows lines 1-2000, the result tells "
        "you the next offset. You MUST read a file before editing or overwriting it."
    ),
    parameters=P(
        {
            "path": {"type": "string", "description": "Relative path from the project root."},
            "offset": {"type": "integer", "description": "1-based first line to show. Default 1."},
            "limit": {"type": "integer", "description": "Max lines to show. Default 2000."},
        },
        ["path"],
    ),
    risk="safe",
    fn=read_file,
)

# ------------------------------------------------------------------- list_dir


def list_dir(ctx: AgentContext, path: str = ".") -> str:
    p = ctx.resolve(path)
    if not p.exists():
        raise ToolError(f"directory not found: {path}")
    if not p.is_dir():
        raise ToolError(f"'{path}' is a file - use read_file on it")

    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    lines = []
    for e in entries[:500]:
        name = e.name + ("/" if e.is_dir() else "")
        lines.append(name)
    out = "\n".join(lines)
    if len(entries) > 500:
        out += f"\n[{len(entries) - 500} more entries hidden]"
    return out or "(empty directory)"


list_dir_tool = Tool(
    name="list_dir",
    description="List one directory level (dirs get a trailing /). Use it to explore structure.",
    parameters=P(
        {"path": {"type": "string", "description": "Directory relative to project root. Default '.'."}},
        [],
    ),
    risk="safe",
    fn=list_dir,
)

# ----------------------------------------------------------------------- glob


def glob_files(ctx: AgentContext, pattern: str) -> str:
    if pattern.startswith(("/", "~")) or ".." in pattern:
        raise ToolError("pattern must be relative to the project root (no '..', no absolute paths)")
    try:
        matches = sorted(ctx.root.glob(pattern))
    except (ValueError, NotImplementedError) as exc:
        raise ToolError(f"bad glob pattern '{pattern}': {exc}") from exc

    files = [m for m in matches if m.is_file()]
    if not files:
        return f"no files match pattern '{pattern}'"
    rels = [str(m.relative_to(ctx.root)) for m in files[:200]]
    out = "\n".join(rels)
    if len(files) > 200:
        out += f"\n[{len(files) - 200} more matches hidden - make the pattern more specific]"
    return out


glob_tool = Tool(
    name="glob",
    description=(
        "Find files by glob pattern relative to project root, e.g. 'src/**/*.py', "
        "'**/*.md', 'tests/test_*.py'. Returns up to 200 paths."
    ),
    parameters=P({"pattern": {"type": "string", "description": "Glob pattern, relative to root."}}, ["pattern"]),
    risk="safe",
    fn=glob_files,
)

# ----------------------------------------------------------------------- grep


def grep(
    ctx: AgentContext,
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    ignore_case: bool = False,
) -> str:
    if not pattern:
        raise ToolError("empty pattern")
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        raise ToolError(f"invalid regex '{pattern}': {exc}") from exc

    base = ctx.resolve(path)
    if not base.exists():
        raise ToolError(f"search path not found: {path}")

    if base.is_file():
        targets = [base]
    else:
        targets = []
        for dirpath, dirnames, filenames in os.walk(base, topdown=True):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in filenames:
                if file_glob and not fnmatch.fnmatch(fname, file_glob):
                    continue
                targets.append(Path(dirpath) / fname)

    max_matches = 100
    results: list[str] = []
    stopped = False
    for fpath in targets:
        if _is_binary(fpath):
            continue
        try:
            rel = fpath.relative_to(ctx.root)
        except ValueError:
            rel = fpath
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    if rx.search(line):
                        results.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                        if len(results) >= max_matches:
                            stopped = True
                            break
        except OSError:
            continue
        if stopped:
            break

    if not results:
        return f"no matches for pattern '{pattern}'" + (f" in {path}" if path != "." else "")
    out = "\n".join(results)
    if stopped:
        out += f"\n[stopped at {max_matches} matches - refine the pattern or narrow the path]"
    return out


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return b"\x00" in fh.read(1024)
    except OSError:
        return True


grep_tool = Tool(
    name="grep",
    description=(
        "Regex search across project files. Returns 'path:line: text' matches (max 100). "
        "PREFERRED over reading many files: search first, then read only the hits."
    ),
    parameters=P(
        {
            "pattern": {"type": "string", "description": "Python regex, e.g. 'def train_model', 'TODO', 'class .*Agent'."},
            "path": {"type": "string", "description": "File or directory to search. Default '.' (whole project)."},
            "file_glob": {"type": "string", "description": "Optional filename filter, e.g. '*.py'."},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive search. Default false."},
        },
        ["pattern"],
    ),
    risk="safe",
    fn=grep,
)
