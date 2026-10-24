"""Mutation tools: edit_file and write_file. Both are risk="ask".

Hard rules enforced here (the model is also told about them in the system prompt):
- read before write: an existing file must have been read via read_file first
- edit by exact string match, never line numbers; ambiguous matches are refused
"""
from __future__ import annotations

from .base import AgentContext, Tool, ToolError, unified_diff
from .read import read_file as _read_file_impl


def _require_read(ctx: AgentContext, p, display: str) -> None:
    if str(p) not in ctx.read_files:
        raise ToolError(
            f"'{display}' must be read before it is modified - call read_file on it first "
            "(this keeps edits grounded in the real file content)"
        )


# ------------------------------------------------------------------ edit_file


def edit_file(ctx: AgentContext, path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = ctx.resolve(path)
    if not p.is_file():
        raise ToolError(f"file not found: {path}")
    _require_read(ctx, p, path)
    if old_string == new_string:
        raise ToolError("old_string and new_string are identical - nothing to do")

    content = p.read_text(encoding="utf-8", errors="replace")
    count = content.count(old_string)
    if count == 0:
        raise ToolError(
            "old_string not found in file. Whitespace must match exactly - "
            "re-read the file and copy the text verbatim (include enough surrounding lines to be unique)"
        )
    if count > 1 and not replace_all:
        raise ToolError(
            f"old_string matches {count} times. Add surrounding context to make it unique, "
            "or pass replace_all=true to change every occurrence"
        )

    new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
    p.write_text(new_content, encoding="utf-8")
    diff = unified_diff(content, new_content, path)
    added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    ctx.log_modification("edit_file", f"{path} - {count if replace_all else 1} replacement(s), +{added}/-{removed}")
    return f"edited {path} ({count if replace_all else 1} replacement(s))\n{diff}"


def _edit_preview(ctx: AgentContext, args: dict) -> str:
    path = args.get("path", "?")
    p = ctx.resolve(path)
    if not p.is_file():
        return f"file not found: {path}"
    content = p.read_text(encoding="utf-8", errors="replace")
    old, new = args.get("old_string", ""), args.get("new_string", "")
    if old and old in content:
        new_content = content.replace(old, new) if args.get("replace_all") else content.replace(old, new, 1)
        return unified_diff(content, new_content, path)
    return f"(old_string not found in current file content)\npath: {path}"


edit_file_tool = Tool(
    name="edit_file",
    description=(
        "Replace an exact string inside a file. old_string must match the file content "
        "character-for-character and be unique (or pass replace_all=true). The file must be "
        "read with read_file first. Returns the unified diff of the change."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File relative to project root."},
            "old_string": {"type": "string", "description": "Exact text to replace (copy it from read_file output)."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false."},
        },
        "required": ["path", "old_string", "new_string"],
    },
    risk="ask",
    fn=edit_file,
    preview=_edit_preview,
)

# ------------------------------------------------------------------ write_file


def write_file(ctx: AgentContext, path: str, content: str) -> str:
    p = ctx.resolve(path)
    existed = p.exists()
    if existed:
        if not p.is_file():
            raise ToolError(f"'{path}' exists and is not a file")
        _require_read(ctx, p, path)
    old = p.read_text(encoding="utf-8", errors="replace") if existed else ""

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    ctx.read_files.add(str(p))

    rel = str(p.relative_to(ctx.root))
    verb = "overwrote" if existed else "created"
    ctx.log_modification("write_file", f"{rel} - {verb} ({len(content)} bytes)")
    if existed:
        diff = unified_diff(old, content, rel)
        return f"{verb} {rel} ({len(content)} bytes)\n{diff}"
    preview = content[:1500] + ("\n... (truncated preview)" if len(content) > 1500 else "")
    return f"created {rel} ({len(content)} bytes)\n{preview}"


def _write_preview(ctx: AgentContext, args: dict) -> str:
    path = args.get("path", "?")
    p = ctx.resolve(path)
    content = args.get("content", "")
    if p.exists() and p.is_file():
        old = p.read_text(encoding="utf-8", errors="replace")
        return unified_diff(old, content, path)
    shown = content[:1500] + ("\n... (truncated preview)" if len(content) > 1500 else "")
    return f"(new file)\n{shown}"


write_file_tool = Tool(
    name="write_file",
    description=(
        "Create a new file or fully overwrite an existing one (existing files must be read "
        "first - prefer edit_file for targeted changes to big files)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File relative to project root. Parent dirs are created."},
            "content": {"type": "string", "description": "Full file content to write."},
        },
        "required": ["path", "content"],
    },
    risk="ask",
    fn=write_file,
    preview=_write_preview,
)
