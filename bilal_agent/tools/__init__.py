"""Tool registry - the agent's complete capability set.

build_tools(ctx) closes every tool over the SAME AgentContext, so read/edit
state (ctx.read_files) is shared across tools: edit_file knows whether
read_file was called on that path.
"""
from __future__ import annotations

from .base import AgentContext, Tool, ToolError
from .read import glob_tool, grep_tool, list_dir_tool, read_file_tool
from .shell import run_command_tool
from .todo import todo_write_tool
from .write import edit_file_tool, write_file_tool

__all__ = ["AgentContext", "Tool", "ToolError", "ToolRegistry", "build_tools"]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            known = ", ".join(sorted(self._tools))
            raise ToolError(f"unknown tool '{name}' - available tools: {known}")
        return self._tools[name]

    def remove(self, name: str) -> None:
        """Drop a tool (used to strip mutating tools from read-only sub-agents)."""
        self._tools.pop(name, None)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    @property
    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]


def build_tools(ctx: AgentContext) -> ToolRegistry:
    reg = ToolRegistry()

    def bind(tool: Tool) -> None:
        tool.fn = _bind_fn(tool.fn, ctx)
        if tool.preview is not None:
            tool.preview = _bind_fn(tool.preview, ctx)
        reg.add(tool)

    for tool in (
        read_file_tool,
        list_dir_tool,
        glob_tool,
        grep_tool,
        todo_write_tool,
        edit_file_tool,
        write_file_tool,
        run_command_tool,
    ):
        # dataclass instances are shared module-level objects: bind onto copies
        bind(_copy_tool(tool))
    return reg


def _copy_tool(tool: Tool) -> Tool:
    import dataclasses

    return dataclasses.replace(tool)


def _bind_fn(fn, ctx: AgentContext):
    def wrapped(*args, **kwargs):
        return fn(ctx, *args, **kwargs)

    wrapped.__name__ = getattr(fn, "__name__", "tool_fn")
    return wrapped
