"""todo_write: the model writes its own plan and keeps it updated. risk="safe"."""
from __future__ import annotations

from .base import P, AgentContext, Tool, ToolError

VALID_STATUS = {"pending", "in_progress", "completed"}
_CHECK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def todo_write(ctx: AgentContext, todos: list[dict]) -> str:
    if not isinstance(todos, list):
        raise ToolError("todos must be a list of {content, status} objects")
    if len(todos) > 20:
        raise ToolError("keep the plan small: max 20 items")
    cleaned: list[dict] = []
    for i, item in enumerate(todos):
        if not isinstance(item, dict) or "content" not in item:
            raise ToolError(f"item {i} must be an object with a 'content' field")
        status = str(item.get("status", "pending")).lower()
        if status not in VALID_STATUS:
            raise ToolError(f"item {i} status '{status}' not in {sorted(VALID_STATUS)}")
        cleaned.append({"content": str(item["content"])[:200], "status": status})

    ctx.todos = cleaned
    if ctx.ui is not None and hasattr(ctx.ui, "render_todos"):
        ctx.ui.render_todos(cleaned)
    lines = [f"{_CHECK[t['status']]} {t['content']}" for t in cleaned]
    return "todo list updated:\n" + "\n".join(lines)


todo_write_tool = Tool(
    name="todo_write",
    description=(
        "Write your current plan as a checklist (replaces the previous list). Use it for any "
        "task with 2+ steps; mark items in_progress/completed as you work so the user can follow."
    ),
    parameters=P(
        {
            "todos": {
                "type": "array",
                "description": "Full list of plan items.",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Short actionable step."},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                    },
                    "required": ["content", "status"],
                },
            }
        },
        ["todos"],
    ),
    risk="safe",
    fn=todo_write,
)
