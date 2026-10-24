"""Permission gate: every risky tool call passes through Policy.decide() first.

- risk="safe" tools (read/grep/glob/ls/todo) always allowed
- risk="ask" tools (write/edit/bash) prompt the user with a preview
- `--yes` auto-approves everything (still logged)
- answering `a` once allows that tool for the rest of the session
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .tools.base import Tool

if TYPE_CHECKING:  # pragma: no cover
    from .ui.render import UI


class Policy:
    def __init__(self, ui: "UI", confirm_edits: bool = True, auto_yes: bool = False):
        self.ui = ui
        self.confirm_edits = confirm_edits
        self.auto_yes = auto_yes
        self.always_allowed: set[str] = set()

    def decide(self, tool: Tool, args: dict) -> tuple[bool, str]:
        """Return (allowed, reason)."""
        if tool.risk == "safe":
            return True, "safe tool"
        if self.auto_yes:
            return True, "auto-approved (--yes)"
        if not self.confirm_edits:
            return True, "confirm_edits=false in config"
        if tool.name in self.always_allowed:
            return True, "allowed for this session"

        preview = self._preview(tool, args)
        answer = self.ui.confirm(f"Allow {tool.name}?", preview)
        if answer == "always":
            self.always_allowed.add(tool.name)
            return True, f"'{tool.name}' allowed for the rest of the session"
        if answer == "yes":
            return True, "approved by user"
        return False, "denied by user"

    @staticmethod
    def _preview(tool: Tool, args: dict) -> str:
        try:
            if tool.preview is not None:
                return tool.preview(args)
        except Exception as exc:  # noqa: BLE001 - preview must never crash the gate
            return f"(preview failed: {exc})\nargs: {args}"
        return "args: " + ", ".join(f"{k}={v!r}" for k, v in args.items())
