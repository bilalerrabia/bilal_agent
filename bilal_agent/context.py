"""The conversation ledger + token estimation + compaction.

The ledger is the model's only memory (stateless API). We keep it as plain
dicts exactly in OpenAI wire format. Compaction replaces the old head of the
conversation with one summary message, always cutting at a `user` message so
assistant tool_calls are never separated from their tool results.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .llm import LLMClient

COMPACT_TAG = "<compacted_session>"


class Context:
    def __init__(self, max_tokens: int = 60000, keep_recent: int = 8):
        self.max_tokens = max_tokens
        self.keep_recent = max(4, keep_recent)
        self.messages: list[dict] = []
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.turns = 0

    # ------------------------------------------------------------------ ledger
    def add(self, message: dict) -> None:
        self.messages.append(message)

    def track_usage(self, usage: dict | None) -> None:
        if usage:
            self.total_tokens_in += usage.get("prompt") or 0
            self.total_tokens_out += usage.get("completion") or 0

    def clear(self) -> None:
        self.messages = []
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.turns = 0

    # ----------------------------------------------------------------- sizing
    @staticmethod
    def _estimate_message(msg: dict) -> int:
        chars = len(str(msg.get("content") or ""))
        for tc in msg.get("tool_calls") or []:
            chars += len(tc.get("function", {}).get("arguments") or "")
        return max(1, chars // 4)

    def estimate(self) -> int:
        return sum(self._estimate_message(m) for m in self.messages)

    # -------------------------------------------------------------- compaction
    def _split_point(self) -> int | None:
        """Index of the user message that starts the kept tail, or None."""
        user_indexes = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
        if not user_indexes:
            return None
        window_start = max(0, len(self.messages) - self.keep_recent)
        for i in user_indexes:
            if i >= window_start and i > 0:
                return i
        # no user message (with non-empty history before it) inside the window
        last = user_indexes[-1]
        return last if last > 0 else None

    def needs_compaction(self) -> bool:
        return self.estimate() >= self.max_tokens

    def compact(self, llm: "LLMClient") -> str | None:
        """Summarize old history into one message. Returns summary text or None if skipped."""
        if len(self.messages) <= self.keep_recent + 2 or not self.needs_compaction():
            return None
        split = self._split_point()
        if split is None:
            return None
        old, recent = self.messages[:split], self.messages[split:]
        try:
            summary = llm.summarize(old)
        except Exception as exc:  # noqa: BLE001 - never crash the session on compaction
            return f"(compaction failed: {exc})"
        self.messages = [{"role": "user", "content": f"{COMPACT_TAG}\n{summary}"}, *recent]
        return summary
