"""Thin OpenAI-compatible chat wrapper.

- lazy `openai` import so tools/policy/context are testable without it
- converts SDK responses into plain dicts (the exact ledger format we store)
- streams tokens live (on_delta hook) so long requests never look frozen
- dedicated llm_timeout (read timeout) separate from the shell command timeout
- retries transient provider errors; timeouts get fewer, cheaper retries
"""
from __future__ import annotations

import time
from typing import Any, Callable, Iterable

from .config import Config


class LLMError(RuntimeError):
    pass


def _is_timeout(exc: Exception) -> bool:
    if type(exc).__name__ in {"APITimeoutError", "ReadTimeout", "ConnectTimeout"}:
        return True
    return "timed out" in str(exc).lower()


def _is_transient(exc: Exception) -> bool:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if name in {"APIConnectionError", "InternalServerError", "RateLimitError"}:
        return True
    return status in {408, 409, 429, 500, 502, 503, 504}


def _usage_to_dict(u: Any) -> dict:
    """Normalize an OpenAI usage object; keeps cached prompt tokens when the
    provider reports them (prompt_tokens_details.cached_tokens)."""
    d = {
        "prompt": getattr(u, "prompt_tokens", 0) or 0,
        "completion": getattr(u, "completion_tokens", 0) or 0,
    }
    details = getattr(u, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    if cached:
        d["cached"] = cached
    return d


def _message_to_dict(message: Any) -> dict:
    """OpenAI ChatCompletionMessage -> plain dict in ledger format (non-streaming path)."""
    out: dict = {"role": "assistant", "content": message.content or ""}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
    return out


def _assemble_stream(chunks: Iterable[Any], on_delta: Callable[[str], None] | None = None) -> tuple[dict, dict | None, str | None]:
    """Fold a streaming response into (assistant_message_dict, usage|None, finish_reason|None).

    Pure function over chunk-like objects -> fully unit-testable offline.
    Handles: content deltas, tool_call deltas split across chunks (arguments
    arrive in pieces), parallel tool calls (index-keyed), usage-only final chunk.
    """
    content_parts: list[str] = []
    calls: dict[int, dict] = {}
    usage: dict | None = None
    finish: str | None = None

    for chunk in chunks:
        u = getattr(chunk, "usage", None)
        if u is not None and getattr(u, "prompt_tokens", None):
            usage = _usage_to_dict(u)

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        choice = choices[0]
        fr = getattr(choice, "finish_reason", None)
        if fr:
            finish = fr
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue

        piece = getattr(delta, "content", None)
        if piece:
            content_parts.append(piece)
            if on_delta:
                on_delta(piece)

        for tc in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tc, "index", 0) or 0
            slot = calls.setdefault(idx, {"id": "", "name": "", "args": ""})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None) and not slot["name"]:
                    slot["name"] = fn.name
                if getattr(fn, "arguments", None):
                    slot["args"] += fn.arguments

    msg: dict = {"role": "assistant", "content": "".join(content_parts)}
    if calls:
        msg["tool_calls"] = [
            {
                "id": slot["id"] or f"call_{idx}",
                "type": "function",
                "function": {"name": slot["name"], "arguments": slot["args"]},
            }
            for idx, slot in sorted(calls.items())
        ]
    return msg, usage, finish


class LLMClient:
    def __init__(self, cfg: Config):
        from openai import OpenAI  # lazy: keep the rest of the package import-light

        try:
            import httpx

            timeout: Any = httpx.Timeout(float(cfg.llm_timeout), connect=15.0)
        except ImportError:  # pragma: no cover - httpx ships with openai
            timeout = float(cfg.llm_timeout)

        api_key = "no-key-required"
        if cfg.provider.api_key_env:
            import os

            api_key = os.environ.get(cfg.provider.api_key_env) or "missing-key"
        self.client = OpenAI(
            base_url=cfg.provider.base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,  # we retry ourselves so we can log it
        )
        self.timeout = timeout
        self.model = cfg.provider.model
        self.max_output_tokens = cfg.max_output_tokens
        self.stream = cfg.stream
        self.temperature = getattr(cfg, "temperature", None)
        self.on_delta: Callable[[str], None] | None = None  # set by the Agent for live output

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[dict, dict | None, str | None]:
        """One completion. Returns (assistant_message_dict, usage_dict_or_None, finish_reason_or_None)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.max_output_tokens:
            kwargs["max_tokens"] = self.max_output_tokens
        if getattr(self, "temperature", None) is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                if self.stream:
                    chunks = self.client.chat.completions.create(**kwargs, stream=True)
                    return _assemble_stream(chunks, self.on_delta)
                return self._request_once(kwargs)
            except Exception as exc:  # noqa: BLE001 - providers raise many shapes
                last_exc = exc
                is_timeout = _is_timeout(exc)
                # a timed-out big request is retried once, not 4 times
                limit = 2 if is_timeout else 4
                if (not _is_transient(exc) and not is_timeout) or attempt + 1 >= limit:
                    break
                time.sleep(2.0 if is_timeout else 1.5 * (attempt + 1))

        if last_exc is not None and _is_timeout(last_exc):
            raise LLMError(
                "LLM request timed out. The provider/proxy is slow or the request is huge. "
                f"Try: raise agent.llm_timeout in config.toml (now {self._timeout_text()}), "
                "set agent.stream=false there if the proxy dislikes streaming, "
                "or split the task into smaller requests."
            ) from last_exc
        raise LLMError(f"LLM request failed: {last_exc}") from last_exc

    def _timeout_text(self) -> str:
        read = getattr(self.timeout, "read", None)
        return f"{int(read)}s" if read else f"{self.timeout}s"

    def _request_once(self, kwargs: dict) -> tuple[dict, dict | None, str | None]:
        resp = self.client.chat.completions.create(**kwargs)
        msg = _message_to_dict(resp.choices[0].message)
        finish = getattr(resp.choices[0], "finish_reason", None)
        usage = None
        if getattr(resp, "usage", None):
            usage = _usage_to_dict(resp.usage)
        return msg, usage, finish

    # ------------------------------------------------------------- summarize
    def summarize(self, messages: list[dict]) -> str:
        """Compress a chunk of the conversation into a dense summary (used by compaction)."""
        saved_delta, self.on_delta = self.on_delta, None  # never stream summaries to the terminal
        try:
            sys = (
                "You compress a coding-agent session transcript into a dense state summary. "
                "Keep verbatim: file paths, function/class names, key decisions, current plan, "
                "pending next steps. Drop small talk and redundant tool output. Max 350 words."
            )
            transcript = _render_for_summary(messages)
            resp, _, _ = self.chat(
                [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": f"Compress this session:\n\n{transcript}"},
                ],
                tools=None,
            )
        finally:
            self.on_delta = saved_delta
        return resp["content"].strip() or "(empty summary)"


def _render_for_summary(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "tool":
            content = str(content)[:400]
            lines.append(f"[tool result] {content}")
        elif m.get("tool_calls"):
            calls = ", ".join(
                f"{tc['function']['name']}({tc['function']['arguments'][:120]})"
                for tc in m["tool_calls"]
            )
            if content:
                lines.append(f"[assistant] {str(content)[:400]}")
            lines.append(f"[assistant tool calls] {calls}")
        else:
            lines.append(f"[{role}] {str(content)[:600]}")
    return "\n".join(lines)
