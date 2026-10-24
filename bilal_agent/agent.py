"""The Agent: model + tools + while loop + permission gate + ledger.

One run() = one user request. The loop:
  1. maybe compact the ledger
  2. append the user message
  3. call the LLM with tool schemas
  4. no tool_calls  -> that content is the final answer, done
  5. tool_calls     -> policy gate -> execute -> append tool results -> goto 3

Safety details that matter:
- the system message is prepended per-request but never stored in the ledger
- if the user hits Ctrl+C mid-turn, every pending tool_call still gets a
  synthetic tool result ("interrupted") so the ledger stays valid for the API
- tools can never raise out of the loop: every failure becomes a string
- the first mutating tool of a turn snapshots tracked files into a git
  checkpoint so /undo can roll the turn back
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import subprocess
import time
from pathlib import Path

from .config import Config, resolve_pricing
from .context import Context
from .llm import LLMClient, LLMError
from .policy import Policy
from .system_prompt import SUBAGENT_RULES, build_system_prompt
from .tools import AgentContext, Tool, ToolError, build_tools
from .tools.task import task_tool

# tools that can change the working tree - they trigger the git checkpoint
MUTATING_TOOLS = {"edit_file", "write_file", "run_command"}


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class Agent:
    def __init__(self, cfg: Config, llm: LLMClient, ui, root: Path, extra_rules: str = "",
                 read_only: bool = False):
        self.cfg = cfg
        self.llm = llm
        self.ui = ui
        self.ctx = AgentContext(root=root, config=cfg, ui=ui)
        self.registry = build_tools(self.ctx)
        if read_only:
            # sub-agents: research only - no writes, no shell, no nested tasks
            for name in ("edit_file", "write_file", "run_command", "todo_write"):
                self.registry.remove(name)
        else:
            t = dataclasses.replace(task_tool)
            t.fn = self._run_subagent
            self.registry.add(t)
        self.policy = Policy(ui, confirm_edits=cfg.confirm_edits, auto_yes=cfg.auto_yes)
        self.context = Context(max_tokens=cfg.max_context_tokens, keep_recent=cfg.keep_recent)
        # ------------------------------------------------------------- telemetry
        # session-lifetime stats for the HUD, /cost and /stats. Tokens live in
        # Context; money/latency/per-tool/per-turn live here. Sub-agent stats
        # are rolled up into the parent's so the dashboard sees everything.
        self.stats: dict = {
            "api_calls": 0, "latency_sum": 0.0, "cached": 0, "cost": 0.0,
            "turns": [], "tools": {},
        }
        self._pin, self._pout, self._psrc = resolve_pricing(cfg)
        self._budget_stage = 0        # 0 silent, 1 warned 80%, 2 warned 100%
        self._turn_tool_calls = 0
        self._turn_mutations = 0
        self._last_chat_secs = 0.0
        self.system = build_system_prompt(root, extra_rules=extra_rules)
        self._streamed = False
        self._stream_chars = 0
        self._last_hud = 0.0
        llm.on_delta = self._handle_delta  # live token output while streaming
        # per-turn git checkpoint (see _ensure_checkpoint)
        self._checkpoint_sha: str | None = None
        # session persistence
        self.session_id: str | None = None
        self.last_saved_path: Path | None = None

    def _handle_delta(self, text: str) -> None:
        if not self._streamed:
            self._streamed = True
            self.ui.stream_start()
        self.ui.delta(text)
        self._stream_chars += len(text)
        now = time.monotonic()
        if now - self._last_hud >= 0.5:  # live window-title update, throttled
            self._last_hud = now
            self._update_hud(live_out_chars=self._stream_chars)

    def _update_hud(self, live_out_chars: int = 0) -> None:
        est = self.context.estimate()
        out = self.context.total_tokens_out + live_out_chars // 4
        pct = int(100 * est / self.cfg.max_context_tokens) if self.cfg.max_context_tokens else 0
        self.ui.set_title(f"bilal \u2191{_fmt_tokens(self.context.total_tokens_in)} "
                          f"\u2193{_fmt_tokens(out)} \u00b7 ctx {pct}%")

    def hud_line(self) -> str:
        """The live economy line shown above each REPL prompt (mockup bottom bar):
        tok: est/cap (pct) · $cost · turn N · ctx: N files · ready"""
        est = self.context.estimate()
        cap = self.cfg.max_context_tokens
        pct = int(100 * est / cap) if cap else 0
        cost = f"${self.stats['cost']:.2f}"
        budget = getattr(self.cfg, "budget_usd", None)
        if budget:
            cost += f"/${budget:.2f}"
        parts = [f"tok: {_fmt_tokens(est)}/{_fmt_tokens(cap)} ({pct}%)", cost,
                 f"turn {self.context.turns}"]
        files = len(getattr(self.ctx, "read_files", ()) or ())
        if files:
            parts.append(f"ctx: {files} files")
        parts.append("saved" if self.last_saved_path else "ready")
        return "  ·  ".join(parts)

    # ------------------------------------------------------------- telemetry
    def _account_usage(self, usage: dict | None) -> None:
        """Money + cache accounting for one API call, plus budget warnings."""
        if not usage:
            return
        cost = ((usage.get("prompt") or 0) / 1e6 * self._pin
                + (usage.get("completion") or 0) / 1e6 * self._pout)
        self.stats["cost"] += cost
        self.stats["cached"] += usage.get("cached") or 0
        self._check_budget()

    def _check_budget(self) -> None:
        budget = getattr(self.cfg, "budget_usd", None)
        if not budget:
            return
        frac = self.stats["cost"] / budget
        if frac >= 1.0 and self._budget_stage < 2:
            self._budget_stage = 2
            self.ui.warn(
                f"session budget exceeded: ${self.stats['cost']:.2f} / ${budget:.2f} - "
                "/reset-budget zeroes the counter, or raise agent.budget_usd in config.toml"
            )
        elif frac >= 0.8 and self._budget_stage < 1:
            self._budget_stage = 1
            self.ui.warn(f"80% of session budget used: ${self.stats['cost']:.2f} / ${budget:.2f}")

    def _stats_tool(self, name: str, secs: float, error: bool) -> None:
        e = self.stats["tools"].setdefault(name, {"calls": 0, "errors": 0, "secs": 0.0})
        e["calls"] += 1
        e["secs"] += secs
        if error:
            e["errors"] += 1

    def _record_turn(self, secs: float, in0: int, out0: int, cost0: float) -> None:
        """Append this turn's consumption row (for /stats per-turn table)."""
        self.stats["turns"].append({
            "turn": self.context.turns,
            "in": self.context.total_tokens_in - in0,
            "out": self.context.total_tokens_out - out0,
            "cost": round(self.stats["cost"] - cost0, 6),
            "secs": round(secs, 2),
            "tools": self._turn_tool_calls,
        })
        if len(self.stats["turns"]) > 64:
            self.stats["turns"] = self.stats["turns"][-64:]
        if self._turn_mutations:
            self.ui.info(
                f"✎ {self._turn_mutations} change(s) applied this turn - "
                "/diff to review · /undo to revert"
            )

    # -------------------------------------------------------------------- run
    def run(self, user_input: str) -> str:
        self._empty_streak = 0
        self._stream_chars = 0
        self._checkpoint_sha = None  # /undo rolls back ONE turn
        self.context.turns += 1
        self._update_hud()
        self.ui.user_echo(user_input)
        turn_t0 = time.monotonic()
        in0, out0 = self.context.total_tokens_in, self.context.total_tokens_out
        cost0 = self.stats["cost"]
        self._turn_tool_calls = 0
        self._turn_mutations = 0
        summary = self.context.compact(self.llm) if self.context.needs_compaction() else None
        if summary:
            self.ui.info("context compacted (old history summarized)")
        self.context.add({"role": "user", "content": user_input})

        try:
            for _ in range(self.cfg.max_iterations):
                self._streamed = False
                self.ui.thinking_start()
                try:
                    t0 = time.perf_counter()
                    msg, usage, finish = self.llm.chat(
                        [{"role": "system", "content": self.system}, *self.context.messages],
                        self.registry.schemas,
                    )
                    self._last_chat_secs = time.perf_counter() - t0
                finally:
                    self.ui.thinking_stop()
                    if self._streamed:
                        self.ui.stream_end()
                self.stats["api_calls"] += 1
                self.stats["latency_sum"] += self._last_chat_secs
                self.context.add(msg)
                self.context.track_usage(usage)
                self._account_usage(usage)
                if usage:
                    self.ui.usage(usage)
                self._update_hud()

                if msg.get("tool_calls") and (msg.get("content") or "").strip() and not self._streamed:
                    # intermediate assistant text = its visible plan; render the
                    # numbered ◇ thinking block (streamed text already showed live)
                    self.ui.thinking_panel(msg["content"], self._last_chat_secs)

                if finish == "length":
                    # visible in BOTH phases: a truncated final answer AND a truncated
                    # tool_call (whose cut JSON would otherwise surface as a confusing
                    # "invalid JSON in arguments" error with no explanation)
                    self.ui.warn(
                        "the response was cut at agent.max_output_tokens - "
                        "raise it in config.toml to get longer answers"
                    )

                if not msg.get("tool_calls"):
                    answer = (msg.get("content") or "").strip()
                    if not answer:
                        # Some models end the turn with an empty content field (output went
                        # to a hidden reasoning field, or got truncated). Never stop silently:
                        # make the ledger provider-safe, then nudge the model to continue.
                        self._empty_streak += 1
                        if self._empty_streak >= 3:
                            self.ui.warn(
                                "model returned empty messages 3 times - stopping. "
                                "Try rephrasing the request or /clear."
                            )
                            return "(stopped: repeated empty model responses)"
                        self.ui.info("(model returned an empty message - asking it to continue)")
                        self.context.messages[-1] = {**msg, "content": "(empty response)"}
                        self.context.add({
                            "role": "user",
                            "content": (
                                "Your last message was empty. Either continue the task with tool "
                                "calls now, or write your final answer to the user."
                            ),
                        })
                        continue
                    self._empty_streak = 0
                    self.ui.answer(answer)
                    return answer

                self._empty_streak = 0
                if not self._run_tool_calls(msg):
                    self.ui.warn(
                        "(turn interrupted - partial results kept in context; "
                        "type 'continue' to resume)"
                    )
                    return "stopped: turn interrupted"

            self.ui.warn(
                f"stopped after {self.cfg.max_iterations} tool rounds - all progress is kept in "
                "context. Type 'continue' to resume, or raise agent.max_iterations in config.toml."
            )
            return "stopped: max iterations reached"
        finally:
            self._record_turn(time.monotonic() - turn_t0, in0, out0, cost0)

    # ------------------------------------------------------------- tool calls
    def _run_tool_calls(self, msg: dict) -> bool:
        """Dispatch every tool_call inside a visible `◆ agent::tools` tree.
        Returns False if the user interrupted."""
        calls = msg["tool_calls"]
        self.ui.tools_start(len(calls))
        t0 = time.monotonic()
        completed: set[str] = set()
        ok = True
        try:
            for tc in calls:
                result = self.dispatch(tc)
                completed.add(tc["id"])
                self.context.add({"role": "tool", "tool_call_id": tc["id"], "content": result})
        except KeyboardInterrupt:
            # keep the ledger valid: every remaining tool_call needs a result
            for tc in calls:
                if tc["id"] not in completed:
                    self.context.add(
                        {"role": "tool", "tool_call_id": tc["id"],
                         "content": "error: interrupted by user before this tool ran"}
                    )
            ok = False
        self.ui.tools_end(ok=ok, secs=time.monotonic() - t0)
        return ok

    def dispatch(self, tc: dict) -> str:
        t0 = time.perf_counter()
        name = "?"
        try:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
                if not isinstance(args, dict):
                    raise ToolError("arguments must be a JSON object")
            except json.JSONDecodeError as exc:
                return self._dispatch_fail(name, {},
                                           f"error: invalid JSON in arguments for {name}: {exc}", t0)

            try:
                tool = self.registry.get(name)
            except ToolError as exc:
                return self._dispatch_fail(name, {}, f"error: {exc}", t0)

            self.ui.tool_call(name, args)
            self._turn_tool_calls += 1
            allowed, reason = self.policy.decide(tool, args)
            if not allowed:
                self._stats_tool(name, 0.0, True)
                self.ui.tool_result(name, f"denied ({reason})", error=True)
                return f"error: {reason} - do not retry the same call; ask the user how to proceed"

            if name in MUTATING_TOOLS:
                self._ensure_checkpoint()  # snapshot BEFORE the first change of this turn

            try:
                result = tool.fn(**args)
            except ToolError as exc:
                result = f"error: {exc}"
            except TypeError as exc:
                result = f"error: bad arguments for {name}: {exc}"
            except Exception as exc:  # noqa: BLE001 - never crash the loop from a tool
                result = f"error: {type(exc).__name__}: {exc}"

            secs = time.perf_counter() - t0
            error = result.startswith("error")
            self._stats_tool(name, secs, error)
            if name in MUTATING_TOOLS and not error:
                self._turn_mutations += 1
            self.ui.tool_result(name, result, error=error, secs=secs)
            return result
        except Exception as exc:  # noqa: BLE001 - absolute last resort
            return self._dispatch_fail(name, {},
                                       f"error: dispatch failure: {type(exc).__name__}: {exc}", t0)

    def _dispatch_fail(self, name: str, args: dict, result: str, t0: float) -> str:
        """Failures that happen before the normal call path - still rendered in
        the tools tree and counted, never silent."""
        self.ui.tool_call(name, args)
        self._turn_tool_calls += 1
        self._stats_tool(name, time.perf_counter() - t0, True)
        self.ui.tool_result(name, result, error=True)
        return result

    # ------------------------------------------------------------- sub-agents
    def _run_subagent(self, prompt: str, label: str = "") -> str:
        """Run a fresh-context read-only child agent; return only its report.

        The child shares the LLM client (its chats are serialized after ours)
        but has its OWN Context ledger, so heavy exploration never bloats the
        main conversation. Child usage is rolled into the parent's totals.
        """
        from .ui.render import SubAgentUI

        saved_delta = self.llm.on_delta
        self.llm.on_delta = None  # child output must not bleed into the main view
        try:
            sub = Agent(self.cfg, self.llm, SubAgentUI(self.ui), self.ctx.root,
                        extra_rules=SUBAGENT_RULES, read_only=True)
            sub.context.turns = self.context.turns  # cosmetic continuity
            header = f"(sub-agent mission: {label})\n\n" if label else ""
            answer = sub.run(f"{header}{prompt}")
        except LLMError as exc:
            return f"error: sub-agent failed: {exc}"
        except Exception as exc:  # noqa: BLE001 - a broken child must not break the parent
            return f"error: sub-agent crashed: {type(exc).__name__}: {exc}"
        finally:
            self.llm.on_delta = saved_delta
        # roll child usage AND telemetry into the parent's totals (HUD + /stats)
        self.context.total_tokens_in += sub.context.total_tokens_in
        self.context.total_tokens_out += sub.context.total_tokens_out
        cs = sub.stats
        self.stats["api_calls"] += cs.get("api_calls", 0)
        self.stats["latency_sum"] += cs.get("latency_sum", 0.0)
        self.stats["cost"] += cs.get("cost", 0.0)
        self.stats["cached"] += cs.get("cached", 0)
        for tname, e in cs.get("tools", {}).items():
            mine = self.stats["tools"].setdefault(tname, {"calls": 0, "errors": 0, "secs": 0.0})
            mine["calls"] += e.get("calls", 0)
            mine["errors"] += e.get("errors", 0)
            mine["secs"] += e.get("secs", 0.0)
        for t in cs.get("turns", []):
            self.stats["turns"].append({**t, "turn": f"sub\u00b7{t.get('turn', '?')}"})
        self._turn_tool_calls += sum(e.get("calls", 0) for e in cs.get("tools", {}).values())
        self._update_hud()
        if not answer:
            return "error: sub-agent returned an empty report"
        return answer

    # -------------------------------------------------------- git checkpoints
    def _git(self, *args: str, timeout: int = 15) -> tuple[int, str, str]:
        try:
            proc = subprocess.run(
                ["git", "-C", str(self.ctx.root), *args],
                capture_output=True, text=True, timeout=timeout,
            )
            return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
        except (OSError, subprocess.TimeoutExpired):
            return 1, "", "git unavailable"

    def _ensure_checkpoint(self) -> None:
        """Snapshot tracked files ONCE per turn, before the first mutation.

        `git stash create` builds a commit object WITHOUT touching the working
        tree or the index - a pure snapshot. On a clean tree we fall back to
        HEAD. /undo restores tracked files to this snapshot.
        """
        if self._checkpoint_sha is not None:
            return
        code, out, _ = self._git("stash", "create")
        sha = out if (code == 0 and out) else ""
        if not sha:
            code, out, _ = self._git("rev-parse", "HEAD")
            sha = out if code == 0 else ""
        self._checkpoint_sha = sha or None

    def undo_checkpoint(self) -> str:
        """Restore tracked files to this turn's pre-mutation snapshot."""
        sha = self._checkpoint_sha
        if not sha:
            return ("no checkpoint to undo - nothing was changed this turn "
                    "(or this is not a git repo)")
        code, _, err = self._git("restore", "--source", sha, "--worktree", "--staged", "--", ".")
        if code != 0:
            return f"error: git restore failed: {err}"
        self._checkpoint_sha = None
        self.ctx.log_modification("undo", f"restored tracked files to checkpoint {sha[:8]}")
        msg = f"rolled back to checkpoint {sha[:8]} (tracked files restored)"
        code, out, _ = self._git("status", "--porcelain")
        if code == 0:
            new_files = [ln[3:] for ln in out.splitlines() if ln.startswith("??")]
            if new_files:
                shown = ", ".join(new_files[:5]) + (" ..." if len(new_files) > 5 else "")
                msg += (f" - note: {len(new_files)} file(s) created after the checkpoint "
                        f"were left in place: {shown}")
        return msg

    def diff_summary(self) -> str:
        """Human summary of working-tree changes vs HEAD (for the /diff command):
        per-file stat, shortstat, plus a note about untracked files."""
        code, out, _err = self._git("diff", "HEAD", "--stat")
        if code != 0:
            return "not a git repo (or git unavailable) - nothing to diff"
        short = self._git("diff", "HEAD", "--shortstat")[1]
        lines = [out or "(no tracked changes vs HEAD)"]
        if short:
            lines.append(short)
        code, status, _ = self._git("status", "--porcelain")
        if code == 0:
            untracked = [ln[3:] for ln in status.splitlines() if ln.startswith("??")]
            if untracked:
                shown = ", ".join(untracked[:5]) + (" ..." if len(untracked) > 5 else "")
                lines.append(f"untracked: {shown}")
        return "\n".join(lines)

    # ------------------------------------------------------------- sessions
    def _sessions_dir(self) -> Path:
        return self.ctx.root / ".bilal" / "sessions"

    def save_session(self) -> Path | None:
        """Persist the ledger to .bilal/sessions/<id>.json. Silent best-effort."""
        if not self.context.messages:
            return None
        try:
            sdir = self._sessions_dir()
            sdir.mkdir(parents=True, exist_ok=True)
            if not self.session_id:
                stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                self.session_id = f"{stamp}-{id(self) % 0xFFFF:04x}"
            payload = {
                "id": self.session_id,
                "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "model": getattr(getattr(self.cfg, "provider", None), "model", ""),
                "total_in": self.context.total_tokens_in,
                "total_out": self.context.total_tokens_out,
                "turns": self.context.turns,
                "stats": {
                    "api_calls": self.stats["api_calls"],
                    "latency_sum": self.stats["latency_sum"],
                    "cached": self.stats["cached"],
                    "cost": self.stats["cost"],
                    "turns": self.stats["turns"][-32:],
                    "tools": self.stats["tools"],
                },
                "messages": self.context.messages,
            }
            path = sdir / f"{self.session_id}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            self.last_saved_path = path
            self._prune_sessions(keep=20)
            return path
        except (OSError, TypeError, ValueError):
            return None

    def load_session(self, path: Path) -> bool:
        """Replace the current ledger with a saved session. Returns success."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            messages = data["messages"]
            if not isinstance(messages, list):
                return False
            self.context.messages = messages
            self.context.total_tokens_in = int(data.get("total_in", 0))
            self.context.total_tokens_out = int(data.get("total_out", 0))
            self.context.turns = int(data.get("turns", 0))
            st = data.get("stats") or {}
            self.stats.update({
                "api_calls": int(st.get("api_calls", 0)),
                "latency_sum": float(st.get("latency_sum", 0.0)),
                "cached": int(st.get("cached", 0) or 0),
                "cost": float(st.get("cost", 0.0)),
                "turns": list(st.get("turns", [])),
                "tools": dict(st.get("tools", {})),
            })
            self.session_id = data.get("id") or path.stem
            self.last_saved_path = Path(path)
            return True
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def list_sessions(self, limit: int = 10) -> list[dict]:
        """Newest-first summaries: id, saved_at, turns, first user message."""
        out: list[dict] = []
        try:
            files = sorted(self._sessions_dir().glob("*.json"), reverse=True)
        except OSError:
            return out
        for p in files[:limit]:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                first = next((str(m.get("content") or "")[:70]
                              for m in d.get("messages", [])
                              if m.get("role") == "user" and
                              not str(m.get("content") or "").startswith("<compacted")), "")
                out.append({
                    "path": p,
                    "id": d.get("id", p.stem),
                    "saved_at": d.get("saved_at", ""),
                    "turns": d.get("turns", 0),
                    "first": first,
                })
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return out

    def _prune_sessions(self, keep: int) -> None:
        try:
            files = sorted(self._sessions_dir().glob("*.json"), reverse=True)
            for old in files[keep:]:
                old.unlink(missing_ok=True)
        except OSError:
            pass
