"""Terminal rendering with rich - the modern telemetry TUI.

Design language (from the user's mockups):
- slim header: traffic lights, name, version + session tag, ONLINE | tty, chips row
- session card: workdir, policy, memory layers, shortcuts
- message flow: `❯ user` echo with timestamp, `◇ thinking` numbered plan blocks,
  `◆ agent::tools` call trees with inline ✓/✗ results, `◆ bilal_agent · ready`
  response label before markdown
- live economy line above every prompt (tok / cost / turn) + full /stats dashboard
"""
from __future__ import annotations

import datetime
import hashlib
import os
import time

from rich.box import ROUNDED, SIMPLE_HEAVY
from rich.columns import Columns
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .. import __version__

_ACCENT = "#5cb3ff"
_GREEN = "#3fb950"
_RED = "#f85149"
_YELLOW = "#e3b341"
_MAGENTA = "#d2a8ff"
_DIM = "bright_black"

_TOOL_COLOR = {"safe": "dim", "ask": "yellow"}

# ANSI-shadow block letters for the classic banner (/banner easter egg).
_FIGLETS = {
    "B": ["██████╗", "██╔══██╗", "██████╔╝", "██╔══██╗", "██████═╝", "╚═╝  ╚═╝"],
    "I": ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    "L": ["██╗", "██║", "██║", "██║", "███████╗", "╚══════╝"],
    "A": [" █████╗", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
}
_ART_STYLES = ["cyan", "bright_cyan", "bright_blue", "blue", "bright_magenta", "magenta"]


def _word_art(word: str) -> list[str]:
    """Join per-letter block fonts into full-width rows (letters must all be in _FIGLETS).
    Each letter's rows are padded to that letter's own width so multi-row shapes
    (L's foot, A's peak) stay column-aligned."""
    blocks = []
    for ch in word.upper():
        if ch in _FIGLETS:
            w = max(len(r) for r in _FIGLETS[ch])
            blocks.append([r.ljust(w) for r in _FIGLETS[ch]])
    return ["  ".join(parts).rstrip() for parts in zip(*blocks)]


# ------------------------------------------------------------------ formatting
def _fmt_tok(n: float) -> str:
    n = int(n)
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _fmt_clock(secs: float) -> str:
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _bar(frac: float, width: int = 24) -> Text:
    frac = 0.0 if frac < 0 else min(1.0, frac)
    filled = round(frac * width)
    style = _ACCENT if frac < 0.8 else (_YELLOW if frac < 1.0 else _RED)
    t = Text("━" * filled, style=style)
    t.append("╌" * (width - filled), style="dim")
    return t


def _tool_glyph(name: str, args: dict) -> tuple[str, str]:
    """(glyph, left-column text) - the compact per-tool signature line."""
    a = args or {}
    if name == "run_command":
        return "$", str(a.get("command", ""))[:70]
    if name == "edit_file":
        return "✎", str(a.get("path", ""))
    if name == "write_file":
        return "+", str(a.get("path", ""))
    if name == "read_file":
        return "≡", str(a.get("path", ""))
    if name in ("grep", "grep_search", "glob", "glob_files"):
        return "?", str(a.get("pattern", ""))[:50]
    if name == "list_dir":
        return "≡", str(a.get("path", "."))
    if name == "todo_write":
        return "☐", f"{len(a.get('todos', []) or [])} items"
    if name == "task":
        return "»", str(a.get("label") or a.get("prompt", ""))[:50]
    first = str(next(iter(a.values()), ""))[:40] if a else ""
    return "•", f"{name} {first}".strip()


def _summarize_result(name: str, result: str) -> str:
    lines = [ln.strip() for ln in str(result).splitlines() if ln.strip()]
    if not lines:
        return "(empty)"
    if name == "run_command":
        exit_line = next((ln for ln in lines if ln.lower().startswith("exit code")), "")
        out = [ln for ln in lines if not ln.lower().startswith("exit code")]
        first = out[0][:60] if out else ""
        if first and exit_line:
            return f"{first} · {exit_line.lower()}"
        return first or exit_line or lines[0][:60]
    if name == "read_file":
        return f"{len(lines)} lines"
    return lines[0][:70]


class UI:
    def __init__(self, auto_yes: bool = False):
        self.console = Console(highlight=False)
        self.auto_yes = auto_yes
        self._status = None
        self._pending: tuple[str, dict] | None = None
        self._started = time.time()
        self._think_t0 = 0.0
        self.session_tag = hashlib.sha1(f"{time.time_ns()}-{os.getpid()}".encode()).hexdigest()[:5]

    # ----------------------------------------------------------------- header
    def _join_lr(self, left: Text, right: Text) -> Text:
        """One terminal row with left + right-aligned parts."""
        width = max(24, self.console.width - 2)
        gap = width - left.cell_len - right.cell_len
        return Text.assemble(left, Text(" " * max(1, gap)), right)

    def banner(self, version: str, provider: str, model: str, root, yes: bool) -> None:
        """Modern slim header: traffic lights, identity, chips, session card."""
        c = self.console
        c.print()
        tty = ""
        try:
            tty = "/".join(os.ttyname(1).split("/")[-2:])
        except (OSError, ValueError, AttributeError):
            pass
        left = Text.assemble(
            ("● ", _RED), ("● ", _YELLOW), ("●  ", _GREEN),
            ("bilal_agent", "bold white"),
            (f" v{version}", f"bold {_ACCENT}"),
            (f" [{self.session_tag}]", "dim"),
        )
        right = Text.assemble(("● ONLINE", f"bold {_GREEN}"), (f" | {tty}" if tty else "", "dim"))
        c.print(self._join_lr(left, right))
        chips = Text.assemble(
            ("▌ ", f"bold {_ACCENT}"), ("repl", f"bold {_ACCENT}"), ("  ", ""),
            (f" {model} ", f"black on {_ACCENT}"), ("  ", ""),
            (f"[pid: {os.getpid()}]", "dim"),
        )
        c.print(chips)
        policy = "auto-approve" if yes else "confirm"
        grid = Table.grid(expand=False, padding=(0, 1))
        grid.add_column()
        grid.add_column(justify="right")
        grid.add_row(
            Text.assemble(("● session::active", f"bold {_GREEN}")),
            Text.assemble(("policy: ", "dim"), (policy, _YELLOW)),
        )
        grid.add_row(
            Text.assemble(("workdir: ", "dim"), (str(root), "white")),
            Text.assemble(("provider: ", "dim"), (provider, _MAGENTA)),
        )
        grid.add_row(
            Text.assemble(("memory: ", "dim"), ("AGENT.md · journal · checkpoints · sessions", "white")),
            Text("", ""),
        )
        grid.add_row(
            Text.assemble(("type a request · ", "dim"), ("/help", "dim bold"),
                          (" for commands", "dim")),
            Text("", ""),
        )
        c.print(Panel(grid, box=ROUNDED, border_style=_ACCENT, expand=False, padding=(0, 1)))
        c.print(Text.assemble(
            ("shortcuts  ", "dim"),
            ("/help · /stats · /cost · /diff · /compact · /resume · /undo · /clear", f"{_DIM}"),
        ))
        if yes:
            c.print("[bold yellow]--yes mode: ALL risky actions auto-approved[/]")
        c.print()

    def banner_classic(self, version: str, provider: str, model: str) -> None:
        """The v0.3 big block-letter banner - kept alive via /banner."""
        self.console.print()
        art = _word_art("BILAL")
        width = max(len(row) for row in art)

        def _centered(text: str, style: str) -> None:
            pad = max(0, (width - len(text)) // 2)
            self.console.print(Text(" " * pad + text, style=style))

        _centered("✻ WELCOME TO", "dim")
        for row, style in zip(art, _ART_STYLES):
            self.console.print(Text(row, style=f"bold {style}"))
        _centered("C O D I N G   A G E N T", "bold magenta")
        self.console.print(
            Panel(
                Text.assemble(
                    ("provider ", "dim"), (provider, "magenta"),
                    ("  ·  model ", "dim"), (model, "magenta"),
                    ("  ·  version ", "dim"), (version, "cyan"),
                ),
                expand=False, border_style="cyan",
            )
        )
        self.console.print()

    def info(self, msg: str) -> None:
        self.console.print(f"[dim]{msg}[/]")

    def warn(self, msg: str) -> None:
        self.console.print(f"[bold yellow]{msg}[/]")

    def error(self, msg: str) -> None:
        self.console.print(f"[bold red]{msg}[/]")

    # ------------------------------------------------------------------- REPL
    def prompt(self) -> str:
        return self.console.input(f"[bold {_ACCENT}]❯ [/]")

    # ------------------------------------------------------------------ trace
    def thinking_start(self, label: str = "◇ thinking") -> None:
        self._think_t0 = time.monotonic()
        self._status = self.console.status(f"[green]{label}[/]", spinner="dots")
        self._status.start()

    def thinking_stop(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
            self._last_think_secs = time.monotonic() - self._think_t0

    def set_title(self, text: str) -> None:
        """Live token HUD part 1: update the terminal window title (top bar).
        Plain ANSI escape - does not disturb rich rendering."""
        try:
            import sys

            sys.stdout.write(f"\x1b]2;{text}\x07")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - title is cosmetic, never fatal
            pass

    def hud(self, text: str) -> None:
        """Live economy line: one dim status row above each REPL prompt."""
        self.console.print(f"[{_DIM}]{'─' * min(self.console.width - 1, 56)}[/]")
        self.console.print(f"[{_DIM}]{text}[/]")

    def user_echo(self, text: str) -> None:
        """Timestamped `❯ user` anchor at the start of every turn (mockup style)."""
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.console.print(Text.assemble(("❯ user", f"bold {_ACCENT}"), (f"  {stamp}", "dim")))
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        for ln in lines[:2]:
            self.console.print(Text("│ ", style="dim").append_text(Text(ln[:110], style="dim")))
        if len(lines) > 2:
            self.console.print(Text(f"│ … (+{len(lines) - 2} lines)", style="dim"))

    def thinking_panel(self, content: str, secs: float = 0.0) -> None:
        """The model's intermediate text (its visible plan between tool rounds),
        rendered as the mockup's numbered `◇ thinking` block."""
        lines = [ln.strip() for ln in content.strip().splitlines() if ln.strip()]
        if not lines:
            return
        est = max(1, len(content) // 4)
        head = Text.assemble(
            ("◇ thinking", "bold dim"),
            (f"  ({secs:.1f}s · ~{_fmt_tok(est)} tokens)", "dim"),
        )
        self.console.print(head)
        for i, ln in enumerate(lines[:8], 1):
            self.console.print(Text(f"  {i:02d}  {ln[:100]}", style="dim italic"))
        if len(lines) > 8:
            self.console.print(Text(f"  … (+{len(lines) - 8} more)", style="dim"))

    # ------------------------------------------------------------- tools tree
    def tools_start(self, n: int) -> None:
        word = "call" if n == 1 else "calls"
        self.console.print(Text.assemble(
            ("╭─", "dim"), (" ◆ agent::tools", f"bold {_ACCENT}"), (f" · {n} {word}", "dim"),
        ))

    def tool_call(self, name: str, args: dict) -> None:
        """Called right before execution - the line is completed by tool_result()."""
        self._pending = (name, args or {})

    def _tool_line(self, name: str, args: dict, result: str, error: bool,
                   secs: float, prefix: str = "│ ") -> Text:
        glyph, left = _tool_glyph(name, args)
        glyph_style = _YELLOW if name == "run_command" else _ACCENT
        mark, mark_style = ("✗", _RED) if error else ("✓", _GREEN)
        line = Text.assemble((prefix, "dim"), (f"{glyph} ", glyph_style), (left, "white"))
        pad = max(2, 52 - line.cell_len)
        tail = Text.assemble(
            (mark, mark_style), (" ", ""),
            (_summarize_result(name, result), "dim"),
            (f" · {secs:.1f}s" if secs >= 0.05 else "", "dim"),
        )
        line.append(" " * pad)
        line.append_text(tail)
        return line

    def tool_result(self, name: str, result: str, error: bool = False, secs: float = 0.0) -> None:
        pname, pargs = self._pending if self._pending and self._pending[0] == name else (name, {})
        self._pending = None
        self.console.print(self._tool_line(pname, pargs, result, error, secs))

    def tools_end(self, ok: bool, secs: float = 0.0, note: str = "") -> None:
        if self._pending:  # a call never received its result (interrupt / crash)
            name, args = self._pending
            self._pending = None
            self.console.print(self._tool_line(name, args, "no result (interrupted)", True, 0.0))
        mark, style = ("✓ ok", _GREEN) if ok else ("✗ failed", _RED)
        self.console.print(Text.assemble(
            ("╰─ ", "dim"), (mark, style),
            (f" · {secs:.1f}s" if secs >= 0.05 else "", "dim"),
            (f" · {note}" if note else "", "dim"),
        ))

    # ---------------------------------------------------------------- answers
    def stream_start(self) -> None:
        """First token of a streamed answer arrived: drop the spinner, label, go live."""
        self.thinking_stop()
        self.console.print(Text.assemble(
            ("◆ bilal_agent", f"bold {_ACCENT}"), ("  ·  streaming", "dim"),
        ))

    def delta(self, text: str) -> None:
        self.console.print(text, markup=False, highlight=False, end="", soft_wrap=True)

    def stream_end(self) -> None:
        self.console.print("\n")

    def usage(self, usage: dict | None) -> None:
        """Per-call usage hook - silent in the modern TUI; the status line and
        /stats aggregate it (kept as a method so agent/tests can hook it)."""
        self._last_usage = usage

    def answer(self, content: str) -> None:
        self.console.print()
        if content.strip():
            self.console.print(Text.assemble(
                ("◆ bilal_agent", f"bold {_ACCENT}"), ("  ·  ready", "dim"),
            ))
            self.console.print(Markdown(content))
        self.console.print()

    # ----------------------------------------------------------------- economy
    def _pricing_of(self, agent) -> tuple[float, float, str]:
        from ..config import resolve_pricing

        return resolve_pricing(agent.cfg)

    def cost_card(self, agent) -> None:
        """Quick /cost view: money + tokens + api calls for this session."""
        st = getattr(agent, "stats", {}) or {}
        ctx = agent.context
        cfg = agent.cfg
        pin, pout, src = self._pricing_of(agent)
        cost = float(st.get("cost", 0.0))
        budget = getattr(cfg, "budget_usd", None)
        calls = int(st.get("api_calls", 0))
        avg = (float(st.get("latency_sum", 0.0)) / calls) if calls else 0.0

        if budget:
            econ = Text.assemble(
                ("Limit: ", "dim"), (f"${budget:.2f}", "white"),
                (f"  {int(100 * cost / budget)}%" if budget else "", "dim"), ("  ", ""),
            )
            bar = _bar(cost / budget, 26)
        else:
            econ = Text("no budget set (agent.budget_usd in config.toml)", style="dim")
            bar = Text("", "")
        body = Table.grid(padding=(0, 3))
        body.add_row(
            Text.assemble(("COST", "bold"), ("  USD", "dim")),
            Text.assemble(("TOKENS", "bold"), ("  TOTAL", "dim")),
        )
        body.add_row(
            Text(f"${cost:.4f}".rstrip("0").rstrip(".") + ("0" if abs(cost) < 0.001 else ""), style=f"bold {_GREEN}"),
            Text(_fmt_tok(ctx.total_tokens_in + ctx.total_tokens_out), style=f"bold {_ACCENT}"),
        )
        body.add_row(bar, _bar(ctx.estimate() / cfg.max_context_tokens if cfg.max_context_tokens else 0, 26))
        body.add_row(
            econ,
            Text.assemble(("In: ", "dim"), (_fmt_tok(ctx.total_tokens_in), "white"),
                          ("  Out: ", "dim"), (_fmt_tok(ctx.total_tokens_out), "white")),
        )
        if st.get("cached"):
            body.add_row(Text("", ""), Text.assemble(("Cached: ", "dim"), (_fmt_tok(st["cached"]), "white")))
        body.add_row(
            Text.assemble(("api calls: ", "dim"), (str(calls), "white"),
                          (f"  avg latency {avg:.2f}s" if calls else "", "dim")),
            Text.assemble(("$ in / $ out per 1M: ", "dim"),
                          (f"{pin}/{pout}", "white"), (f"  ({src})", "dim")),
        )
        self.console.print(Panel(body, title="cost", box=ROUNDED, border_style=_ACCENT,
                                 expand=False, padding=(0, 1)))
        self.console.print(Text.assemble(("reset with ", "dim"), ("/reset-budget", "dim bold"),
                                         ("  ·  full view: ", "dim"), ("/stats", "dim bold")))

    def telemetry(self, agent) -> None:
        """/stats - the full dashboard: session, cost, tokens, calls, per-turn,
        tool invocations, runtime environment."""
        c = self.console
        st = getattr(agent, "stats", {}) or {}
        ctx = agent.context
        cfg = agent.cfg
        pin, pout, src = self._pricing_of(agent)
        cost = float(st.get("cost", 0.0))
        budget = getattr(cfg, "budget_usd", None)
        calls = int(st.get("api_calls", 0))
        avg = (float(st.get("latency_sum", 0.0)) / calls) if calls else 0.0
        tools = st.get("tools", {})
        n_tool_calls = sum(t.get("calls", 0) for t in tools.values())
        n_tool_errors = sum(t.get("errors", 0) for t in tools.values())
        uptime = _fmt_clock(time.time() - self._started)
        policy = "AUTO-APPROVE" if cfg.auto_yes else ("CONFIRM_RISKY" if cfg.confirm_edits else "PERMISSIVE")

        c.print()
        # -- session identity ------------------------------------------------
        grid = Table.grid(expand=False, padding=(0, 1))
        grid.add_column()
        grid.add_column(justify="right")
        grid.add_row(
            Text.assemble(("AGENT_PID  ", "dim"), (str(os.getpid()), "white")),
            Text.assemble(("UPTIME  ", "dim"), (uptime, "white")),
        )
        grid.add_row(
            Text.assemble(("POLICY     ", "dim"), (policy, _YELLOW)),
            Text.assemble(("SNAPSHOT  ", "dim"), (f"v{__version__}", "white")),
        )
        grid.add_row(
            Text.assemble(("ENGINE     ", "dim"), (cfg.provider.model, _MAGENTA)),
            Text.assemble(("ROOT  ", "dim"), (str(agent.ctx.root), "white")),
        )
        c.print(Panel(
            Group(
                Text.assemble(("SESSION TELEMETRY", "bold"), ("   ● live", f"bold {_GREEN}")),
                Text(""),
                grid,
            ),
            box=ROUNDED, border_style=_ACCENT, expand=False, padding=(0, 1),
        ))

        # -- stat cards --------------------------------------------------------
        def _card(title: str, tag: str, big: Text, under) -> Panel:
            body = Table.grid(padding=(0, 0))
            body.add_row(Text.assemble((title, "bold"), (f"  {tag}", "dim")))
            body.add_row(big)
            for row in under:
                body.add_row(row)
            return Panel(body, box=ROUNDED, border_style="dim", expand=False, padding=(0, 1))

        if budget:
            cost_under: list = [_bar(cost / budget, 26),
                                Text.assemble(("Limit: ", "dim"), (f"${budget:.2f}", "white"),
                                              (f"  {int(100 * cost / budget)}%", "dim"))]
        else:
            cost_under = [Text.assemble(("Limit: none", "dim"), (f"  {src}", "dim"))]
        cached = int(st.get("cached", 0) or 0)
        cache_txt = f"  cached {_fmt_tok(cached)}" if cached else ""
        cards = [
            _card("COST", "USD", Text(f"${cost:.4f}", style=f"bold {_GREEN}"), cost_under),
            _card("TOKENS", "TOTAL",
                  Text(_fmt_tok(ctx.total_tokens_in + ctx.total_tokens_out), style=f"bold {_ACCENT}"),
                  [_bar(ctx.estimate() / cfg.max_context_tokens if cfg.max_context_tokens else 0, 26),
                   Text.assemble(("In: ", "dim"), (_fmt_tok(ctx.total_tokens_in), "white"),
                                 ("  Out: ", "dim"), (_fmt_tok(ctx.total_tokens_out), "white"),
                                 (cache_txt, "dim"))]),
            _card("API CALLS", "COUNT", Text(str(calls), style=f"bold white"),
                  [Text.assemble(("Avg latency ", "dim"), (f"{avg:.2f}s", "white")),
                   Text.assemble(("Err rate ", "dim"),
                                 (f"{(100 * n_tool_errors / n_tool_calls):.1f}%" if n_tool_calls else "0.0%",
                                  _GREEN if not n_tool_errors else _RED), ("  (tools)", "dim"))]),
            _card("TOOL CALLS", "COUNT", Text(str(n_tool_calls), style=f"bold white"),
                  [Text.assemble((f"{len(tools)}", "white"),
                                 (" distinct tool" + ("s" if len(tools) != 1 else ""), "dim")),
                   Text.assemble(("turns ", "dim"), (str(ctx.turns), "white"))]),
        ]
        tw = max(36, (self.console.width - 8) // 2)
        c.print(Columns(cards, width=tw, padding=(0, 1)))

        # -- consumption per turn ----------------------------------------------
        turns = st.get("turns", [])[-8:]
        if turns:
            tt = Table(box=SIMPLE_HEAVY, expand=False, show_header=True, header_style="dim")
            tt.add_column("turn", justify="right")
            tt.add_column("tokens", justify="right")
            tt.add_column("cost", justify="right")
            tt.add_column("time", justify="right")
            tt.add_column("tools", justify="right")
            peak = max((t.get("in", 0) + t.get("out", 0) for t in turns), default=0) or 1
            for t in turns:
                tok = int(t.get("in", 0)) + int(t.get("out", 0))
                tt.add_row(
                    Text(str(t.get("turn", "?")), style=_ACCENT),
                    Text(f"{_bar(tok / peak, 14)} ", style=""), Text(str(tok), style="white"),
                    Text(f"${t.get('cost', 0):.4f}", style=_GREEN),
                    Text(f"{t.get('secs', 0):.1f}s", style="dim"),
                    Text(str(t.get("tools", 0)), style="dim"),
                )
            c.print(Panel(Group(Text.assemble(("CONSUMPTION PER TURN", "bold")), Text(""),
                               tt, Text("")),
                          title=Text.assemble(("last 8 turns", "dim")),
                          box=ROUNDED, border_style="dim", expand=False, padding=(0, 1)))

        # -- tool invocations ---------------------------------------------------
        if tools:
            tt = Table(box=SIMPLE_HEAVY, expand=False, show_header=True, header_style="dim")
            tt.add_column("tool")
            tt.add_column("calls", justify="right")
            tt.add_column("errors", justify="right")
            tt.add_column("avg", justify="right")
            for name in sorted(tools, key=lambda k: -tools[k].get("calls", 0)):
                e = tools[name]
                n = int(e.get("calls", 0))
                errs = int(e.get("errors", 0))
                avg_t = float(e.get("secs", 0.0)) / n if n else 0.0
                tt.add_row(
                    Text(name, style=_ACCENT),
                    Text(str(n), style="white"),
                    Text(str(errs), style=_GREEN if not errs else _RED),
                    Text(f"{avg_t * 1000:.0f}ms" if avg_t < 1 else f"{avg_t:.2f}s", style="dim"),
                )
            c.print(Panel(
                Group(Text.assemble(("TOOL INVOCATIONS", "bold"), (f"  N={n_tool_calls}", "dim")),
                      Text(""), tt),
                box=ROUNDED, border_style="dim", expand=False, padding=(0, 1),
            ))

        # -- runtime environment -------------------------------------------------
        temp = getattr(cfg, "temperature", None)
        env = Table.grid(padding=(0, 1))
        env.add_column(); env.add_column(justify="right")
        env.add_row(Text("Engine", "dim"), Text(cfg.provider.model, style="white"))
        env.add_row(Text("Max output", "dim"), Text(f"{cfg.max_output_tokens} tokens", style="white"))
        env.add_row(Text("Context compacts at", "dim"), Text(f"{cfg.max_context_tokens} tokens", style="white"))
        env.add_row(Text("Streaming", "dim"), Text("on" if cfg.stream else "off",
                                                    style=_GREEN if cfg.stream else "dim"))
        env.add_row(Text("Temperature", "dim"), Text("n/a" if temp is None else str(temp), style="white"))
        env.add_row(Text("Pricing", "dim"), Text(f"${pin}/${pout} per 1M ({src})", style="dim"))
        c.print(Panel(Group(Text.assemble(("RUNTIME ENVIRONMENT", "bold"), ("   active", f"bold {_GREEN}")),
                             Text(""), env),
                      box=ROUNDED, border_style="dim", expand=False, padding=(0, 1)))
        c.print(Text.assemble(
            ("controls  ", "dim"),
            ("/reset-budget", f"dim bold"), (" zero tally  ", "dim"),
            ("/cost", f"dim bold"), (" quick money  ", "dim"),
            ("/compact", f"dim bold"), (" shrink context", "dim"),
        ))

    # --------------------------------------------------------------- confirm
    def confirm(self, title: str, preview: str) -> str:
        """Returns 'yes' | 'no' | 'always'."""
        body = preview if preview.strip() else "(no preview available)"
        lang = "diff" if preview.lstrip().startswith(("---", "+++", "@@")) else "text"
        self.console.print(Panel(Syntax(body, lang, word_wrap=True), title=title, expand=False))
        ans = Prompt.ask(
            "[bold]allow this action?[/] \\[y]es / \\[n]o / \\[a]lways this session",
            choices=["y", "n", "a"],
            default="n",
            console=self.console,
        )
        return {"y": "yes", "n": "no", "a": "always"}[ans]

    # ------------------------------------------------------------------ todos
    def render_todos(self, todos: list[dict]) -> None:
        icon = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
        style = {"pending": "dim", "in_progress": "bold cyan", "completed": "dim strikethrough"}
        lines = [Text(f"{icon[t['status']]} {t['content']}", style=style[t["status"]]) for t in todos]
        body = Group(*lines) if lines else Text("(no items)")
        self.console.print(Panel(body, title="plan", expand=False))


class SubAgentUI:
    """UI proxy for Task-tool sub-agents.

    Forwards: spinner (relabeled), tool traces rendered as a NESTED tree inside
    the parent's open agent::tools group, warnings/info/errors.
    Suppresses: streamed deltas (child output must not bleed into the main
    answer view), final answers (they come back as the tool result), per-call
    usage (rolled up into the parent at the end), user echo / thinking panels.
    """

    def __init__(self, parent: "UI"):
        self.parent = parent
        # test doubles may not carry console/auto_yes - tolerate that
        self.console = getattr(parent, "console", None)
        self.auto_yes = getattr(parent, "auto_yes", False)
        self._pending: tuple[str, dict] | None = None

    def thinking_start(self) -> None:
        self.parent.thinking_start("◇ subagent working")

    def thinking_stop(self) -> None:
        self.parent.thinking_stop()

    def stream_start(self) -> None:
        pass

    def delta(self, text: str) -> None:
        pass

    def stream_end(self) -> None:
        pass

    def set_title(self, text: str) -> None:
        pass  # only the parent drives the window title

    def hud(self, text: str) -> None:
        pass

    def user_echo(self, text: str) -> None:
        pass

    def thinking_panel(self, content: str, secs: float = 0.0) -> None:
        pass  # child plans stay inside the tool result

    def tools_start(self, n: int) -> None:
        if self.console is not None:
            self.console.print(Text.assemble(
                ("│  ▾ subagent", f"bold {_MAGENTA}"), (f" · {n} calls", "dim"),
            ))

    def tool_call(self, name: str, args: dict) -> None:
        self._pending = (name, args or {})

    def tool_result(self, name: str, result: str, error: bool = False, secs: float = 0.0) -> None:
        pname, pargs = self._pending if self._pending and self._pending[0] == name else (name, {})
        self._pending = None
        if self.console is not None:
            self.console.print(self.parent._tool_line(pname, pargs, result, error, secs, prefix="│  "))

    def tools_end(self, ok: bool, secs: float = 0.0, note: str = "") -> None:
        if self._pending:
            name, args = self._pending
            self._pending = None
            if self.console is not None:
                self.console.print(self.parent._tool_line(name, args, "no result (interrupted)", True, 0.0, prefix="│  "))
        if self.console is not None:
            mark, style = ("◂ ok", _GREEN) if ok else ("◂ failed", _RED)
            self.console.print(Text.assemble(("│  ", "dim"), (mark, style)))

    def usage(self, usage: dict) -> None:
        pass

    def answer(self, content: str) -> None:
        pass  # returned to the parent as the tool result instead

    def info(self, msg: str) -> None:
        self.parent.info(msg)

    def warn(self, msg: str) -> None:
        self.parent.warn(msg)

    def error(self, msg: str) -> None:
        self.parent.warn(msg)

    def confirm(self, title: str, preview: str) -> str:
        # sub-agents hold only safe/read tools - this should never fire
        return "no"
