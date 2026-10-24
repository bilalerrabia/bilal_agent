"""End-to-end loop tests with a scripted FakeLLM - the whole agent runs offline.

These tests pin the core invariants:
- real tool results enter the ledger (never fabricated content)
- a denied action returns an error string to the model, not an execution
- a Ctrl+C mid-turn still leaves a valid ledger (every tool_call gets a result)
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bilal_agent.agent import Agent
from bilal_agent.context import COMPACT_TAG, Context


# ---------------------------------------------------------------------- fakes
class FakeLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.calls.append(messages)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, {"prompt": 11, "completion": 7}, None

    def summarize(self, messages):
        return "summary: the user asked about the codebase"


class FakeUI:
    def __init__(self, confirm_answer="no"):
        self.confirm_answer = confirm_answer
        self.confirms: list[tuple[str, str]] = []
        self.answers: list[str] = []
        self.notes: list[str] = []
        self.deltas: list[str] = []

    def thinking_start(self, label: str = "thinking..."):
        pass

    def thinking_stop(self):
        pass

    def stream_start(self):
        pass

    def delta(self, text):
        self.deltas.append(text)

    def stream_end(self):
        pass

    def set_title(self, text):
        self.titles = getattr(self, "titles", [])
        self.titles.append(text)

    def hud(self, text):
        pass

    def confirm(self, title, preview):
        self.confirms.append((title, preview))
        return {"y": "yes", "n": "no", "a": "always"}.get(self.confirm_answer, self.confirm_answer)

    def tool_call(self, name, args):
        pass

    def tool_result(self, name, result, error=False, secs=0.0):
        self.tool_results = getattr(self, "tool_results", [])
        self.tool_results.append((name, result, error))

    def tools_start(self, n):
        self.tools_groups = getattr(self, "tools_groups", [])
        self.tools_groups.append(n)

    def tools_end(self, ok, secs=0.0, note=""):
        self.tools_ends = getattr(self, "tools_ends", [])
        self.tools_ends.append(ok)

    def user_echo(self, text):
        self.echoes = getattr(self, "echoes", [])
        self.echoes.append(text)

    def thinking_panel(self, content, secs=0.0):
        self.thinking_panels = getattr(self, "thinking_panels", [])
        self.thinking_panels.append(content)

    def usage(self, usage):
        pass

    def answer(self, content):
        self.answers.append(content)

    def info(self, msg):
        self.notes.append(msg)

    def warn(self, msg):
        self.notes.append(msg)

    def error(self, msg):
        self.notes.append(msg)


@pytest.fixture()
def proj(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'hi'\n")
    return tmp_path


@pytest.fixture()
def cfg() -> SimpleNamespace:
    return SimpleNamespace(
        max_iterations=5,
        max_context_tokens=60000,
        keep_recent=8,
        confirm_edits=True,
        auto_yes=False,
        command_timeout=10,
    )


def tool_call_msg(cid: str, name: str, **args) -> dict:
    import json

    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }


# ----------------------------------------------------------------------- tests
def test_tool_roundtrip_real_result_in_ledger(proj, cfg):
    llm = FakeLLM([
        tool_call_msg("c1", "read_file", path="src/app.py"),
        {"role": "assistant", "content": "app.py defines hello returning 'hi'"},
    ])
    ui = FakeUI()
    agent = Agent(cfg, llm, ui, proj)
    answer = agent.run("what does app.py do?")

    assert answer == "app.py defines hello returning 'hi'"
    tool_msg = [m for m in agent.context.messages if m.get("role") == "tool"]
    assert len(tool_msg) == 1
    assert tool_msg[0]["tool_call_id"] == "c1"
    assert "def hello" in tool_msg[0]["content"]  # REAL file content, not model text
    # the second request carried the tool result back to the model
    second_call = llm.calls[1]
    assert any(m.get("role") == "tool" for m in second_call)


def test_safe_tool_auto_allowed_risky_needs_confirm(proj, cfg):
    llm = FakeLLM([
        tool_call_msg("r1", "run_command", command="echo hi"),
        {"role": "assistant", "content": "done"},
    ])
    ui = FakeUI(confirm_answer="n")
    agent = Agent(cfg, llm, ui, proj)
    agent.run("run echo hi")

    assert len(ui.confirms) == 1
    assert "$ echo hi" in ui.confirms[0][1]  # command preview shown to the user
    tool_msg = [m for m in agent.context.messages if m.get("role") == "tool"][0]
    assert "denied" in tool_msg["content"]


def test_confirm_yes_executes(proj, cfg):
    llm = FakeLLM([
        tool_call_msg("r1", "run_command", command="echo hi"),
        {"role": "assistant", "content": "ran it"},
    ])
    ui = FakeUI(confirm_answer="y")
    agent = Agent(cfg, llm, ui, proj)
    agent.run("run echo hi")
    tool_msg = [m for m in agent.context.messages if m.get("role") == "tool"][0]
    assert "exit code: 0" in tool_msg["content"] and "hi" in tool_msg["content"]


def test_confirm_always_skips_later_prompts(proj, cfg):
    llm = FakeLLM([
        tool_call_msg("a", "run_command", command="echo one"),
        tool_call_msg("b", "run_command", command="echo two"),
        {"role": "assistant", "content": "ok"},
    ])
    ui = FakeUI(confirm_answer="a")
    agent = Agent(cfg, llm, ui, proj)
    agent.run("two commands")
    assert len(ui.confirms) == 1  # only the first risky call prompts


def test_yes_mode_bypasses_confirm(proj, cfg):
    cfg.auto_yes = True
    llm = FakeLLM([
        tool_call_msg("r", "run_command", command="echo fast"),
        {"role": "assistant", "content": "ok"},
    ])
    ui = FakeUI()
    agent = Agent(cfg, llm, ui, proj)
    agent.run("go")
    assert ui.confirms == []
    tool_msg = [m for m in agent.context.messages if m.get("role") == "tool"][0]
    assert "fast" in tool_msg["content"]


def test_edit_without_read_returns_error_string(proj, cfg):
    llm = FakeLLM([
        tool_call_msg("e1", "edit_file", path="src/app.py", old_string="x", new_string="y"),
        {"role": "assistant", "content": "understood"},
    ])
    agent = Agent(cfg, llm, FakeUI(confirm_answer="y"), proj)
    agent.run("edit it")
    tool_msg = [m for m in agent.context.messages if m.get("role") == "tool"][0]
    assert tool_msg["content"].startswith("error:")
    assert "read" in tool_msg["content"]
    assert (proj / "src" / "app.py").read_text() == "def hello():\n    return 'hi'\n"  # untouched


def test_unknown_tool_is_error_not_crash(proj, cfg):
    llm = FakeLLM([
        tool_call_msg("u1", "teleport", where="moon"),
        {"role": "assistant", "content": "ok"},
    ])
    agent = Agent(cfg, llm, FakeUI(), proj)
    agent.run("teleport")
    tool_msg = [m for m in agent.context.messages if m.get("role") == "tool"][0]
    assert "unknown tool 'teleport'" in tool_msg["content"]


def test_interrupted_turn_keeps_valid_ledger(proj, cfg):
    llm = FakeLLM([
        tool_call_msg("k1", "read_file", path="src/app.py"),
        {"role": "assistant", "content": "never reached"},
    ])
    agent = Agent(cfg, llm, FakeUI(), proj)
    real_fn = agent.registry.get("read_file").fn

    def boom(**kwargs):
        raise KeyboardInterrupt()

    agent.registry._tools["read_file"].fn = boom
    result = agent.run("read it")
    agent.registry._tools["read_file"].fn = real_fn

    assert "interrupted" in result
    last = agent.context.messages[-1]
    assert last["role"] == "tool" and last["tool_call_id"] == "k1"
    assert "interrupted by user" in last["content"]


def test_max_iterations_guard(proj, cfg):
    cfg.max_iterations = 2
    endless = tool_call_msg("x", "list_dir")
    llm = FakeLLM([endless, endless, endless])  # more script than iterations
    ui = FakeUI()
    agent = Agent(cfg, llm, ui, proj)
    result = agent.run("loop forever")
    assert "max iterations" in result
    assert any("continue" in note for note in ui.notes)  # the stop is VISIBLE, never silent


def test_empty_final_message_triggers_nudge_not_silent_stop(proj, cfg):
    """GLM-style models sometimes end the turn with empty content - the agent must
    nudge them back to work instead of returning to the prompt with no output."""
    llm = FakeLLM([
        {"role": "assistant", "content": ""},  # model goes silent after working
        {"role": "assistant", "content": "here is the refactor plan"},
    ])
    ui = FakeUI()
    agent = Agent(cfg, llm, ui, proj)
    result = agent.run("refactor it")

    assert result == "here is the refactor plan"
    assert any("empty message" in note for note in ui.notes)  # visible, not silent
    users = [m for m in agent.context.messages if m["role"] == "user"]
    assert len(users) == 2  # original request + the nudge
    assert "continue the task" in users[-1]["content"].lower()
    # the empty assistant message was made provider-safe in the ledger
    assert any(
        m["role"] == "assistant" and m["content"] == "(empty response)"
        for m in agent.context.messages
    )


def test_repeated_empty_messages_stop_visibly(proj, cfg):
    llm = FakeLLM([{"role": "assistant", "content": ""}] * 3)
    ui = FakeUI()
    agent = Agent(cfg, llm, ui, proj)
    result = agent.run("hello?")
    assert "empty" in result
    assert any("empty" in note for note in ui.notes)


def test_truncated_response_warns_about_max_output_tokens(proj, cfg):
    llm = FakeLLM([{"role": "assistant", "content": "partial answer"}])
    # FakeLLM returns finish=None; simulate length finish via a custom chat
    def chat_with_length(messages, tools=None):
        llm.calls.append(messages)
        return {"role": "assistant", "content": "partial answer"}, None, "length"
    llm.chat = chat_with_length
    ui = FakeUI()
    agent = Agent(cfg, llm, ui, proj)
    result = agent.run("long task")
    assert result == "partial answer"
    assert any("max_output_tokens" in note for note in ui.notes)


def test_truncated_tool_call_also_warns(proj, cfg):
    """finish_reason=length must be visible in the tool phase too - a cut tool_call
    would otherwise surface as a cryptic 'invalid JSON' error with no explanation."""
    scripted = [
        (tool_call_msg("t1", "list_dir"), None, "length"),
        ({"role": "assistant", "content": "done"}, None, "stop"),
    ]
    llm = FakeLLM([])
    it = iter(scripted)

    def chat(messages, tools=None):
        llm.calls.append(messages)
        return next(it)

    llm.chat = chat
    ui = FakeUI()
    agent = Agent(cfg, llm, ui, proj)
    result = agent.run("go")

    assert result == "done"
    assert any("max_output_tokens" in note for note in ui.notes)
    assert [m for m in agent.context.messages if m.get("role") == "tool"]


def test_interrupted_turn_warns_visibly(proj, cfg):
    """Ctrl+C during a tool must print WHY the turn ended, never fall back silently."""
    ui = FakeUI()
    llm = FakeLLM([tool_call_msg("k1", "read_file", path="src/app.py")])
    agent = Agent(cfg, llm, ui, proj)

    def boom(**kwargs):
        raise KeyboardInterrupt()

    agent.registry._tools["read_file"].fn = boom
    result = agent.run("read it")

    assert "interrupted" in result
    assert any("interrupted" in note for note in ui.notes)


# ------------------------------------------------------------------- context
def test_compaction_cuts_at_user_boundary():
    ctx = Context(max_tokens=10, keep_recent=4)
    for i in range(4):
        ctx.add({"role": "user", "content": f"question {i} " + "word " * 10})
        ctx.add({"role": "assistant", "content": "answer " + "x" * 40})

    class MiniLLM:
        def summarize(self, messages):
            return f"summary of {len(messages)} messages"

    assert ctx.needs_compaction()
    assert ctx.compact(MiniLLM()) is not None
    msgs = ctx.messages
    assert len(msgs) == 5  # 1 summary + 4 recent
    assert msgs[0]["role"] == "user" and COMPACT_TAG in msgs[0]["content"]
    assert msgs[1]["role"] == "user"  # tail starts on a user message, never on a tool result
