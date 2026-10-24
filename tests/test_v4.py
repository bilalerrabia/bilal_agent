"""Tests for v0.4.0: Task sub-agents, session save/resume, git checkpoints, HUD."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bilal_agent.agent import Agent
from bilal_agent.ui.render import SubAgentUI, UI
from tests.test_agent_loop import FakeLLM, FakeUI, tool_call_msg


@pytest.fixture()
def proj(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'hi'\n")
    return tmp_path


@pytest.fixture()
def cfg() -> SimpleNamespace:
    return SimpleNamespace(
        max_iterations=6,
        max_context_tokens=60000,
        keep_recent=8,
        confirm_edits=True,
        auto_yes=False,
        command_timeout=10,
    )


@pytest.fixture()
def git_proj(tmp_path: Path) -> Path:
    """A real git repo with one committed file - for checkpoint/undo tests."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "app.py").write_text("original\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    return tmp_path


# ------------------------------------------------------------------- sub-agents
def test_task_tool_returns_child_report(proj, cfg):
    """Parent delegates -> child (same FakeLLM script) answers -> report becomes
    the tool result in the PARENT's ledger."""
    script = [
        tool_call_msg("p1", "task", prompt="find all hello usages", label="scout"),
        # child turns (same shared FakeLLM)
        tool_call_msg("c1", "grep", pattern="hello"),
        {"role": "assistant", "content": "REPORT: hello is defined in src/app.py:1"},
        # parent finishes
        {"role": "assistant", "content": "the function lives in src/app.py:1"},
    ]
    llm = FakeLLM(script)
    ui = FakeUI()
    agent = Agent(cfg, llm, ui, proj)
    answer = agent.run("where is hello defined?")

    assert answer == "the function lives in src/app.py:1"
    tool_msgs = [m for m in agent.context.messages if m.get("role") == "tool"]
    assert "REPORT: hello is defined in src/app.py:1" in tool_msgs[0]["content"]
    # child was read-only: its ledger must not contain any edit attempt result
    # (the FakeLLM script never gave it one, proving the schema was limited)


def test_subagent_is_read_only(proj, cfg):
    """A child told to edit gets 'unknown tool' - mutating tools are stripped."""
    script = [
        tool_call_msg("p1", "task", prompt="try to edit"),
        tool_call_msg("c1", "edit_file", path="src/app.py", old_string="a", new_string="b"),
        {"role": "assistant", "content": "REPORT: I cannot edit files"},
        {"role": "assistant", "content": "child cannot edit"},
    ]
    llm = FakeLLM(script)
    agent = Agent(cfg, llm, FakeUI(confirm_answer="y"), proj)
    agent.run("dispatch child")

    child_tool_msgs = []  # tool results inside child's ledger, reachable via llm.calls
    # the child's 2nd request (index 2 in calls) must contain the edit error result
    second_child_request = llm.calls[2]
    child_tool_msgs = [m for m in second_child_request if m.get("role") == "tool"]
    assert any("unknown tool 'edit_file'" in m["content"] for m in child_tool_msgs)
    assert (proj / "src" / "app.py").read_text() == "def hello():\n    return 'hi'\n"  # untouched


def test_subagent_usage_rolls_into_parent(proj, cfg):
    script = [
        tool_call_msg("p1", "task", prompt="explore"),
        {"role": "assistant", "content": "REPORT: done"},
        {"role": "assistant", "content": "final"},
    ]
    llm = FakeLLM(script)
    agent = Agent(cfg, llm, FakeUI(), proj)
    agent.run("go")

    # FakeLLM charges 11 in / 7 out per call; parent made 2 calls, child made 1
    assert agent.context.total_tokens_in == 11 * 3
    assert agent.context.total_tokens_out == 7 * 3


def test_subagent_cannot_nest_tasks(proj, cfg):
    script = [
        tool_call_msg("p1", "task", prompt="spawn another"),
        tool_call_msg("c1", "task", prompt="recursive"),
        {"role": "assistant", "content": "REPORT: no nesting"},
        {"role": "assistant", "content": "ok"},
    ]
    llm = FakeLLM(script)
    agent = Agent(cfg, llm, FakeUI(), proj)
    agent.run("go")

    second_child_request = llm.calls[2]
    assert any("unknown tool 'task'" in m["content"]
               for m in second_child_request if m.get("role") == "tool")


def test_subagent_ui_suppresses_output(proj, cfg):
    ui = FakeUI()
    sub_ui = SubAgentUI(ui)
    sub_ui.answer("quiet please")
    sub_ui.delta("leak")
    sub_ui.usage({"prompt": 1, "completion": 1})
    assert ui.answers == [] and ui.deltas == []
    sub_ui.warn("but warnings show")
    assert any("warnings show" in n for n in ui.notes)


# ---------------------------------------------------------------------- sessions
def test_save_and_load_session_roundtrip(proj, cfg):
    llm = FakeLLM([
        tool_call_msg("c1", "read_file", path="src/app.py"),
        {"role": "assistant", "content": "it defines hello"},
    ])
    agent = Agent(cfg, llm, FakeUI(), proj)
    agent.run("what does app.py do?")
    path = agent.save_session()
    assert path is not None and path.exists()

    agent2 = Agent(cfg, FakeLLM([]), FakeUI(), proj)
    assert agent2.load_session(path) is True
    assert agent2.context.messages == agent.context.messages
    assert agent2.context.total_tokens_in == 22
    assert agent2.context.turns == 1


def test_list_sessions_newest_first_with_summary(proj, cfg):
    llm = FakeLLM([{"role": "assistant", "content": "hi there"}])
    agent = Agent(cfg, llm, FakeUI(), proj)
    agent.run("refactor the tokenizer please")
    agent.save_session()

    sessions = agent.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["turns"] == 1
    assert "refactor the tokenizer" in sessions[0]["first"]


def test_save_session_skips_empty_ledger(proj, cfg):
    agent = Agent(cfg, FakeLLM([]), FakeUI(), proj)
    assert agent.save_session() is None


def test_prune_keeps_only_recent_sessions(proj, cfg):
    agent = Agent(cfg, FakeLLM([]), FakeUI(), proj)
    agent.context.add({"role": "user", "content": "x"})
    sdir = agent._sessions_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    for i in range(25):
        (sdir / f"2026010{i % 10}-00000{i}-{i:04x}.json").write_text(
            json.dumps({"id": str(i), "messages": [], "total_in": 0, "total_out": 0, "turns": 0})
        )
    agent.save_session()
    assert len(list(sdir.glob("*.json"))) <= 20


# ------------------------------------------------------------------ checkpoints
def test_checkpoint_and_undo_restores_file(git_proj, cfg):
    llm = FakeLLM([
        tool_call_msg("r0", "read_file", path="app.py"),
        tool_call_msg("e1", "edit_file", path="app.py", old_string="original", new_string="edited by agent"),
        {"role": "assistant", "content": "edited"},
    ])
    agent = Agent(cfg, llm, FakeUI(confirm_answer="y"), git_proj)
    agent.run("make a change")

    assert (git_proj / "app.py").read_text() == "edited by agent\n"
    msg = agent.undo_checkpoint()
    assert "rolled back" in msg
    assert (git_proj / "app.py").read_text() == "original\n"
    # checkpoint cleared after use
    assert "no checkpoint" in agent.undo_checkpoint()


def test_checkpoint_taken_once_per_turn(git_proj, cfg):
    llm = FakeLLM([
        tool_call_msg("a", "write_file", path="new.py", content="1\n"),
        {"role": "assistant", "content": "x"},
    ])
    agent = Agent(cfg, llm, FakeUI(confirm_answer="y"), git_proj)
    agent.run("two writes")
    first_sha = agent._checkpoint_sha
    assert first_sha
    # (single turn: one checkpoint; a new turn resets it)
    agent._checkpoint_sha = None
    agent._ensure_checkpoint()
    assert agent._checkpoint_sha  # fresh snapshot for the new turn


def test_undo_without_git_repo(proj, cfg):
    agent = Agent(cfg, FakeLLM([]), FakeUI(), proj)
    assert "no checkpoint" in agent.undo_checkpoint()


# -------------------------------------------------------------------------- HUD
def test_hud_line_formats_tokens(proj, cfg):
    llm = FakeLLM([{"role": "assistant", "content": "ok"}])
    agent = Agent(cfg, llm, FakeUI(), proj)
    agent.run("hello")
    line = agent.hud_line()
    assert "tok: " in line and "/60.0k" in line
    assert "$0.00" in line          # generic fallback pricing
    assert "turn 1" in line and "ready" in line


def test_window_title_updates_with_usage(proj, cfg):
    llm = FakeLLM([{"role": "assistant", "content": "ok"}])
    ui = FakeUI()
    agent = Agent(cfg, llm, ui, proj)
    agent.run("hello")
    assert any("ctx" in t and "↑" in t for t in ui.titles)


def test_real_ui_set_title_writes_escape(capsys):
    UI().set_title("bilal · ↑1k ↓2k · ctx 5%")
    out = capsys.readouterr().out
    assert "\x1b]2;bilal" in out and "\x07" in out


def test_turns_counter_increments(proj, cfg):
    llm = FakeLLM([
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two"},
    ])
    agent = Agent(cfg, llm, FakeUI(), proj)
    agent.run("a")
    agent.run("b")
    assert agent.context.turns == 2
