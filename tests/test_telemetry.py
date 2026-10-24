"""Tests for v0.4.2: telemetry stats (cost/latency/per-turn/per-tool), budget
warnings, the modern TUI surface (header, user echo, thinking panel, tools tree,
/cost, /stats) and stats persistence in sessions."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bilal_agent.agent import Agent
from bilal_agent.config import resolve_pricing
from bilal_agent.llm import _usage_to_dict
from bilal_agent.ui.render import UI
from tests.test_agent_loop import FakeLLM, FakeUI, tool_call_msg


@pytest.fixture()
def proj(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'hi'\n")
    return tmp_path


def _cfg(**kw) -> SimpleNamespace:
    base = dict(
        max_iterations=5, max_output_tokens=8192, llm_timeout=300, stream=True,
        max_context_tokens=60000, keep_recent=8, confirm_edits=True, auto_yes=False,
        command_timeout=10, temperature=None, pricing=None, budget_usd=None,
        provider=SimpleNamespace(model="test-model", name="test", base_url="x", api_key_env=""),
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------- stats
def test_stats_after_tool_round(proj):
    llm = FakeLLM([
        tool_call_msg("c1", "read_file", path="src/app.py"),
        {"role": "assistant", "content": "answer"},
    ])
    agent = Agent(_cfg(), llm, FakeUI(), proj)
    agent.run("what?")

    assert agent.stats["api_calls"] == 2
    assert agent.stats["latency_sum"] >= 0.0
    rf = agent.stats["tools"]["read_file"]
    assert rf["calls"] == 1 and rf["errors"] == 0
    assert len(agent.stats["turns"]) == 1
    t = agent.stats["turns"][0]
    assert t["in"] == 22 and t["out"] == 14          # 2 calls x (11 in, 7 out)
    assert t["tools"] == 1 and t["cost"] > 0
    assert "ctx: 1 files" in agent.hud_line()        # read_files counted


def test_budget_warnings_fire_in_stages(proj):
    cfg = _cfg(budget_usd=0.000039)  # call 1 ~82% -> 80% warn, call 2 ~164% -> exceeded
    llm = FakeLLM([
        {"role": "assistant", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "assistant", "content": "three"},
    ])
    agent = Agent(cfg, llm, FakeUI(), proj)
    agent.run("a")
    assert any("80%" in n for n in agent.ui.notes)
    agent.run("b")
    assert any("exceeded" in n for n in agent.ui.notes)
    assert agent._budget_stage == 2
    # /reset-budget semantics: zero the counter -> warnings re-arm
    agent.stats["cost"] = 0.0
    agent._budget_stage = 0
    agent._account_usage({"prompt": 11, "completion": 7})
    assert any("80%" in n for n in agent.ui.notes)


def test_no_budget_warning_without_budget(proj):
    llm = FakeLLM([{"role": "assistant", "content": "ok"}])
    agent = Agent(_cfg(), llm, FakeUI(), proj)
    agent.run("hi")
    assert not any("budget" in n for n in agent.ui.notes)


def test_hud_line_shows_budget(proj):
    agent = Agent(_cfg(budget_usd=2.0), FakeLLM([{"role": "assistant", "content": "ok"}]),
                  FakeUI(), proj)
    agent.run("hi")
    assert "$0.00/$2.00" in agent.hud_line()


# ------------------------------------------------------------ UI surfaces
def test_intermediate_plan_rendered_as_thinking_panel(proj):
    plan = tool_call_msg("c1", "read_file", path="src/app.py")
    plan["content"] = "01 Inspect the file\n02 Report findings"
    llm = FakeLLM([plan, {"role": "assistant", "content": "done"}])
    ui = FakeUI()
    Agent(_cfg(), llm, ui, proj).run("go")
    assert ui.thinking_panels and "01 Inspect" in ui.thinking_panels[0]


def test_tools_group_lifecycle_and_mutation_hint(proj):
    cfg = _cfg(auto_yes=True)
    llm = FakeLLM([
        tool_call_msg("r1", "read_file", path="src/app.py"),
        tool_call_msg("e1", "edit_file", path="src/app.py",
                      old_string="hi", new_string="hello"),
        {"role": "assistant", "content": "edited"},
    ])
    ui = FakeUI()
    agent = Agent(cfg, llm, ui, proj)
    agent.run("edit it")

    # read and edit were separate model rounds -> two groups of one call each
    assert ui.tools_groups == [1, 1] and ui.tools_ends == [True, True]
    assert agent._turn_mutations == 1
    assert any("/diff" in n and "/undo" in n for n in ui.notes)   # mockup action row
    assert agent.stats["tools"]["edit_file"]["errors"] == 0


def test_denied_tool_counts_as_error(proj):
    llm = FakeLLM([
        tool_call_msg("r1", "run_command", command="echo hi"),
        {"role": "assistant", "content": "ok"},
    ])
    agent = Agent(_cfg(), llm, FakeUI(), proj)
    agent.run("run it")
    assert agent.stats["tools"]["run_command"]["errors"] == 1
    assert agent.stats["tools"]["run_command"]["calls"] == 1


def test_unknown_tool_is_visible_in_tree(proj):
    llm = FakeLLM([
        tool_call_msg("u1", "teleport", where="moon"),
        {"role": "assistant", "content": "ok"},
    ])
    ui = FakeUI()
    Agent(_cfg(), llm, ui, proj).run("teleport")
    assert ui.tools_groups == [1] and ui.tools_ends == [True]
    assert any(t[0] == "teleport" and t[2] for t in ui.tool_results)  # visible, marked error


# ------------------------------------------------------------- real renders
def test_tool_tree_renders_inline_results(proj, capsys):
    ui = UI()
    llm = FakeLLM([
        tool_call_msg("s1", "run_command", command="echo hi"),
        {"role": "assistant", "content": "done"},
    ])
    Agent(_cfg(auto_yes=True), llm, ui, proj).run("run echo")
    out = capsys.readouterr().out
    assert "agent::tools" in out
    assert "$ echo hi" in out
    assert "✓" in out and "╰─" in out


def test_user_echo_and_answer_label(proj, capsys):
    ui = UI()
    llm = FakeLLM([{"role": "assistant", "content": "hello there"}])
    Agent(_cfg(), llm, ui, proj).run("say hi")
    out = capsys.readouterr().out
    assert "❯ user" in out and "say hi" in out          # timestamped user anchor
    assert "◆ bilal_agent" in out and "ready" in out    # response label


def test_modern_banner_renders(tmp_path, capsys):
    UI().banner("0.4.2", "zenmux", "glm-test", tmp_path, False)
    out = capsys.readouterr().out
    assert "bilal_agent" in out and "0.4.2" in out and "ONLINE" in out
    assert str(tmp_path) in out and "glm-test" in out
    assert "session::active" in out and "/stats" in out


def test_telemetry_dashboard_renders(proj, capsys):
    ui = UI()
    llm = FakeLLM([
        tool_call_msg("c1", "read_file", path="src/app.py"),
        {"role": "assistant", "content": "answer"},
    ])
    agent = Agent(_cfg(budget_usd=2.0), llm, ui, proj)
    agent.run("look")
    ui.telemetry(agent)
    out = capsys.readouterr().out
    assert "SESSION TELEMETRY" in out
    assert "CONSUMPTION PER TURN" in out
    assert "TOOL INVOCATIONS" in out and "read_file" in out
    assert "RUNTIME ENVIRONMENT" in out
    assert "test-model" in out


def test_cost_card_renders(proj, capsys):
    ui = UI()
    llm = FakeLLM([{"role": "assistant", "content": "ok"}])
    agent = Agent(_cfg(budget_usd=5.0), llm, ui, proj)
    agent.run("hi")
    ui.cost_card(agent)
    out = capsys.readouterr().out
    assert "COST" in out and "$5.00" in out and "In:" in out


# ------------------------------------------------------------------ pricing
def test_resolve_pricing_precedence():
    override = SimpleNamespace(pricing={"input_per_m": 2.0, "output_per_m": 9.0},
                               provider=SimpleNamespace(model="claude-sonnet-4"))
    pin, pout, src = resolve_pricing(override)
    assert (pin, pout) == (2.0, 9.0) and "[pricing]" in src

    listed = SimpleNamespace(pricing={}, provider=SimpleNamespace(model="anthropic/claude-sonnet-4.5"))
    pin, pout, src = resolve_pricing(listed)
    assert (pin, pout) == (3.0, 15.0) and "claude-sonnet" in src

    mystery = SimpleNamespace(pricing={}, provider=SimpleNamespace(model="mystery-model"))
    assert resolve_pricing(mystery)[2].startswith("generic fallback")


def test_usage_to_dict_captures_cached_tokens():
    u = SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=8))
    assert _usage_to_dict(u) == {"prompt": 10, "completion": 5, "cached": 8}
    u2 = SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None)
    assert _usage_to_dict(u2) == {"prompt": 10, "completion": 5}


# ------------------------------------------------------------- persistence
def test_session_roundtrip_keeps_stats(proj):
    llm = FakeLLM([{"role": "assistant", "content": "ok"}])
    agent = Agent(_cfg(), llm, FakeUI(), proj)
    agent.run("hi")
    agent.save_session()

    agent2 = Agent(_cfg(), FakeLLM([]), FakeUI(), proj)
    assert agent2.load_session(agent.last_saved_path)
    assert agent2.stats["api_calls"] == 1
    assert abs(agent2.stats["cost"] - agent.stats["cost"]) < 1e-9
    assert agent2.stats["turns"][0]["in"] == 11


def test_diff_summary_without_git(proj):
    agent = Agent(_cfg(), FakeLLM([]), FakeUI(), proj)
    assert "not a git repo" in agent.diff_summary()
