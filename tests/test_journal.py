"""Tests for the persistent modification journal (.bilal/JOURNAL.md).

The journal is written by the CODE on every applied modification - never by the
model - so future sessions can skip re-exploring the whole project.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bilal_agent.cli import _env_candidates
from bilal_agent.system_prompt import build_system_prompt
from bilal_agent.tools.base import AgentContext
from bilal_agent.tools.read import read_file
from bilal_agent.tools.shell import run_command
from bilal_agent.tools.write import edit_file, write_file
from bilal_agent.ui.render import _word_art


@pytest.fixture()
def ctx(tmp_path: Path) -> AgentContext:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'hi'\n")
    return AgentContext(root=tmp_path, config=SimpleNamespace(command_timeout=10))


def _entries(ctx: AgentContext) -> list[str]:
    return [
        ln for ln in ctx.journal_path.read_text(encoding="utf-8").splitlines()
        if ln.startswith("- ")
    ]


# ------------------------------------------------------------------ journaling
def test_edit_file_writes_journal_entry(ctx):
    read_file(ctx, "src/app.py")
    edit_file(ctx, "src/app.py", "hi", "hello")
    entries = _entries(ctx)

    assert len(entries) == 1
    assert "[edit_file]" in entries[0]
    assert "src/app.py" in entries[0]
    assert "+1/-1" in entries[0]  # diff stat recorded
    assert ctx.journal_path.parent.name == ".bilal"


def test_write_file_writes_journal_entry(ctx):
    write_file(ctx, "src/new.py", "X = 1\n")
    entries = _entries(ctx)

    assert len(entries) == 1
    assert "[write_file]" in entries[0]
    assert "src/new.py" in entries[0]
    assert "created" in entries[0]


def test_failed_edit_writes_nothing(ctx):
    with pytest.raises(Exception):
        edit_file(ctx, "src/app.py", "not-in-file", "x")
    assert not ctx.journal_path.exists()


def test_run_command_writes_journal_entry(ctx):
    run_command(ctx, "echo ok")
    entries = _entries(ctx)

    assert len(entries) == 1
    assert "[shell]" in entries[0]
    assert "echo ok" in entries[0]
    assert "exit 0" in entries[0]


def test_journal_appends_across_calls(ctx):
    read_file(ctx, "src/app.py")
    edit_file(ctx, "src/app.py", "hi", "hello")
    write_file(ctx, "src/new.py", "X = 1\n")
    entries = _entries(ctx)

    assert len(entries) == 2
    assert "[edit_file]" in entries[0] and "[write_file]" in entries[1]


# ------------------------------------------------------- system prompt wiring
def test_system_prompt_includes_journal_tail(ctx):
    read_file(ctx, "src/app.py")
    edit_file(ctx, "src/app.py", "hi", "hello")
    prompt = build_system_prompt(ctx.root)

    assert "<journal>" in prompt
    assert "[edit_file] src/app.py" in prompt
    assert "newest last" in prompt


def test_system_prompt_omits_journal_when_absent(ctx):
    prompt = build_system_prompt(ctx.root)
    # rule 2 mentions the <journal> tag by name, so check for the actual section marker
    assert "(auto-written log of modifications" not in prompt


def test_journal_tail_is_capped(ctx):
    ctx.journal_path.parent.mkdir(exist_ok=True)
    ctx.journal_path.write_text("# header\n" + "".join(f"- line {i}\n" for i in range(200)))
    prompt = build_system_prompt(ctx.root)

    assert "older entries omitted" in prompt
    assert "- line 199" in prompt  # newest entries kept
    assert "- line 0\n" not in prompt  # oldest dropped


def test_journal_dir_is_skipped_by_search(tmp_path):
    from bilal_agent.tools.base import SKIP_DIRS

    assert ".bilal" in SKIP_DIRS


# ------------------------------------------------------------------ misc bits
def test_env_candidates_agent_folder_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # cwd == "somewhere else"
    root = tmp_path / "target-project"
    repo = tmp_path / "bilal_agent-repo"

    candidates = _env_candidates(root, repo_dir=repo)

    assert candidates[0] == repo / ".env"                    # agent repo first
    assert candidates[-1] == root / ".env"                   # target project LAST
    assert Path.home() / ".config" / "bilal" / ".env" in candidates  # tool-install fallback


def test_banner_word_art_rows():
    art = _word_art("BILAL")
    assert len(art) == 6                       # 6 rows per block letter
    assert all(len(row) > 20 for row in art)   # five letters wide
    assert art[0].startswith("█")
    assert art[-1].startswith("╚")             # bottom shadow row
