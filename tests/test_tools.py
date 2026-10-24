"""Offline smoke tests for the tool layer - no network, no openai, no rich needed."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from bilal_agent.tools.base import AgentContext, ToolError
from bilal_agent.tools.read import glob_files, grep, list_dir, read_file
from bilal_agent.tools.shell import run_command
from bilal_agent.tools.todo import todo_write
from bilal_agent.tools.write import edit_file, write_file


@pytest.fixture()
def ctx(tmp_path: Path) -> AgentContext:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def hello():\n    return 'hi'\n\n\ndef bye():\n    return 'bye'\n")
    (tmp_path / "src" / "util.py").write_text("VALUE = 42\n")
    (tmp_path / "notes.md").write_text("# notes\nhello world\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    cfg = SimpleNamespace(command_timeout=10)
    return AgentContext(root=tmp_path, config=cfg)


# ------------------------------------------------------------------- read_file
def test_read_file_numbered(ctx):
    out = read_file(ctx, "src/app.py")
    assert "     1\tdef hello():" in out
    assert out.count("def ") == 2


def test_read_file_offset_hint(ctx):
    out = read_file(ctx, "src/app.py", offset=3, limit=2)
    assert "     3\t" in out
    assert "offset=" in out  # continuation hint present


def test_read_file_beyond_eof(ctx):
    with pytest.raises(ToolError, match="beyond end of file"):
        read_file(ctx, "src/app.py", offset=999)


def test_read_file_missing(ctx):
    with pytest.raises(ToolError, match="not found"):
        read_file(ctx, "nope.py")


def test_path_escape_denied(ctx):
    with pytest.raises(ToolError, match="escapes the project root"):
        read_file(ctx, "../outside.txt")
    with pytest.raises(ToolError, match="escapes"):
        read_file(ctx, "/etc/passwd")


# ---------------------------------------------------------------- list_dir/glob
def test_list_dir(ctx):
    out = list_dir(ctx)
    assert "src/" in out and "notes.md" in out


def test_glob(ctx):
    out = glob_files(ctx, "src/**/*.py")
    assert "src/app.py" in out and "src/util.py" in out


def test_glob_rejects_absolute(ctx):
    with pytest.raises(ToolError):
        glob_files(ctx, "/etc/*")


# ----------------------------------------------------------------------- grep
def test_grep_basic(ctx):
    out = grep(ctx, "def \\w+", path="src", file_glob="*.py")
    assert "src/app.py:1" in out and "src/app.py:5" in out


def test_grep_no_match(ctx):
    assert "no matches" in grep(ctx, "zzz_nowhere")


def test_grep_skips_git_dir(ctx):
    (ctx.root / ".git" / "leaked.py").write_text("def hello():\n    pass\n")
    assert ".git" not in grep(ctx, "def hello")


def test_grep_invalid_regex(ctx):
    with pytest.raises(ToolError, match="invalid regex"):
        grep(ctx, "([unclosed")


# ----------------------------------------------------------------- edit/write
def test_edit_requires_read_first(ctx):
    with pytest.raises(ToolError, match="read before it is modified"):
        edit_file(ctx, "src/app.py", "hi", "hello")


def test_edit_and_diff(ctx):
    read_file(ctx, "src/app.py")
    out = edit_file(ctx, "src/app.py", "return 'hi'", "return 'hello'")
    assert out.startswith("edited src/app.py")
    assert "+    return 'hello'" in out and "-    return 'hi'" in out
    assert "return 'hello'" in (ctx.root / "src" / "app.py").read_text()


def test_edit_ambiguous(ctx):
    read_file(ctx, "src/app.py")
    with pytest.raises(ToolError, match="matches 2 times"):
        edit_file(ctx, "src/app.py", "def ", "DEF ", replace_all=False)


def test_edit_replace_all(ctx):
    read_file(ctx, "src/app.py")
    out = edit_file(ctx, "src/app.py", "return", "RETURN", replace_all=True)
    assert "2 replacement(s)" in out


def test_edit_string_not_found(ctx):
    read_file(ctx, "src/app.py")
    with pytest.raises(ToolError, match="not found in file"):
        edit_file(ctx, "src/app.py", "DOES_NOT_EXIST", "x")


def test_write_new_file(ctx):
    out = write_file(ctx, "new/dir/file.txt", "content here")
    assert "created" in out
    assert (ctx.root / "new" / "dir" / "file.txt").read_text() == "content here"
    # newly written file counts as read -> can be edited right away
    edit_file(ctx, "new/dir/file.txt", "here", "there")


def test_write_existing_requires_read(ctx):
    with pytest.raises(ToolError, match="read before it is modified"):
        write_file(ctx, "notes.md", "clobber")


# --------------------------------------------------------------------- shell
def test_run_command(ctx):
    out = run_command(ctx, "echo hi")
    assert "exit code: 0" in out and "hi" in out


def test_run_command_captures_stderr_and_code(ctx):
    out = run_command(ctx, "echo oops >&2; exit 3")
    assert "exit code: 3" in out and "oops" in out


# ---------------------------------------------------------------------- todo
def test_todo_write(ctx):
    out = todo_write(ctx, [{"content": "step one", "status": "in_progress"}])
    assert "[~] step one" in out
    with pytest.raises(ToolError):
        todo_write(ctx, [{"content": "bad", "status": "done_yesterday"}])
