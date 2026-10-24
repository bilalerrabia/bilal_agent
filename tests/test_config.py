"""Tests for config loading + .env lookup (pure stdlib, offline)."""
from __future__ import annotations

import os
from pathlib import Path

from bilal_agent.config import _find_config, load_dotenv, load_config


def test_load_dotenv_first_file_wins(tmp_path: Path, monkeypatch):
    a, b = tmp_path / "a.env", tmp_path / "b.env"
    a.write_text("MY_KEY=from_a\n")
    b.write_text("MY_KEY=from_b\nOTHER=1\n")
    monkeypatch.delenv("MY_KEY", raising=False)
    monkeypatch.delenv("OTHER", raising=False)

    loaded = load_dotenv([a, b])
    assert os.environ["MY_KEY"] == "from_a"          # first file wins
    assert os.environ["OTHER"] == "1"                # missing keys still filled from later files
    assert loaded == [(a, ["MY_KEY"]), (b, ["OTHER"])]


def test_load_dotenv_never_overrides_real_env(tmp_path: Path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("MY_KEY=from_file\n")
    monkeypatch.setenv("MY_KEY", "from_shell")
    load_dotenv(f)
    assert os.environ["MY_KEY"] == "from_shell"


def test_load_dotenv_export_prefix_and_quotes(tmp_path: Path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('export A=1\nB="two words"\n# comment\nC=\n')
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    monkeypatch.delenv("C", raising=False)
    load_dotenv(f)
    assert os.environ["A"] == "1"
    assert os.environ["B"] == "two words"


def test_find_config_project_root_beats_cwd(tmp_path: Path, monkeypatch):
    proj, elsewhere = tmp_path / "proj", tmp_path / "elsewhere"
    proj.mkdir(); elsewhere.mkdir()
    (proj / "config.toml").write_text('active = "local"\n[providers.local]\nbase_url="http://x/v1"\nmodel="m"\napi_key_env=""\n')
    (elsewhere / "config.toml").write_text('active = "nope"\n')
    monkeypatch.chdir(elsewhere)  # cwd also has a config.toml -> must NOT win

    found = _find_config(None, root=proj)
    assert found == proj / "config.toml"
    cfg = load_config(root=proj)
    assert cfg.provider.name == "local"


def test_load_config_builtin_fallback_without_file(tmp_path: Path):
    cfg = load_config(root=tmp_path / "does-not-exist")
    assert cfg.provider.name == "zenmux"
    assert cfg.provider.api_key_env == "ZENMUX_API_KEY"
