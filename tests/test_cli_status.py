from __future__ import annotations

from click.testing import CliRunner

from obsidian_llm_wiki.cli import cli
from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.pipeline.lock import pipeline_lock
from obsidian_llm_wiki.state import StateDB


def _init_status_vault(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / "wiki" / "sources").mkdir()
    (tmp_path / ".olw").mkdir()
    (tmp_path / "wiki.toml").write_text(
        """
[models]
fast = "test-fast"
heavy = "test-heavy"

[provider]
name = "ollama"
url = "http://localhost:11434"
""".strip()
    )
    return Config(vault=tmp_path)


def test_status_hides_released_lock_file(tmp_path):
    _init_status_vault(tmp_path)
    StateDB(tmp_path / ".olw" / "state.db")

    with pipeline_lock(tmp_path) as acquired:
        assert acquired is True

    result = CliRunner().invoke(cli, ["status", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    assert "Lock file present" not in result.output
    assert "Pipeline lock held" not in result.output


def test_status_shows_live_pipeline_lock(tmp_path):
    _init_status_vault(tmp_path)
    StateDB(tmp_path / ".olw" / "state.db")

    with pipeline_lock(tmp_path) as acquired:
        assert acquired is True
        result = CliRunner().invoke(cli, ["status", "--vault", str(tmp_path)])

    assert result.exit_code == 0
    assert "Pipeline lock held by PID" in result.output
