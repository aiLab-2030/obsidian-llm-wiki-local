from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from obsidian_llm_wiki.config import Config
from obsidian_llm_wiki.ollama_client import OllamaClient
from obsidian_llm_wiki.state import StateDB


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Minimal vault structure for testing."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki" / ".drafts").mkdir()
    (tmp_path / ".olw").mkdir()
    return tmp_path


@pytest.fixture
def config(vault: Path) -> Config:
    return Config(vault=vault)


@pytest.fixture
def db(config: Config) -> StateDB:
    return StateDB(config.state_db_path)


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def make_mock_client(response: str = "{}") -> OllamaClient:
    """Return a mock OllamaClient that returns a fixed string from generate()."""
    client = MagicMock(spec=OllamaClient)
    client.generate.return_value = response
    client.embed_batch.return_value = [[0.1] * 768]
    client.embed.return_value = [0.1] * 768
    client.healthcheck.return_value = True
    return client
