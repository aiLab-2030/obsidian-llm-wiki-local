"""Tests for state.py SQLite tracking."""

from __future__ import annotations

import pytest

from obsidian_llm_wiki.models import RawNoteRecord, WikiArticleRecord
from obsidian_llm_wiki.state import StateDB


@pytest.fixture
def db(tmp_path):
    return StateDB(tmp_path / ".olw" / "state.db")


def test_upsert_and_get_raw(db):
    r = RawNoteRecord(path="raw/note.md", content_hash="abc123", status="new")
    db.upsert_raw(r)
    got = db.get_raw("raw/note.md")
    assert got is not None
    assert got.content_hash == "abc123"
    assert got.status == "new"


def test_dedup_by_hash(db):
    r1 = RawNoteRecord(path="raw/a.md", content_hash="samehash", status="new")
    r2 = RawNoteRecord(path="raw/b.md", content_hash="samehash", status="new")
    db.upsert_raw(r1)
    db.upsert_raw(r2)
    existing = db.get_raw_by_hash("samehash")
    assert existing is not None
    # Should find first occurrence
    assert existing.path == "raw/a.md"


def test_mark_ingested(db):
    db.upsert_raw(RawNoteRecord(path="raw/n.md", content_hash="h1"))
    db.mark_raw_status("raw/n.md", "ingested")
    got = db.get_raw("raw/n.md")
    assert got.status == "ingested"
    assert got.ingested_at is not None


def test_mark_failed_with_error(db):
    db.upsert_raw(RawNoteRecord(path="raw/n.md", content_hash="h2"))
    db.mark_raw_status("raw/n.md", "failed", error="LLM timeout")
    got = db.get_raw("raw/n.md")
    assert got.status == "failed"
    assert got.error == "LLM timeout"


def test_list_raw_by_status(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="new"))
    ingested = db.list_raw(status="ingested")
    assert len(ingested) == 1
    assert ingested[0].path == "raw/a.md"


def test_article_upsert_and_draft(db):
    a = WikiArticleRecord(
        path="wiki/.drafts/test.md",
        title="Test Article",
        sources=["raw/note.md"],
        content_hash="contenthash",
        is_draft=True,
    )
    db.upsert_article(a)
    got = db.get_article("wiki/.drafts/test.md")
    assert got is not None
    assert got.is_draft is True
    assert got.title == "Test Article"


def test_publish_article(db):
    a = WikiArticleRecord(
        path="wiki/.drafts/test.md",
        title="Test",
        sources=[],
        content_hash="h",
        is_draft=True,
    )
    db.upsert_article(a)
    db.publish_article("wiki/.drafts/test.md", "wiki/test.md")
    got = db.get_article("wiki/test.md")
    assert got is not None
    assert got.is_draft is False


# ── Concepts ──────────────────────────────────────────────────────────────────


def test_upsert_concepts_and_list(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Quantum Computing", "Qubit", "Shor's Algorithm"])
    names = db.list_all_concept_names()
    assert "Quantum Computing" in names
    assert "Qubit" in names
    assert len(names) == 3


def test_upsert_concepts_idempotent(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Quantum Computing", "Quantum Computing"])
    assert db.list_all_concept_names().count("Quantum Computing") == 1


def test_get_sources_for_concept(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Quantum Computing"])
    db.upsert_concepts("raw/b.md", ["Quantum Computing", "Machine Learning"])
    srcs = db.get_sources_for_concept("Quantum Computing")
    assert set(srcs) == {"raw/a.md", "raw/b.md"}
    ml_srcs = db.get_sources_for_concept("Machine Learning")
    assert ml_srcs == ["raw/b.md"]


def test_get_sources_case_insensitive(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_concepts("raw/a.md", ["Quantum Computing"])
    srcs = db.get_sources_for_concept("quantum computing")
    assert srcs == ["raw/a.md"]


def test_concepts_needing_compile(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="compiled"))
    db.upsert_concepts("raw/a.md", ["New Concept"])
    db.upsert_concepts("raw/b.md", ["Old Concept"])
    needing = db.concepts_needing_compile()
    assert "New Concept" in needing
    assert "Old Concept" not in needing  # source already compiled


def test_concepts_needing_compile_empty_when_all_compiled(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="compiled"))
    db.upsert_concepts("raw/a.md", ["Done Concept"])
    assert db.concepts_needing_compile() == []


def test_stats(db):
    db.upsert_raw(RawNoteRecord(path="raw/a.md", content_hash="h1", status="ingested"))
    db.upsert_raw(RawNoteRecord(path="raw/b.md", content_hash="h2", status="new"))
    db.upsert_article(
        WikiArticleRecord(
            path="wiki/.drafts/x.md", title="X", sources=[], content_hash="hx", is_draft=True
        )
    )
    s = db.stats()
    assert s["raw"]["ingested"] == 1
    assert s["raw"]["new"] == 1
    assert s["drafts"] == 1
    assert s["published"] == 0
