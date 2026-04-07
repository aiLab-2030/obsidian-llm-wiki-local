"""
Ingest pipeline: raw note → chunk → analyze → embed → update state.

Uses fast model (gemma4:e4b) for analysis.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path

from ..config import Config
from ..models import AnalysisResult, RawNoteRecord
from ..ollama_client import OllamaClient
from ..state import StateDB
from ..structured_output import request_structured
from ..vault import atomic_write, chunk_text, generate_aliases, parse_note, sanitize_filename

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a knowledge analyst. Read the provided note and extract structured information. "
    "Be concise and accurate. Do not invent information not present in the note."
)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_MAX_BODY_CHARS = 4000


def _build_analysis_prompt(body: str, existing_concepts: list[str], path_name: str = "") -> str:
    concepts_hint = ", ".join(existing_concepts[:30]) if existing_concepts else "none yet"
    if len(body) > _MAX_BODY_CHARS:
        log.warning(
            "Note %s truncated from %d to %d chars for analysis — "
            "concepts in later sections may be missed",
            path_name or "unknown",
            len(body),
            _MAX_BODY_CHARS,
        )
    body_trunc = body[:_MAX_BODY_CHARS]
    return (
        f"Analyze this note and extract structured metadata.\n\n"
        f"Existing wiki concepts (reuse these names where applicable): {concepts_hint}\n\n"
        f"NOTE CONTENT:\n{body_trunc}"
    )


def _normalize_concept_names(raw_names: list[str], db: StateDB) -> list[str]:
    """Case-insensitive match against existing canonical concept names.

    If a name matches an existing concept (case-insensitive), reuse the canonical form.
    Otherwise accept as-is. Deduplicates by canonical name.
    """
    existing = {n.lower(): n for n in db.list_all_concept_names()}
    seen: set[str] = set()
    normalized: list[str] = []
    for name in raw_names:
        name = name.strip()
        if not name:
            continue
        canonical = existing.get(name.lower(), name)
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return normalized


_HEADER_SCAN_LINES = 30  # only strip short lines from the opening section


def _preprocess_web_clip(content: str) -> str:
    """Clean common Obsidian Web Clipper artifacts (nav bars, cookie banners, HTML tags).

    Only the first _HEADER_SCAN_LINES are filtered — body content is preserved as-is.
    """
    # Remove residual HTML tags throughout
    content = re.sub(r"<[^>]+>", "", content)
    lines = content.splitlines()
    _MD_STARTS = ("#", "-", "*", ">", "[", "!")  # markdown structural chars — always keep

    cleaned = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # In the header region: skip short non-empty non-markdown lines (nav/banner heuristic)
        if (
            i < _HEADER_SCAN_LINES
            and stripped
            and len(stripped.split()) <= 5
            and not stripped.startswith(_MD_STARTS)
        ):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _create_source_summary_page(
    path: Path,
    meta: dict,
    result: AnalysisResult,
    config: Config,
) -> Path:
    """
    Generate wiki/sources/{Title}.md from AnalysisResult. No extra LLM call.
    Returns the path written.
    """
    # Derive title from note frontmatter > file stem
    title = meta.get("title") or path.stem.replace("-", " ").title()
    safe_name = sanitize_filename(title)
    out_path = config.sources_dir / f"{safe_name}.md"
    config.sources_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d")
    rel_raw = str(path.relative_to(config.vault))
    source_url = meta.get("source") or meta.get("url") or ""
    aliases = generate_aliases(title, "")  # source pages rarely have abbreviations

    # Build concept list as [[wikilinks]]
    concept_lines = "\n".join(f"- [[{c}]]" for c in result.key_concepts[:8] if c.strip())

    fm_lines = [
        "---",
        f"title: {title}",
        f"aliases: {aliases!r}",
        "tags: [source]",
        "status: published",
        f"source_file: {rel_raw}",
    ]
    if source_url:
        fm_lines.append(f"source_url: {source_url}")
    fm_lines += [
        f"quality: {result.quality}",
        f"created: {now}",
        "---",
        "",
        f"# {title}",
        "",
        "## Summary",
        result.summary,
        "",
        "## Concepts",
        concept_lines,
        "",
        "## Source Info",
        f"- **Quality:** {result.quality}",
        f"- **Raw file:** {rel_raw}",
        f"- **Ingested:** {now}",
    ]
    if source_url:
        fm_lines.append(f"- **URL:** {source_url}")

    atomic_write(out_path, "\n".join(fm_lines) + "\n")
    log.info("Source summary written: %s", out_path.name)
    return out_path


def ingest_note(
    path: Path,
    config: Config,
    client: OllamaClient,
    db: StateDB,
    rag=None,  # Optional RAGStore, injected in Phase 2
    existing_topics: list[str] | None = None,  # existing concept names for prompt
    force: bool = False,
) -> AnalysisResult | None:
    """
    Ingest a single raw note.

    Returns AnalysisResult or None if skipped (duplicate / already ingested).
    """
    content = path.read_text(encoding="utf-8")
    # Hash body only (strip frontmatter) so copies are detected as duplicates
    # even after ingest has updated the original's frontmatter (olw_status etc.)
    try:
        _, body_for_hash = parse_note(path)
    except Exception:
        body_for_hash = content
    h = _content_hash(body_for_hash)

    # Dedup check
    existing = db.get_raw_by_hash(h)
    if existing and existing.path != str(path.relative_to(config.vault)):
        log.info("Duplicate of %s, skipping %s", existing.path, path.name)
        return None

    rel_path = str(path.relative_to(config.vault))
    record = db.get_raw(rel_path)

    if record and record.status == "ingested" and not force:
        log.info("Already ingested: %s", path.name)
        return None

    # Pre-process web clips
    meta, body = parse_note(path)
    if meta.get("source") or meta.get("url"):  # web clipper adds these
        body = _preprocess_web_clip(body)

    # Chunk + embed only when RAG store is wired in (Phase 2)
    if rag is not None:
        chunks = chunk_text(
            body, chunk_size=config.rag.chunk_size, overlap=config.rag.chunk_overlap
        )
        embeddings = client.embed_batch(chunks, model=config.models.embed)
        rag.add_document(
            doc_id=rel_path,
            chunks=chunks,
            embeddings=embeddings,
            metadata={"source": rel_path, "type": "raw"},
        )

    # LLM analysis — use existing concept names so model can reuse canonical names
    if existing_topics is None:
        existing_topics = db.list_all_concept_names()
    prompt = _build_analysis_prompt(body, existing_topics, path_name=path.name)
    try:
        result: AnalysisResult = request_structured(
            client=client,
            prompt=prompt,
            model_class=AnalysisResult,
            model=config.models.fast,
            system=_SYSTEM,
            num_ctx=config.ollama.fast_ctx,
        )
    except Exception as e:
        log.error("Analysis failed for %s: %s", path.name, e)
        db.upsert_raw(
            RawNoteRecord(
                path=rel_path,
                content_hash=h,
                status="failed",
                error=str(e),
            )
        )
        return None

    # Update state DB (raw files stay immutable — metadata lives in state.db only)
    db.upsert_raw(
        RawNoteRecord(
            path=rel_path,
            content_hash=h,
            status="ingested",
            summary=result.summary,
            quality=result.quality,
            ingested_at=datetime.now(),
        )
    )

    # Normalize concept names against existing canonical names, store linkages
    max_concepts = config.pipeline.max_concepts_per_source
    normalized_concepts = _normalize_concept_names(result.key_concepts[:max_concepts], db)
    db.upsert_concepts(rel_path, normalized_concepts)

    # Create source summary page in wiki/sources/ (no extra LLM call)
    try:
        _create_source_summary_page(path, meta, result, config)
    except Exception as e:
        log.warning("Source summary page failed for %s: %s", path.name, e)

    log.info(
        "Ingested: %s (quality=%s, concepts=%s)", path.name, result.quality, result.key_concepts[:3]
    )
    return result


def ingest_all(
    config: Config,
    client: OllamaClient,
    db: StateDB,
    rag=None,
    force: bool = False,
) -> list[tuple[Path, AnalysisResult | None]]:
    """Ingest all .md files in raw/ (excluding raw/processed/ subfolders)."""
    raw_files = [
        p
        for p in config.raw_dir.rglob("*.md")
        if "processed" not in p.parts and not p.name.startswith(".")
    ]
    # Snapshot concept names once before loop (for consistent prompt context)
    existing_topics = db.list_all_concept_names()
    results = []
    for path in sorted(raw_files):
        result = ingest_note(
            path=path,
            config=config,
            client=client,
            db=db,
            rag=rag,
            existing_topics=existing_topics,
            force=force,
        )
        results.append((path, result))
    return results
