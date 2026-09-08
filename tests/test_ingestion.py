from pathlib import Path

from insurance_rag_assistant.ingestion.chunker import (
    chunk_document,
    parse_markdown_sections,
)
from insurance_rag_assistant.ingestion.loaders import load_markdown_document
from insurance_rag_assistant.ingestion.pipeline import ingest_markdown_corpus


def test_parse_markdown_sections() -> None:
    markdown = """# Policy

## Deductible

The deductible is CAD 2,500.

## Exclusions

Flood is excluded.
"""

    sections = parse_markdown_sections(markdown)

    assert len(sections) == 2
    assert sections[0].title == "Deductible"
    assert sections[0].path == ["Policy", "Deductible"]
    assert sections[1].title == "Exclusions"


def test_chunk_document_preserves_metadata() -> None:
    document = load_markdown_document(Path("data/raw/commercial_property_policy.md"))

    chunks = chunk_document(
        document=document,
        max_characters=1_600,
        overlap_characters=200,
    )

    assert chunks
    assert chunks[0].document_id == "commercial_property_policy_v1"
    assert chunks[0].coverage == "commercial_property"
    assert len(chunks[0].content_hash) == 64
    assert "Commercial Property Policy" in chunks[0].section_path


def test_ingestion_creates_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "chunks.jsonl"

    chunks = ingest_markdown_corpus(
        source_dir=Path("data/raw"),
        output_path=output_path,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert chunks
    assert output_path.exists()
    assert len(lines) == len(chunks)
