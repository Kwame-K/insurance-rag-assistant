"""Convert downloaded public documents into front-matter Markdown files.

Usage:
    uv run python scripts/convert_corpus.py
    uv run python scripts/convert_corpus.py --only ccq_assurances

Reads data/sources.yaml and data/external/raw/<id>.<format>, writes
data/external/markdown/<id>.md, then validates each file with the project's own
loader and chunker and prints a short report.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from insurance_rag_assistant.config import (
    CHUNK_MAX_CHARACTERS,
    CHUNK_OVERLAP_CHARACTERS,
)
from insurance_rag_assistant.ingestion.chunker import chunk_document
from insurance_rag_assistant.ingestion.convert import (
    count_headings,
    guideline_pdf_to_markdown,
    html_to_markdown,
    legal_pages_to_markdown,
    pdf_pages_text,
    render_document,
)
from insurance_rag_assistant.ingestion.loaders import load_markdown_document

ROOT = Path(__file__).resolve().parents[1]
SOURCES_FILE = ROOT / "data" / "sources.yaml"
RAW_DIR = ROOT / "data" / "external" / "raw"
OUTPUT_DIR = ROOT / "data" / "external" / "markdown"


def convert_source(source: dict[str, Any], raw_path: Path) -> tuple[str, str]:
    """Return (markdown body, short description) for one source."""
    default = "html" if source["format"] == "html" else "guideline_pdf"
    converter = source.get("converter") or default

    if converter == "html":
        body = html_to_markdown(raw_path.read_text(encoding="utf-8", errors="replace"))
        return body, "html"
    if converter == "guideline_pdf":
        return guideline_pdf_to_markdown(raw_path), "pdf (guideline)"
    if converter == "legal_pdf":
        body, stats = legal_pages_to_markdown(
            pdf_pages_text(raw_path), first_article=str(source["first_article"])
        )
        detail = (
            f"legal pdf: {stats['articles']} articles "
            f"({stats['first_article']} to {stats['last_article']})"
        )
        return body, detail
    message = f"Unknown converter {converter!r} for {source['id']}."
    raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", action="append", default=[], help="Source id (repeatable)."
    )
    args = parser.parse_args()

    sources = yaml.safe_load(SOURCES_FILE.read_text(encoding="utf-8"))
    if args.only:
        sources = [s for s in sources if s["id"] in args.only]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = 0

    for source in sources:
        source_id = source["id"]
        raw_path = RAW_DIR / f"{source_id}.{source['format']}"
        if not raw_path.exists():
            print(
                f"MISSING {source_id}: {raw_path} (run fetch_corpus.py first)",
                file=sys.stderr,
            )
            failures += 1
            continue

        try:
            body, detail = convert_source(source, raw_path)
            output_path = OUTPUT_DIR / f"{source_id}.md"
            output_path.write_text(render_document(source, body), encoding="utf-8")

            document = load_markdown_document(output_path)
            chunks = chunk_document(
                document=document,
                max_characters=CHUNK_MAX_CHARACTERS,
                overlap_characters=CHUNK_OVERLAP_CHARACTERS,
            )
        except (ValueError, TypeError, KeyError, ImportError) as error:
            failures += 1
            print(f"FAIL    {source_id}: {error}", file=sys.stderr)
            continue

        longest = max(len(chunk.text) for chunk in chunks)
        print(
            f"OK      {source_id}: {detail}; {len(body):,} chars, "
            f"{count_headings(body)} headings, {len(chunks)} chunks (longest {longest})"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
