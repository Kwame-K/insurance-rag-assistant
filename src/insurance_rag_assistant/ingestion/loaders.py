import re
from pathlib import Path

import yaml

from insurance_rag_assistant.models.documents import (
    DocumentMetadata,
    SourceDocument,
)

FRONT_MATTER_PATTERN = re.compile(
    r"\A---\s*\n(?P<metadata>.*?)\n---\s*\n(?P<content>.*)\Z",
    re.DOTALL,
)


def load_markdown_document(path: Path) -> SourceDocument:
    """Load one Markdown document with YAML front matter."""
    raw_file_content = path.read_text(encoding="utf-8")
    match = FRONT_MATTER_PATTERN.match(raw_file_content)

    if match is None:
        message = f"Missing or invalid YAML front matter in: {path}"
        raise ValueError(message)

    metadata_data = yaml.safe_load(match.group("metadata"))

    if not isinstance(metadata_data, dict):
        message = f"Front matter must be a YAML mapping in: {path}"
        raise TypeError(message)

    raw_text = match.group("content").strip()

    if not raw_text:
        message = f"Document content is empty in: {path}"
        raise ValueError(message)

    return SourceDocument(
        metadata=DocumentMetadata.model_validate(metadata_data),
        source_path=path,
        raw_text=raw_text,
    )


def load_markdown_documents(source_dir: Path) -> list[SourceDocument]:
    """Load all Markdown documents in deterministic filename order."""
    paths = sorted(source_dir.glob("*.md"))

    if not paths:
        message = f"No Markdown files found in: {source_dir}"
        raise FileNotFoundError(message)

    return [load_markdown_document(path) for path in paths]
