import hashlib
import re
from dataclasses import dataclass

from insurance_rag_assistant.models.documents import (
    DocumentChunk,
    SourceDocument,
)

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WHITESPACE_PATTERN = re.compile(r"\s+")
SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class MarkdownSection:
    title: str
    path: list[str]
    content: str


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def slugify(value: str) -> str:
    normalized = normalize_whitespace(value).lower()
    normalized = (
        normalized.replace("à", "a")
        .replace("â", "a")
        .replace("ç", "c")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("ë", "e")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("ù", "u")
        .replace("û", "u")
        .replace("ü", "u")
    )
    return SLUG_PATTERN.sub("-", normalized).strip("-")


def content_hash(text: str) -> str:
    normalized_text = normalize_whitespace(text)
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def parse_markdown_sections(markdown: str) -> list[MarkdownSection]:
    """Extract sections while preserving the current hierarchical heading path."""
    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []

    def save_current_section() -> None:
        if not heading_stack:
            return

        content = "\n".join(current_lines).strip()
        if not content:
            return

        sections.append(
            MarkdownSection(
                title=heading_stack[-1][1],
                path=[heading for _, heading in heading_stack],
                content=content,
            )
        )

    for line in markdown.splitlines():
        match = HEADING_PATTERN.match(line)

        if match is None:
            current_lines.append(line)
            continue

        save_current_section()
        current_lines = []

        level = len(match.group(1))
        title = match.group(2).strip()

        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()

        heading_stack.append((level, title))

    save_current_section()

    if not sections:
        raise ValueError(
            "No Markdown headings found; cannot create section-aware chunks."
        )

    return sections


def split_text(text: str, max_characters: int, overlap_characters: int) -> list[str]:
    """Split normalized text into overlapping chunks without splitting words."""
    normalized_text = normalize_whitespace(text)

    if len(normalized_text) <= max_characters:
        return [normalized_text]

    words = normalized_text.split(" ")
    chunks: list[str] = []
    start_index = 0

    while start_index < len(words):
        current_words: list[str] = []
        current_length = 0
        word_index = start_index

        while word_index < len(words):
            word = words[word_index]
            separator_length = 1 if current_words else 0
            candidate_length = current_length + separator_length + len(word)

            if current_words and candidate_length > max_characters:
                break

            current_words.append(word)
            current_length = candidate_length
            word_index += 1

        chunks.append(" ".join(current_words))

        if word_index >= len(words):
            break

        overlap_length = 0
        next_start_index = word_index

        while next_start_index > start_index:
            previous_word = words[next_start_index - 1]
            added_length = len(previous_word) + (1 if overlap_length else 0)

            if overlap_length + added_length > overlap_characters:
                break

            overlap_length += added_length
            next_start_index -= 1

        if next_start_index == start_index:
            next_start_index = word_index

        start_index = next_start_index

    return chunks


def chunk_document(
    document: SourceDocument,
    max_characters: int,
    overlap_characters: int,
) -> list[DocumentChunk]:
    """Create deterministic section-aware chunks for a source document."""
    chunks: list[DocumentChunk] = []
    sections = parse_markdown_sections(document.raw_text)

    for section in sections:
        section_prefix = " > ".join(section.path)
        section_text = f"{section_prefix}\n\n{section.content}"
        text_parts = split_text(
            text=section_text,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        )
        section_slug = slugify("-".join(section.path))

        for text_part in text_parts:
            chunk_index = len(chunks)
            chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"{document.metadata.document_id}:"
                        f"{section_slug}:"
                        f"{chunk_index:03d}"
                    ),
                    document_id=document.metadata.document_id,
                    document_name=document.metadata.title,
                    document_type=document.metadata.document_type,
                    coverage=document.metadata.coverage,
                    jurisdiction=document.metadata.jurisdiction,
                    language=document.metadata.language,
                    version=document.metadata.version,
                    effective_date=document.metadata.effective_date,
                    section_title=section.title,
                    section_path=section.path,
                    chunk_index=chunk_index,
                    text=text_part,
                    content_hash=content_hash(text_part),
                )
            )

    return chunks
