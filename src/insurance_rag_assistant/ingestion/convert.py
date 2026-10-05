"""Convert downloaded public documents (HTML, PDF) into front-matter Markdown.

The output matches what ``loaders.load_markdown_document`` expects: a YAML front
matter block followed by Markdown with ``#`` headings that the chunker uses to
build section paths. Heavy PDF/HTML libraries are imported lazily so that the
pure text helpers stay importable (and testable) without them.
"""

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pymupdf4llm  # type: ignore[import-untyped]
import yaml

FRONT_MATTER_FIELDS = (
    "document_id",
    "title",
    "document_type",
    "coverage",
    "jurisdiction",
    "language",
    "version",
    "effective_date",
    "status",
)

_HEADINGS = {f"h{level}": level for level in range(1, 7)}
_NOISE_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "aside",
    "header",
    "footer",
    "form",
    "button",
    "svg",
    "iframe",
)
_CONTAINERS = {
    "div",
    "section",
    "article",
    "main",
    "blockquote",
    "details",
    "figure",
    "fieldset",
    "body",
}
_BLOCKS = {*_HEADINGS, *_CONTAINERS, "p", "ul", "ol", "dl", "table", "pre"}
_TOC_TITLES = {"table of contents", "table des matières", "table des matieres"}

_ARTICLE = re.compile(r"^(?P<number>\d{3,4}(?:\.\d+)?)\.\s*(?P<rest>.*)$")
_STRUCTURE_LEVELS = (
    (re.compile(r"^CHAPITRE\b"), 2),
    (re.compile(r"^SECTION\b"), 3),
    (re.compile(r"^§\s*\d"), 4),
    (re.compile(r"^[IVX]{1,5}\.\s*[—–-]"), 5),
)
_ARTICLE_LEVEL = 6
_AMENDMENT_NOTE = re.compile(r"^\d{4},\s*c\.\s*\d+")
_NUMBERED_HEADING = re.compile(r"^(?:\*\*)?(\d{1,2})\.\s+([A-Z][^\n]{3,90}?)(?:\*\*)?$")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


# ---------------------------------------------------------------- front matter
def build_front_matter(source: dict[str, Any]) -> str:
    """Build the YAML front matter from a ``sources.yaml`` entry."""
    values: dict[str, Any] = {"document_id": source["id"]}
    for field in FRONT_MATTER_FIELDS[1:]:
        values[field] = str(source[field])
    dumped = yaml.safe_dump(values, sort_keys=False, allow_unicode=True)
    return f"---\n{dumped}---\n"


def render_document(source: dict[str, Any], body: str) -> str:
    return f"{build_front_matter(source)}\n{ensure_title(body, source['title'])}\n"


def ensure_title(markdown: str, title: str) -> str:
    if re.search(r"^# \S", markdown, re.MULTILINE):
        return markdown.strip()
    return f"# {title}\n\n{markdown.strip()}"


def clean_markdown(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\u200b", "")
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def count_headings(markdown: str) -> int:
    return len(_MARKDOWN_HEADING.findall(markdown))


# ----------------------------------------------------------------------- HTML
def html_to_markdown(html: str) -> str:
    """Convert the ``<main>`` content of a web page to Markdown."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("main") or soup.body or soup

    for tag in root.find_all(_NOISE_TAGS):
        tag.decompose()
    _drop_table_of_contents(root)
    for line_break in root.find_all("br"):
        line_break.replace_with("\n")

    blocks: list[str] = []
    _walk(root, blocks, list_depth=0)
    return clean_markdown("\n\n".join(block for block in blocks if block.strip()))


def _drop_table_of_contents(root: Any) -> None:
    for heading in root.find_all(list(_HEADINGS)):
        if heading.get_text(" ", strip=True).lower() in _TOC_TITLES:
            following = heading.find_next_sibling()
            if following is not None and following.name in {"ul", "ol", "nav", "div"}:
                following.decompose()
            heading.decompose()


def _inline(node: Any) -> str:
    return " ".join(node.get_text().split())


def _has_block_child(tag: Any) -> bool:
    return any(getattr(child, "name", None) in _BLOCKS for child in tag.children)


def _walk(node: Any, blocks: list[str], list_depth: int) -> None:
    loose: list[str] = []

    def flush() -> None:
        text = " ".join(" ".join(loose).split())
        if text:
            blocks.append(text)
        loose.clear()

    for child in node.children:
        name = getattr(child, "name", None)
        if name is None:
            if type(child).__name__ == "NavigableString":
                loose.append(str(child))
            continue
        if name in _HEADINGS:
            flush()
            text = _inline(child)
            if text:
                blocks.append(f"{'#' * _HEADINGS[name]} {text}")
        elif name == "p":
            flush()
            text = _inline(child)
            if text:
                blocks.append(text)
        elif name in {"ul", "ol"}:
            flush()
            blocks.extend(_list_items(child, list_depth))
        elif name == "dl":
            flush()
            blocks.extend(_definition_items(child))
        elif name == "table":
            flush()
            table = _table(child)
            if table:
                blocks.append(table)
        elif name == "pre":
            flush()
            blocks.append(child.get_text().strip())
        elif name in _CONTAINERS:
            flush()
            if _has_block_child(child):
                _walk(child, blocks, list_depth)
            else:
                text = _inline(child)
                if text:
                    blocks.append(text)
        else:
            loose.append(child.get_text())
    flush()


def _list_items(list_tag: Any, depth: int) -> list[str]:
    items: list[str] = []
    ordered = list_tag.name == "ol"
    for index, item in enumerate(list_tag.find_all("li", recursive=False), start=1):
        marker = f"{index}." if ordered else "-"
        nested = item.find_all(["ul", "ol"], recursive=False)
        for sub in nested:
            sub.extract()
        text = _inline(item)
        if text:
            items.append(f"{'  ' * depth}{marker} {text}")
        for sub in nested:
            items.extend(_list_items(sub, depth + 1))
    return ["\n".join(items)] if items else []


def _definition_items(definition_list: Any) -> list[str]:
    items: list[str] = []
    term = ""
    for child in definition_list.find_all(["dt", "dd"], recursive=False):
        if child.name == "dt":
            term = _inline(child)
        else:
            definition = _inline(child)
            items.append(f"**{term}**: {definition}" if term else definition)
    return items


def _table(table: Any) -> str:
    rows: list[list[str]] = []
    for row in table.find_all("tr"):
        cells = [_inline(cell).replace("|", "/") for cell in row.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(padded[0]) + " |", "|" + " --- |" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    caption = table.find("caption")
    prefix = (
        f"**{_inline(caption)}**\n\n"
        if caption is not None and _inline(caption)
        else ""
    )
    return prefix + "\n".join(lines)


# ------------------------------------------------------------ page boilerplate
def _edge_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _has_letters(line: str) -> bool:
    return len(re.findall(r"[^\W\d_]", line)) >= 3


def strip_page_boilerplate(
    pages: Sequence[str],
    edge_lines: int = 2,
    min_ratio: float = 0.3,
    anywhere_ratio: float = 0.5,
    anywhere_min_pages: int = 10,
) -> list[str]:
    """Remove running headers and footers.

    Two rules, both ignoring lines without letters (so article numbers such as
    ``2390.`` at the top of a page are never removed):
    - short lines repeated at page edges (headers, footers, 'Page 3');
    - short lines repeated on at least half of the pages wherever they appear
      (headers that PDF text extraction places mid-page); long documents only.
    """
    split = [page.splitlines() for page in pages]

    def edge_indexes(lines: list[str]) -> set[int]:
        indexes = [i for i, line in enumerate(lines) if line.strip()]
        if len(indexes) <= 3 * edge_lines:
            return set()
        return set(indexes[:edge_lines] + indexes[-edge_lines:])

    edge_counts: Counter[str] = Counter()
    page_counts: Counter[str] = Counter()
    for lines in split:
        edge_counts.update(
            {
                _edge_key(lines[i])
                for i in edge_indexes(lines)
                if _has_letters(lines[i]) and len(lines[i]) <= 90
            }
        )
        page_counts.update(
            {
                _edge_key(line)
                for line in lines
                if _has_letters(line) and len(line.strip()) <= 80
            }
        )

    edge_threshold = max(3, int(len(pages) * min_ratio))
    edge_repeated = {
        key for key, count in edge_counts.items() if count >= edge_threshold
    }
    anywhere_repeated: set[str] = set()
    if len(pages) >= anywhere_min_pages:
        anywhere_threshold = int(len(pages) * anywhere_ratio)
        anywhere_repeated = {
            k for k, c in page_counts.items() if c >= anywhere_threshold
        }

    cleaned: list[str] = []
    for lines in split:
        edges = edge_indexes(lines)
        kept: list[str] = []
        for i, line in enumerate(lines):
            key = _edge_key(line)
            if key in anywhere_repeated and _has_letters(line):
                continue
            if i in edges and (
                line.strip().isdigit() or (_has_letters(line) and key in edge_repeated)
            ):
                continue
            kept.append(line)
        cleaned.append("\n".join(kept))
    return cleaned


# ------------------------------------------------------------------ legal PDF
def legal_pages_to_markdown(
    pages: Sequence[str], first_article: str
) -> tuple[str, dict[str, Any]]:
    """Turn the text of a statute PDF into Markdown, starting at ``first_article``.

    Chapters, sections and subsections become headings; each article becomes a
    level-6 heading ("Article 2389") followed by its text. Amendment notes are dropped.
    """
    lines = "\n".join(strip_page_boilerplate(pages)).splitlines()
    lines = [line.strip() for line in lines]

    start = next(
        (
            i
            for i, line in enumerate(lines)
            if (m := _ARTICLE.match(line)) and m.group("number") == first_article
        ),
        None,
    )
    if start is None:
        message = f"Article {first_article} not found in the PDF text."
        raise ValueError(message)

    for i in range(start, max(start - 40, -1), -1):
        if lines[i].startswith("CHAPITRE"):
            start = i
            break
    chapter_start = start

    end = len(lines)
    for i in range(chapter_start + 1, len(lines)):
        if re.match(r"^(CHAPITRE|LIVRE|TITRE)\b", lines[i]):
            end = i
            break

    body = _structure_lines(lines[chapter_start:end])
    markdown = clean_markdown("\n\n".join(body))
    articles = re.findall(r"^#{6} Article (\d{3,4}(?:\.\d+)?)$", markdown, re.MULTILINE)
    stats = {
        "articles": len(articles),
        "first_article": articles[0] if articles else None,
        "last_article": articles[-1] if articles else None,
        "headings": count_headings(markdown),
    }
    return markdown, stats


def _heading_level(line: str) -> int | None:
    for pattern, level in _STRUCTURE_LEVELS:
        if pattern.match(line):
            return level
    return None


def _is_uppercase_title(line: str) -> bool:
    letters = [char for char in line if char.isalpha()]
    return bool(letters) and all(char.isupper() for char in letters)


def _structure_lines(lines: Sequence[str]) -> list[str]:
    blocks: list[str] = []
    paragraph: list[str] = []
    in_note = False

    def flush() -> None:
        if paragraph:
            blocks.append(" ".join(paragraph))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line:
            continue

        level = _heading_level(line)
        article = _ARTICLE.match(line)

        if level is not None:
            flush()
            in_note = False
            title = line
            while (
                index < len(lines)
                and lines[index]
                and _is_uppercase_title(lines[index])
                and _heading_level(lines[index]) is None
                and not _ARTICLE.match(lines[index])
            ):
                title += f" - {lines[index]}"
                index += 1
            blocks.append(f"{'#' * level} {title}")
        elif article:
            flush()
            in_note = False
            blocks.append(f"{'#' * _ARTICLE_LEVEL} Article {article.group('number')}")
            if article.group("rest"):
                paragraph.append(article.group("rest"))
        elif _AMENDMENT_NOTE.match(line):
            in_note = True
        elif not in_note:
            paragraph.append(line)
    flush()
    return blocks


# ------------------------------------------------------------- guideline PDF
def promote_numbered_headings(markdown: str) -> str:
    """Fallback when the PDF has no detectable headings: '3. Title' -> '## 3. Title'."""
    promoted: list[str] = []
    for line in markdown.splitlines():
        match = _NUMBERED_HEADING.match(line.strip())
        if match and not line.rstrip().endswith("."):
            promoted.append(f"## {match.group(1)}. {match.group(2).strip('*').strip()}")
        else:
            promoted.append(line)
    return "\n".join(promoted)


def guideline_pdf_to_markdown(path: Path) -> str:
    """Convert a regulatory-guideline PDF with PyMuPDF4LLM (font-based headings)."""

    chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    pages = strip_page_boilerplate([chunk["text"] for chunk in chunks])
    markdown = clean_markdown("\n\n".join(pages))
    if count_headings(markdown) < 3:
        markdown = promote_numbered_headings(markdown)
    return markdown


def pdf_pages_text(path: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(path) as document:
        return [page.get_text("text") for page in document]
