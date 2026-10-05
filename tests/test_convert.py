import pytest
import yaml

from insurance_rag_assistant.ingestion.convert import (
    build_front_matter,
    clean_markdown,
    count_headings,
    ensure_title,
    html_to_markdown,
    legal_pages_to_markdown,
    promote_numbered_headings,
    render_document,
    strip_page_boilerplate,
)

SOURCE = {
    "id": "demo_doc",
    "title": "Demo Document",
    "document_type": "regulatory_guideline",
    "coverage": "multi_line",
    "jurisdiction": "canada",
    "language": "en",
    "version": "2024-08-22",
    "effective_date": "2024-08-22",
    "status": "active",
}


def test_front_matter_round_trips_through_yaml() -> None:
    data = yaml.safe_load(build_front_matter(SOURCE).strip("-\n"))
    assert data["document_id"] == "demo_doc"
    assert data["version"] == "2024-08-22"
    assert list(data) == [
        "document_id",
        "title",
        "document_type",
        "coverage",
        "jurisdiction",
        "language",
        "version",
        "effective_date",
        "status",
    ]


def test_render_document_adds_title_when_missing() -> None:
    text = render_document(SOURCE, "## Section\n\nBody.")
    assert text.startswith("---\n")
    assert "\n# Demo Document\n" in text


def test_ensure_title_keeps_existing_h1() -> None:
    assert ensure_title("# Own title\n\nBody", "Other") == "# Own title\n\nBody"


def test_clean_markdown_normalises_whitespace() -> None:
    assert clean_markdown("a\xa0b  \n\n\n\n\nc") == "a b\n\nc"


def test_strip_page_boilerplate_removes_running_header_and_page_numbers() -> None:
    pages = [
        f"Guideline E-21\nOSFI\nLine one of page {n}\nLine two {n}\n"
        f"Line three\nLine four\nLine five\nPage {n}"
        for n in range(1, 8)
    ]
    cleaned = strip_page_boilerplate(pages)
    assert all("Guideline E-21" not in page for page in cleaned)
    assert all("Page " not in page for page in cleaned)
    assert "Line one of page 3" in cleaned[2]


def test_strip_page_boilerplate_leaves_short_pages_alone() -> None:
    pages = [f"Body of page {n}" for n in range(1, 8)]
    assert strip_page_boilerplate(pages) == pages


def test_promote_numbered_headings() -> None:
    result = promote_numbered_headings(
        "1. Purpose and Scope\nText line.\n2. Governance Model"
    )
    assert result.splitlines()[0] == "## 1. Purpose and Scope"
    assert result.splitlines()[2] == "## 2. Governance Model"
    assert count_headings(result) == 2


LEGAL_PAGES = [
    "Code civil du Québec\nCHAPITRE QUATORZIÈME\nDES AUTRES CONTRATS\n2300. Texte avant.\nPage 1",
    (
        "Code civil du Québec\nCHAPITRE QUINZIÈME\nDES ASSURANCES\nSECTION I\n"
        "DISPOSITIONS GÉNÉRALES\n"
        "2389. Le contrat d'assurance est celui par lequel\n"
        "l'assureur s'oblige à verser une prestation.\n"
        "1991, c. 64, a. 2389;\n1999, c. 40, a. 45.\nPage 2"
        "Code civil du Québec\n2390. L'assurance est terrestre ou maritime.\n"
        "§ 1. — Assurance de dommages\n2391. Le preneur déclare les risques.\n"
        "2002, c. 19, a. 15.\nPage 3"
        "Code civil du Québec\nCHAPITRE SEIZIÈME\nDU PRÊT\n2395. Autre chapitre.\nPage 4"
    ),
]


def test_legal_pages_to_markdown_extracts_one_chapter() -> None:
    markdown, stats = legal_pages_to_markdown(LEGAL_PAGES, first_article="2389")

    assert "## CHAPITRE QUINZIÈME - DES ASSURANCES" in markdown
    assert "### SECTION I - DISPOSITIONS GÉNÉRALES" in markdown
    assert "#### § 1. — Assurance de dommages" in markdown
    assert (
        "###### Article 2389\n\nLe contrat d'assurance est celui par lequel l'assureur s'oblige"
        in markdown
    )
    assert "1991, c. 64" not in markdown
    assert "Autre chapitre" not in markdown
    assert "Texte avant" not in markdown
    assert stats["first_article"] == "2389"
    assert stats["last_article"] == "2391"
    assert stats["articles"] == 3


def test_legal_pages_missing_article_raises() -> None:
    with pytest.raises(ValueError, match="2389"):
        legal_pages_to_markdown(["nothing here"], first_article="2389")


HTML = """
<html><body><header>Site menu</header>
<main>
  <h1>Guideline E-21</h1>
  <nav><ul><li>Skip me</li></ul></nav>
  <h2>Table of contents</h2><ul><li>A. Overview</li><li>B. Scope</li></ul>
  <h2>A. Overview</h2>
  <p>Operational risk is <strong>important</strong>, see <a href="/x">the guide</a>.</p>
  <div><div><h3>A1. Purpose</h3><p>Purpose text.</p></div></div>
  <ul><li>First item<ul><li>Nested item</li></ul></li><li>Second item</li></ul>
  <table><caption>Limits</caption><tr><th>Item</th><th>Value</th></tr>
  <tr><td>Deductible</td><td>2,500</td></tr></table>
  <dl><dt>FRFI</dt><dd>Federally regulated financial institution</dd></dl>
  <script>var x = 1;</script>
</main><footer>Footer</footer></body></html>
"""


def test_html_to_markdown_keeps_structure_and_drops_noise() -> None:
    markdown = html_to_markdown(HTML)

    assert markdown.startswith("# Guideline E-21")
    assert "## A. Overview" in markdown
    assert "### A1. Purpose" in markdown
    assert "Operational risk is important, see the guide." in markdown
    assert "- First item\n  - Nested item\n- Second item" in markdown
    assert "| Deductible | 2,500 |" in markdown
    assert "**FRFI**: Federally regulated financial institution" in markdown
    for noise in (
        "Site menu",
        "Footer",
        "Skip me",
        "var x",
        "Table of contents",
        "B. Scope",
    ):
        assert noise not in markdown


def test_article_number_at_top_of_page_is_never_removed() -> None:
    pages = []
    for n in range(12):
        number = 2400 + n
        body = "\n".join(f"Texte ligne {k} de la page {n}" for k in range(8))
        pages.append(
            f"{number}.\n{body}\nMention de pied de page\nCCQ-1991 / {n} sur 12"
        )
    cleaned = strip_page_boilerplate(pages)
    assert all(f"{2400 + n}." in cleaned[n] for n in range(12))
    assert all("CCQ-1991" not in page for page in cleaned)


def test_header_in_the_middle_of_a_page_is_removed_on_long_documents() -> None:
    pages = [
        f"Texte un\nCODE CIVIL\nTexte deux\nTexte trois\nTexte quatre\nTexte cinq\nTexte six\nFin {n}"
        for n in range(12)
    ]
    cleaned = strip_page_boilerplate(pages)
    assert all("CODE CIVIL" not in page for page in cleaned)


def test_roman_subheadings_between_articles_are_kept() -> None:
    pages = [
        (
            "CHAPITRE QUINZIÈME\nDES ASSURANCES\n2389. Premier article.\n"
            "1991, c. 64, a. 2389.\nI. — Du contenu de la police\n2480. Deuxième article.\n"
            "1991, c. 64, a. 2480."
        )
    ]
    markdown, stats = legal_pages_to_markdown(pages, first_article="2389")
    assert "##### I. — Du contenu de la police" in markdown
    assert stats["articles"] == 2
