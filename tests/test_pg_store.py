import hashlib
import os
from datetime import date

import pytest

from insurance_rag_assistant.models.documents import DocumentChunk
from insurance_rag_assistant.models.retrieval import SearchFilters
from insurance_rag_assistant.retrieval.pg_store import PgVectorStore, build_or_tsquery

DATABASE_URL_TEST = os.environ.get("DATABASE_URL_TEST")

pytestmark = pytest.mark.skipif(
    DATABASE_URL_TEST is None,
    reason="DATABASE_URL_TEST is not set; pgvector integration tests skipped.",
)

DIM = 768


def _vec(hot: int) -> list[float]:
    v = [0.0] * DIM
    v[hot] = 1.0
    return v


def _chunk(
    chunk_id: str, document_id: str, text: str, language: str = "en"
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=f"Doc {document_id}",
        document_type="policy_wording",
        coverage="commercial_property",
        jurisdiction="CA-QC",
        language=language,  # type: ignore[arg-type]
        version="1.0",
        effective_date=date(2026, 1, 1),
        section_title="Coverage",
        section_path=["Policy", "Coverage"],
        chunk_index=0,
        text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


@pytest.fixture
def store():
    from insurance_rag_assistant.db.connection import create_pool
    from insurance_rag_assistant.db.migrate import apply_migrations

    apply_migrations(DATABASE_URL_TEST)
    s = PgVectorStore(pool=create_pool(DATABASE_URL_TEST))
    s.recreate_collection()
    yield s
    s.recreate_collection()
    s.close()


def test_upsert_then_search_returns_nearest_first(store) -> None:
    chunks = [
        _chunk("a:cov:0", "a", "water damage"),
        _chunk("b:cov:0", "b", "cyber attack"),
    ]
    store.upsert_chunks(chunks, [_vec(0), _vec(1)])

    results = store.search(_vec(0), SearchFilters(), top_k=2, min_score=-1.0)

    assert results[0].chunk_id == "a:cov:0"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1.0, abs=1e-4)


def test_min_score_filters_out_distant_chunks(store) -> None:
    store.upsert_chunks(
        [_chunk("a:cov:0", "a", "x"), _chunk("b:cov:0", "b", "y")], [_vec(0), _vec(1)]
    )

    results = store.search(_vec(0), SearchFilters(), top_k=5, min_score=0.5)

    assert [r.chunk_id for r in results] == ["a:cov:0"]


def test_language_filter(store) -> None:
    store.upsert_chunks(
        [_chunk("a:cov:0", "a", "x", "en"), _chunk("b:cov:0", "b", "y", "fr")],
        [_vec(0), _vec(0)],
    )

    results = store.search(
        _vec(0), SearchFilters(language="fr"), top_k=5, min_score=0.0
    )

    assert [r.document_id for r in results] == ["b"]


def test_archived_documents_are_excluded(store) -> None:
    store.upsert_chunks([_chunk("a:cov:0", "a", "x")], [_vec(0)])
    with store.pool.connection() as conn:
        conn.execute("UPDATE documents SET status = 'archived' WHERE document_id = 'a'")

    assert store.search(_vec(0), SearchFilters(), top_k=5, min_score=0.0) == []


def test_upsert_is_idempotent(store) -> None:
    chunk = _chunk("a:cov:0", "a", "x")
    store.upsert_chunks([chunk], [_vec(0)])
    store.upsert_chunks([chunk], [_vec(0)])

    with store.pool.connection() as conn:
        row = conn.execute("SELECT count(*) FROM chunks").fetchone()
    assert row[0] == 1


def test_dimension_mismatch_raises(store) -> None:
    with pytest.raises(ValueError, match="Embedding dimension"):
        store.upsert_chunks([_chunk("a:cov:0", "a", "x")], [[0.1, 0.2]])


def test_build_or_tsquery_keeps_words_only() -> None:
    query = build_or_tsquery("Is a burst pipe covered? Quelle franchise s'applique")
    assert query == "is | burst | pipe | covered | quelle | franchise | applique"


def test_build_or_tsquery_empty_for_punctuation() -> None:
    assert build_or_tsquery("?! -- ...") == ""


def test_hybrid_promotes_lexical_match(store) -> None:
    chunks = [
        _chunk("a:cov:0", "a", "fire damage to the building"),
        _chunk("b:cov:0", "b", "plumbing leak in the building"),
    ]
    store.upsert_chunks(chunks, [_vec(0), _vec(0)])

    results = store.search(
        _vec(0), SearchFilters(), top_k=2, min_score=-1.0, query_text="plumbing"
    )

    assert results[0].chunk_id == "b:cov:0"
    assert [r.rank for r in results] == [1, 2]


def test_hybrid_falls_back_to_vector_without_text(store) -> None:
    chunks = [
        _chunk("a:cov:0", "a", "water damage"),
        _chunk("b:cov:0", "b", "cyber attack"),
    ]
    store.upsert_chunks(chunks, [_vec(0), _vec(1)])

    results = store.search(_vec(0), SearchFilters(), top_k=2, min_score=-1.0)

    assert results[0].chunk_id == "a:cov:0"


def test_hybrid_respects_filters(store) -> None:
    chunks = [
        _chunk("a:cov:0", "a", "plumbing leak"),
        _chunk("b:cov:0", "b", "plumbing leak"),
    ]
    store.upsert_chunks(chunks, [_vec(0), _vec(0)])
    with store.pool.connection() as conn:
        conn.execute("UPDATE documents SET status = 'archived' WHERE document_id = 'b'")

    results = store.search(
        _vec(0), SearchFilters(), top_k=5, min_score=-1.0, query_text="plumbing"
    )

    assert [r.chunk_id for r in results] == ["a:cov:0"]


def test_multi_line_documents_pass_coverage_filter(store) -> None:
    chunks = [_chunk("a:cov:0", "a", "flood"), _chunk("b:cov:0", "b", "flood")]
    store.upsert_chunks(chunks, [_vec(0), _vec(0)])
    with store.pool.connection() as conn:
        conn.execute("UPDATE documents SET coverage = 'other' WHERE document_id = 'a'")
        conn.execute(
            "UPDATE documents SET coverage = 'multi_line' WHERE document_id = 'b'"
        )

    results = store.search(
        _vec(0),
        SearchFilters(coverage="commercial_property"),
        top_k=5,
        min_score=-1.0,
    )

    assert [r.chunk_id for r in results] == ["b:cov:0"]
