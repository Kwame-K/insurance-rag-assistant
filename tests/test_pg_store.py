import hashlib
import os
from datetime import date

import pytest

from insurance_rag_assistant.models.documents import DocumentChunk
from insurance_rag_assistant.models.retrieval import SearchFilters

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


def _chunk(chunk_id: str, document_id: str, text: str, language: str = "en") -> DocumentChunk:
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
    from insurance_rag_assistant.retrieval.pg_store import PgVectorStore

    apply_migrations(DATABASE_URL_TEST)
    s = PgVectorStore(pool=create_pool(DATABASE_URL_TEST))
    s.recreate_collection()
    yield s
    s.recreate_collection()
    s.close()


def test_upsert_then_search_returns_nearest_first(store) -> None:
    chunks = [_chunk("a:cov:0", "a", "water damage"), _chunk("b:cov:0", "b", "cyber attack")]
    store.upsert_chunks(chunks, [_vec(0), _vec(1)])

    results = store.search(_vec(0), SearchFilters(), top_k=2, min_score=-1.0)

    assert results[0].chunk_id == "a:cov:0"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1.0, abs=1e-4)


def test_min_score_filters_out_distant_chunks(store) -> None:
    store.upsert_chunks([_chunk("a:cov:0", "a", "x"), _chunk("b:cov:0", "b", "y")], [_vec(0), _vec(1)])

    results = store.search(_vec(0), SearchFilters(), top_k=5, min_score=0.5)

    assert [r.chunk_id for r in results] == ["a:cov:0"]


def test_language_filter(store) -> None:
    store.upsert_chunks(
        [_chunk("a:cov:0", "a", "x", "en"), _chunk("b:cov:0", "b", "y", "fr")],
        [_vec(0), _vec(0)],
    )

    results = store.search(_vec(0), SearchFilters(language="fr"), top_k=5, min_score=0.0)

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
