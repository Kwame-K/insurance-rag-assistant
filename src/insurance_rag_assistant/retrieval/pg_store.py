from typing import Any

from pgvector import Vector
from psycopg import Connection
from psycopg.rows import TupleRow, dict_row
from psycopg_pool import ConnectionPool

from insurance_rag_assistant.config import EMBEDDING_DIMENSIONS
from insurance_rag_assistant.db.connection import create_pool
from insurance_rag_assistant.models.documents import DocumentChunk
from insurance_rag_assistant.models.retrieval import RetrievedChunk, SearchFilters

_UPSERT_DOCUMENT = """
INSERT INTO documents (
    document_id, title, document_type, coverage, jurisdiction,
    language, version, effective_date
) VALUES (
    %(document_id)s, %(title)s, %(document_type)s, %(coverage)s, %(jurisdiction)s,
    %(language)s, %(version)s, %(effective_date)s
)
ON CONFLICT (document_id) DO UPDATE SET
    title = EXCLUDED.title,
    document_type = EXCLUDED.document_type,
    coverage = EXCLUDED.coverage,
    jurisdiction = EXCLUDED.jurisdiction,
    language = EXCLUDED.language,
    version = EXCLUDED.version,
    effective_date = EXCLUDED.effective_date
"""

_UPSERT_CHUNK = """
INSERT INTO chunks (
    chunk_id, document_id, section_title, section_path, chunk_index,
    page_start, page_end, text, content_hash, embedding
) VALUES (
    %(chunk_id)s, %(document_id)s, %(section_title)s, %(section_path)s, %(chunk_index)s,
    %(page_start)s, %(page_end)s, %(text)s, %(content_hash)s, %(embedding)s
)
ON CONFLICT (chunk_id) DO UPDATE SET
    document_id = EXCLUDED.document_id,
    section_title = EXCLUDED.section_title,
    section_path = EXCLUDED.section_path,
    chunk_index = EXCLUDED.chunk_index,
    page_start = EXCLUDED.page_start,
    page_end = EXCLUDED.page_end,
    text = EXCLUDED.text,
    content_hash = EXCLUDED.content_hash,
    embedding = EXCLUDED.embedding
WHERE chunks.content_hash IS DISTINCT FROM EXCLUDED.content_hash
"""

_SEARCH = """
SELECT
    c.chunk_id, c.document_id, d.title AS document_name, c.section_title,
    c.section_path, c.page_start, c.page_end, c.text,
    d.language, d.coverage, d.jurisdiction, d.version,
    1 - (c.embedding <=> %(qvec)s) AS score
FROM chunks c
JOIN documents d ON d.document_id = c.document_id
WHERE d.status = 'active'
  AND (%(coverage)s::text IS NULL OR d.coverage = %(coverage)s)
  AND (%(jurisdiction)s::text IS NULL OR d.jurisdiction = %(jurisdiction)s)
  AND (%(language)s::text IS NULL OR d.language = %(language)s)
  AND (%(document_type)s::text IS NULL OR d.document_type = %(document_type)s)
  AND (%(version)s::text IS NULL OR d.version = %(version)s)
ORDER BY c.embedding <=> %(qvec)s
LIMIT %(limit)s
"""


class PgVectorStore:
    """Postgres + pgvector storage with the same contract as the Qdrant store."""

    def __init__(
        self,
        pool: ConnectionPool[Connection[TupleRow]] | None = None,
        vector_size: int = EMBEDDING_DIMENSIONS,
        ef_search: int = 100,
    ) -> None:
        self.pool = pool or create_pool()
        self.vector_size = vector_size
        self.ef_search = ef_search

    def recreate_collection(self) -> None:
        """Delete all documents and chunks (schema is managed by migrations)."""
        with self.pool.connection() as conn:
            conn.execute("TRUNCATE chunks, documents CASCADE")

    def ensure_collection(self) -> None:
        """Fail fast if migrations have not been applied."""
        with self.pool.connection() as conn:
            row = conn.execute("SELECT to_regclass('public.chunks')").fetchone()
        if row is None or row[0] is None:
            message = "Table 'chunks' is missing. Run: python -m insurance_rag_assistant.db.migrate"
            raise RuntimeError(message)

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            message = (
                "Number of chunks and embeddings must match: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings."
            )
            raise ValueError(message)

        documents: dict[str, dict[str, Any]] = {}
        chunk_rows: list[dict[str, Any]] = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if len(embedding) != self.vector_size:
                message = (
                    f"Embedding dimension must be {self.vector_size}; "
                    f"received {len(embedding)} for {chunk.chunk_id}."
                )
                raise ValueError(message)

            documents[chunk.document_id] = {
                "document_id": chunk.document_id,
                "title": chunk.document_name,
                "document_type": chunk.document_type,
                "coverage": chunk.coverage,
                "jurisdiction": chunk.jurisdiction,
                "language": chunk.language,
                "version": chunk.version,
                "effective_date": chunk.effective_date,
            }
            chunk_rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "section_title": chunk.section_title,
                    "section_path": chunk.section_path,
                    "chunk_index": chunk.chunk_index,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "text": chunk.text,
                    "content_hash": chunk.content_hash,
                    "embedding": Vector(embedding),
                }
            )

        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(_UPSERT_DOCUMENT, list(documents.values()))
            cur.executemany(_UPSERT_CHUNK, chunk_rows)

    def search(
        self,
        query_vector: list[float],
        filters: SearchFilters,
        top_k: int,
        min_score: float,
    ) -> list[RetrievedChunk]:
        if len(query_vector) != self.vector_size:
            message = (
                f"Query vector dimension must be {self.vector_size}; "
                f"received {len(query_vector)}."
            )
            raise ValueError(message)

        params = {
            "qvec": Vector(query_vector),
            "limit": top_k,
            **filters.model_dump(),
        }

        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
            conn.execute(
                    "SELECT set_config('hnsw.ef_search', %s, true)",
                    (str(int(self.ef_search)),),
            )
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(_SEARCH, params)
                fetched = cur.fetchall()

        ranked = sorted(fetched, key=lambda r: r["score"], reverse=True)
        kept = [r for r in ranked if r["score"] >= min_score]

        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                document_name=r["document_name"],
                section_title=r["section_title"],
                section_path=r["section_path"],
                page_start=r["page_start"],
                page_end=r["page_end"],
                text=r["text"],
                score=float(r["score"]),
                rank=rank,
                language=r["language"],
                coverage=r["coverage"],
                jurisdiction=r["jurisdiction"],
                version=r["version"],
            )
            for rank, r in enumerate(kept, start=1)
        ]

    def close(self
    ) -> None:
        self.pool.close()