from typing import Protocol

from insurance_rag_assistant.models.documents import DocumentChunk
from insurance_rag_assistant.models.retrieval import RetrievedChunk, SearchFilters


class VectorStore(Protocol):
    """Contract shared by the Qdrant and pgvector backends."""

    def recreate_collection(self) -> None: ...

    def ensure_collection(self) -> None: ...

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None: ...

    def search(
        self,
        query_vector: list[float],
        filters: SearchFilters,
        top_k: int,
        min_score: float,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]: ...

    def close(self) -> None: ...
