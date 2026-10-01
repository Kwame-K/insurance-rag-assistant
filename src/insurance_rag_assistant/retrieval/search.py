from insurance_rag_assistant.config import ABSTENTION_TOP_SCORE, MIN_RETRIEVAL_SCORE
from insurance_rag_assistant.models.retrieval import (
    SearchQuery,
    SearchResult,
)
from insurance_rag_assistant.retrieval.embedder import (
    MultilingualE5Embedder,
)
from insurance_rag_assistant.retrieval.factory import create_vector_store
from insurance_rag_assistant.retrieval.protocols import VectorStore


class SemanticSearchService:
    """Coordinate query embedding, vector retrieval, and relevance gating."""

    def __init__(
        self,
        embedder: MultilingualE5Embedder | None = None,
        vector_store: VectorStore | None = None,
        min_retrieval_score: float = MIN_RETRIEVAL_SCORE,
        abstention_score: float = ABSTENTION_TOP_SCORE,
    ) -> None:
        self.embedder = embedder or MultilingualE5Embedder()
        self.vector_store = vector_store or create_vector_store()
        self.min_retrieval_score = min_retrieval_score
        self.abstention_score = abstention_score

    def search(self, search_query: SearchQuery) -> SearchResult:
        """Embed a user question and retrieve grounded source passages."""
        query_vector = self.embedder.embed_query(search_query.query)

        results = self.vector_store.search(
            query_vector=query_vector,
            filters=search_query.filters,
            top_k=search_query.top_k,
            min_score=self.min_retrieval_score,
            query_text=search_query.query,
        )
        top_score = max((r.score for r in results), default=0.0)
        return SearchResult(
            query=search_query,
            results=results,
            retrieval_sufficient=top_score >= self.abstention_score,
        )

    def close(self) -> None:
        """Close underlying local vector-store resources."""
        self.vector_store.close()
