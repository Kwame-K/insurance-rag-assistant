from insurance_rag_assistant.config import MIN_RETRIEVAL_SCORE
from insurance_rag_assistant.models.retrieval import (
    SearchQuery,
    SearchResult,
)
from insurance_rag_assistant.retrieval.embedder import (
    MultilingualE5Embedder,
)
from insurance_rag_assistant.retrieval.vector_store import (
    LocalQdrantVectorStore,
)


class SemanticSearchService:
    """Coordinate query embedding, vector retrieval, and relevance gating."""

    def __init__(
        self,
        embedder: MultilingualE5Embedder | None = None,
        vector_store: LocalQdrantVectorStore | None = None,
        min_retrieval_score: float = MIN_RETRIEVAL_SCORE,
    ) -> None:
        self.embedder = embedder or MultilingualE5Embedder()
        self.vector_store = vector_store or LocalQdrantVectorStore()
        self.min_retrieval_score = min_retrieval_score

    def search(self, search_query: SearchQuery) -> SearchResult:
        """Embed a user question and retrieve grounded source passages."""
        query_vector = self.embedder.embed_query(search_query.query)

        results = self.vector_store.search(
            query_vector=query_vector,
            filters=search_query.filters,
            top_k=search_query.top_k,
            min_score=self.min_retrieval_score,
        )

        return SearchResult(
            query=search_query,
            results=results,
            retrieval_sufficient=bool(results),
        )

    def close(self) -> None:
        """Close underlying local vector-store resources."""
        self.vector_store.close()
