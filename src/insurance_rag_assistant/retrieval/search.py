from insurance_rag_assistant.config import (
    ABSTENTION_TOP_SCORE,
    MIN_RETRIEVAL_SCORE,
    settings,
)
from insurance_rag_assistant.models.retrieval import (
    SearchQuery,
    SearchResult,
)
from insurance_rag_assistant.retrieval.embedder import (
    MultilingualE5Embedder,
)
from insurance_rag_assistant.retrieval.factory import create_vector_store
from insurance_rag_assistant.retrieval.protocols import Reranker, VectorStore


class SemanticSearchService:
    """Coordinate query embedding, vector retrieval, and relevance gating."""

    def __init__(
        self,
        embedder: MultilingualE5Embedder | None = None,
        vector_store: VectorStore | None = None,
        min_retrieval_score: float = MIN_RETRIEVAL_SCORE,
        abstention_score: float = ABSTENTION_TOP_SCORE,
        reranker: Reranker | None = None,
        rerank_candidates: int | None = None,
        rerank_abstention_score: float | None = None,
    ) -> None:
        self.embedder = embedder or MultilingualE5Embedder()
        self.vector_store = vector_store or create_vector_store()
        self.min_retrieval_score = min_retrieval_score
        self.abstention_score = abstention_score
        if reranker is None and settings.reranker:
            from insurance_rag_assistant.retrieval.reranker import (
                CrossEncoderReranker,
            )

            reranker = CrossEncoderReranker(settings.reranker_model)
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates or settings.reranker_candidates
        self.rerank_abstention_score = (
            rerank_abstention_score
            if rerank_abstention_score is not None
            else settings.rerank_abstention_score
        )

    def search(self, search_query: SearchQuery) -> SearchResult:
        """Embed a user question and retrieve grounded source passages."""
        query_vector = self.embedder.embed_query(search_query.query)

        fetch_k = (
            max(search_query.top_k, self.rerank_candidates)
            if self.reranker is not None
            else search_query.top_k
        )
        results = self.vector_store.search(
            query_vector=query_vector,
            filters=search_query.filters,
            top_k=fetch_k,
            min_score=self.min_retrieval_score,
            query_text=search_query.query,
        )
        if self.reranker is not None:
            results = self.reranker.rerank(
                search_query.query,
                results,
                search_query.top_k,
            )

        top_score = max((r.score for r in results), default=0.0)
        sufficient = top_score >= self.abstention_score
        if self.reranker is not None and self.rerank_abstention_score is not None:
            top_rerank = max(
                (r.rerank_score for r in results if r.rerank_score is not None),
                default=float("-inf"),
            )
            sufficient = top_rerank >= self.rerank_abstention_score

        return SearchResult(
            query=search_query,
            results=results,
            retrieval_sufficient=sufficient,
        )

    def close(self) -> None:
        """Close underlying local vector-store resources."""
        self.vector_store.close()
