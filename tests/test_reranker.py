from insurance_rag_assistant.models.retrieval import (
    RetrievedChunk,
    SearchFilters,
    SearchQuery,
)
from insurance_rag_assistant.retrieval.reranker import CrossEncoderReranker
from insurance_rag_assistant.retrieval.search import SemanticSearchService


def _chunk(rank: int, section: str, score: float = 0.86) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc:{section}",
        document_id="doc",
        document_name="Doc",
        section_title=section,
        section_path=[section],
        text=f"text of {section}",
        score=score,
        rank=rank,
        language="fr",
        coverage="multi_line",
        jurisdiction="quebec",
        version="1",
    )


class _FakeModel:
    def __init__(self, by_text: dict[str, float]) -> None:
        self.by_text = by_text

    def predict(self, sentences, **kwargs):
        return [self.by_text[text] for _, text in sentences]


class _FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class _FakeStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.requested_top_k: int | None = None

    def search(self, query_vector, filters, top_k, min_score, query_text=None):
        self.requested_top_k = top_k
        return self.chunks[:top_k]

    def close(self) -> None:
        return None


def _query(top_k: int = 2) -> SearchQuery:
    return SearchQuery(query="question de test", top_k=top_k, filters=SearchFilters())


def _chunks() -> list[RetrievedChunk]:
    return [_chunk(1, "A"), _chunk(2, "B"), _chunk(3, "C")]


def test_reranker_orders_by_cross_encoder_score_and_renumbers() -> None:
    model = _FakeModel({"text of A": 0.1, "text of B": 0.9, "text of C": 0.5})
    reranker = CrossEncoderReranker("fake", model=model)

    result = reranker.rerank("q", _chunks(), top_k=2)

    assert [c.section_title for c in result] == ["B", "C"]
    assert [c.rank for c in result] == [1, 2]
    assert result[0].rerank_score == 0.9
    assert result[0].score == 0.86


def test_search_service_fetches_candidates_then_returns_top_k() -> None:
    model = _FakeModel({"text of A": 0.1, "text of B": 0.9, "text of C": 0.5})
    store = _FakeStore(_chunks())
    service = SemanticSearchService(
        embedder=_FakeEmbedder(),
        vector_store=store,
        reranker=CrossEncoderReranker("fake", model=model),
        rerank_candidates=20,
    )

    result = service.search(_query(top_k=2))

    assert store.requested_top_k == 20
    assert [c.section_title for c in result.results] == ["B", "C"]


def test_rerank_abstention_uses_rerank_score_when_configured() -> None:
    model = _FakeModel({"text of A": 0.1, "text of B": 0.2, "text of C": 0.15})
    service = SemanticSearchService(
        embedder=_FakeEmbedder(),
        vector_store=_FakeStore(_chunks()),
        reranker=CrossEncoderReranker("fake", model=model),
        rerank_abstention_score=0.5,
    )

    result = service.search(_query())

    assert result.retrieval_sufficient is False


def test_without_reranker_behaviour_is_unchanged() -> None:
    store = _FakeStore(_chunks())
    service = SemanticSearchService(embedder=_FakeEmbedder(), vector_store=store)

    result = service.search(_query(top_k=2))

    assert store.requested_top_k == 2
    assert [c.section_title for c in result.results] == ["A", "B"]
    assert result.results[0].rerank_score is None
