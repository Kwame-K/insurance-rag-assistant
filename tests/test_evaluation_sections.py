from insurance_rag_assistant.evaluation.evaluator import (
    EvaluationCase,
    evaluate_retrieval,
)
from insurance_rag_assistant.models.retrieval import (
    RetrievedChunk,
    SearchQuery,
    SearchResult,
)


def _chunk(rank: int, section: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"doc:{section}:{rank}",
        document_id="doc",
        document_name="Doc",
        section_title=section,
        section_path=[section],
        text="text",
        score=0.8,
        rank=rank,
        language="fr",
        coverage="multi_line",
        jurisdiction="quebec",
        version="1",
    )


class _FakeSearch:
    def __init__(self, sections: list[str]) -> None:
        self.sections = sections

    def search(self, search_query: SearchQuery) -> SearchResult:
        return SearchResult(
            query=search_query,
            results=[_chunk(i, s) for i, s in enumerate(self.sections, start=1)],
            retrieval_sufficient=True,
        )


def _case(titles: list[str]) -> EvaluationCase:
    return EvaluationCase(
        case_id="c1",
        question="question de test",
        expected_document_ids=["doc"],
        expect_retrieval=True,
        expected_grounded=True,
        expected_section_titles=titles,
    )


def test_section_metrics_when_expected_section_is_third() -> None:
    report = evaluate_retrieval(
        _FakeSearch(["Article 2394", "Article 2436", "Article 2389"]),
        [_case(["Article 2389"])],
        top_k=5,
    )
    assert report["section_hit_at_1"] == 0.0
    assert report["section_hit_at_k"] == 1.0
    assert abs(report["section_mrr"] - 1 / 3) < 1e-9
    assert report["case_results"][0]["first_section_rank"] == 3


def test_section_metrics_accept_any_expected_section() -> None:
    report = evaluate_retrieval(
        _FakeSearch(["Article 2436", "Article 2394"]),
        [_case(["Article 2436", "Article 2473"])],
        top_k=5,
    )
    assert report["section_hit_at_1"] == 1.0


def test_cases_without_sections_are_ignored() -> None:
    report = evaluate_retrieval(_FakeSearch(["Article 1"]), [_case([])], top_k=5)
    assert report["section_level_cases"] == 0
    assert report["section_mrr"] is None
