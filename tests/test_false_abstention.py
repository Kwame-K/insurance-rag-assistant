from insurance_rag_assistant.evaluation.evaluator import (
    EvaluationCase,
    evaluate_retrieval,
)
from insurance_rag_assistant.models.retrieval import (
    RetrievedChunk,
    SearchQuery,
    SearchResult,
)


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="doc:1",
        document_id="doc",
        document_name="Doc",
        section_title="Section",
        section_path=["Section"],
        text="text",
        score=0.9,
        rank=1,
        language="en",
        coverage="multi_line",
        jurisdiction="canada",
        version="1",
    )


class _Search:
    def __init__(self, sufficient: bool) -> None:
        self.sufficient = sufficient

    def search(self, query: SearchQuery) -> SearchResult:
        return SearchResult(
            query=query, results=[_chunk()], retrieval_sufficient=self.sufficient
        )


def _case(case_id: str, expect: bool) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question="test question",
        expected_document_ids=["doc"] if expect else [],
        expect_retrieval=expect,
        expected_grounded=expect,
    )


def test_false_abstention_metrics_count_answerable_rejections() -> None:
    report = evaluate_retrieval(
        _Search(False), [_case("answerable", True), _case("negative", False)], 5
    )
    assert report["answerable_retrieval_rate"] == 0.0
    assert report["false_abstention_rate"] == 1.0
    assert report["false_abstention_case_ids"] == ["answerable"]
    assert report["case_results"][0]["false_abstention"] is True
    assert report["correct_abstention_rate"] == 1.0


def test_false_abstention_metrics_accept_answerable_case() -> None:
    report = evaluate_retrieval(_Search(True), [_case("answerable", True)], 5)
    assert report["answerable_retrieval_rate"] == 1.0
    assert report["false_abstention_rate"] == 0.0
    assert report["false_abstention_case_ids"] == []
