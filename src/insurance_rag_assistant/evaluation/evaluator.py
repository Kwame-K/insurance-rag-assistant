import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Protocol

from groq import APIError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from insurance_rag_assistant.generation.answer_generator import (
    GroundedAnswerGenerator,
)
from insurance_rag_assistant.models.retrieval import (
    SearchFilters,
    SearchQuery,
    SearchResult,
)


class EvaluationCase(BaseModel):
    """One retrieval and generation evaluation case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=3)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    expected_document_ids: list[str]
    expected_citation_document_ids: list[str] = Field(default_factory=list)
    expect_retrieval: bool
    expected_grounded: bool
    must_include_terms: list[str] = Field(default_factory=list)
    must_not_include_terms: list[str] = Field(default_factory=list)



class SearchService(Protocol):
    """Minimal search interface needed by the evaluator."""

    def search(self, search_query: SearchQuery) -> SearchResult:
        """Return ranked passages for a query."""

def normalize_evaluation_text(text: str) -> str:
    """Normalize text for resilient deterministic term matching."""
    decomposed = unicodedata.normalize("NFKD", text)

    without_diacritics = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )

    normalized = without_diacritics.casefold()
    normalized = re.sub(r"[\W_]+", " ", normalized)

    return " ".join(normalized.split())



def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load and validate evaluation cases from a JSON file."""
    raw_cases = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw_cases, list):
        raise TypeError("Evaluation cases must be a JSON list.")

    return [EvaluationCase.model_validate(raw_case) for raw_case in raw_cases]


def evaluate_retrieval(
    search_service: SearchService,
    cases: list[EvaluationCase],
    top_k: int,
) -> dict[str, Any]:
    """Evaluate document-level retrieval and abstention behaviour."""
    case_results: list[dict[str, Any]] = []

    recall_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    correct_abstentions = 0
    total_unanswerable_cases = 0

    for case in cases:
        search_result = search_service.search(
            SearchQuery(
                query=case.question,
                top_k=top_k,
                filters=case.filters,
            )
        )

        retrieved_document_ids = [
            result.document_id for result in search_result.results
        ]

        retrieved_passages = [
            {
                "rank": result.rank,
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "section_title": result.section_title,
                "score": result.score,
            }
            for result in search_result.results
        ]

        unique_retrieved_document_ids = set(retrieved_document_ids)
        expected_document_ids = set(case.expected_document_ids)

        if case.expect_retrieval:
            relevant_retrieved_ids = (
                expected_document_ids & unique_retrieved_document_ids
            )

            recall_at_k = len(relevant_retrieved_ids) / len(expected_document_ids)

            first_relevant_rank = next(
                (
                    rank
                    for rank, document_id in enumerate(
                        retrieved_document_ids,
                        start=1,
                    )
                    if document_id in expected_document_ids
                ),
                None,
            )

            reciprocal_rank = (
                1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
            )

            recall_scores.append(recall_at_k)
            reciprocal_ranks.append(reciprocal_rank)

            case_results.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "expected_document_ids": sorted(expected_document_ids),
                    "retrieved_document_ids": retrieved_document_ids,
                    "retrieved_passages": retrieved_passages,
                    "retrieval_sufficient": search_result.retrieval_sufficient,
                    "recall_at_k": recall_at_k,
                    "first_relevant_rank": first_relevant_rank,
                    "reciprocal_rank": reciprocal_rank,
                    "passed": recall_at_k > 0.0,
                }
            )

        else:
            total_unanswerable_cases += 1

            correct_abstention = not search_result.retrieval_sufficient

            if correct_abstention:
                correct_abstentions += 1

            case_results.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "expected_document_ids": [],
                    "retrieved_document_ids": retrieved_document_ids,
                    "retrieved_passages": retrieved_passages,
                    "retrieval_sufficient": search_result.retrieval_sufficient,
                    "correct_abstention": correct_abstention,
                    "passed": correct_abstention,
                }
            )

    answerable_case_count = len(recall_scores)

    return {
        "evaluation_type": "document_level_retrieval",
        "top_k": top_k,
        "total_cases": len(cases),
        "answerable_cases": answerable_case_count,
        "unanswerable_cases": total_unanswerable_cases,
        "macro_recall_at_k": (
            sum(recall_scores) / answerable_case_count
            if answerable_case_count
            else None
        ),
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / answerable_case_count
            if answerable_case_count
            else None
        ),
        "correct_abstention_rate": (
            correct_abstentions / total_unanswerable_cases
            if total_unanswerable_cases
            else None
        ),
        "case_results": case_results,
    }


def write_evaluation_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a readable JSON evaluation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def evaluate_generation(
    search_service: SearchService,
    answer_generator: GroundedAnswerGenerator,
    cases: list[EvaluationCase],
    top_k: int,
) -> dict[str, Any]:
    """Evaluate grounding, abstention, citations, and answer constraints."""
    case_results: list[dict[str, Any]] = []

    expected_grounded_cases = 0
    correctly_grounded_cases = 0

    expected_abstention_cases = 0
    correctly_abstained_cases = 0

    total_citations = 0
    expected_document_citations = 0

    answer_constraint_results: list[bool] = []
    generation_errors = 0

    for case in cases:
        if case.expected_grounded:
            expected_grounded_cases += 1
        else:
            expected_abstention_cases += 1

        has_answer_constraints = bool(
            case.must_include_terms
            or case.must_not_include_terms
        )

        search_result = search_service.search(
            SearchQuery(
                query=case.question,
                top_k=top_k,
                filters=case.filters,
            )
        )

        try:
            response = answer_generator.generate(
                question=case.question,
                search_result=search_result,
            )
        except (
            APIError,
            RuntimeError,
            TypeError,
            ValidationError,
            ValueError,
        ) as error:
            generation_errors += 1

            if has_answer_constraints:
                answer_constraint_results.append(False)

            case_results.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "expected_grounded": case.expected_grounded,
                    "actual_grounded": None,
                    "confidence": None,
                    "answer": None,
                    "insufficient_context_reason": None,
                    "citation_document_ids": [],
                    "expected_citation_document_ids": sorted(
                        case.expected_citation_document_ids
                    ),
                    "citation_count": 0,
                    "must_include_terms": case.must_include_terms,
                    "must_not_include_terms": (
                        case.must_not_include_terms
                    ),
                    "missing_required_terms": [],
                    "present_forbidden_terms": [],
                    "answer_constraints_passed": False,
                    "grounding_matches_expectation": False,
                    "passed": False,
                    "generation_error": str(error),
                }
            )
            continue

        citation_document_ids = [
            citation.document_id
            for citation in response.citations
        ]

        expected_citation_ids = set(
            case.expected_citation_document_ids
        )

        correct_citation_count = sum(
            document_id in expected_citation_ids
            for document_id in citation_document_ids
        )

        total_citations += len(citation_document_ids)
        expected_document_citations += correct_citation_count

        grounding_matches_expectation = (
            response.grounded == case.expected_grounded
        )

        if case.expected_grounded and response.grounded:
            correctly_grounded_cases += 1

        if not case.expected_grounded and not response.grounded:
            correctly_abstained_cases += 1

        normalized_answer = normalize_evaluation_text(
            response.answer
        )


        missing_required_terms = [
            term
            for term in case.must_include_terms
            if normalize_evaluation_text(term) not in normalized_answer
        ]


        present_forbidden_terms = [
            term
            for term in case.must_not_include_terms
            if normalize_evaluation_text(term) in normalized_answer
        ]


        answer_constraints_passed = (
            not missing_required_terms
            and not present_forbidden_terms
        )

        if has_answer_constraints:
            answer_constraint_results.append(
                answer_constraints_passed
            )

        case_results.append(
            {
                "case_id": case.case_id,
                "question": case.question,
                "expected_grounded": case.expected_grounded,
                "actual_grounded": response.grounded,
                "confidence": response.confidence,
                "answer": response.answer,
                "insufficient_context_reason": (
                    response.insufficient_context_reason
                ),
                "citation_document_ids": citation_document_ids,
                "expected_citation_document_ids": sorted(
                    expected_citation_ids
                ),
                "citation_count": len(citation_document_ids),
                "must_include_terms": case.must_include_terms,
                "must_not_include_terms": (
                    case.must_not_include_terms
                ),
                "missing_required_terms": missing_required_terms,
                "present_forbidden_terms": present_forbidden_terms,
                "answer_constraints_passed": (
                    answer_constraints_passed
                ),
                "grounding_matches_expectation": (
                    grounding_matches_expectation
                ),
                "passed": (
                    grounding_matches_expectation
                    and answer_constraints_passed
                ),
            }
        )

    total_cases = len(cases)

    return {
        "evaluation_type": "generation_grounding_citations_completeness",
        "top_k": top_k,
        "total_cases": total_cases,
        "generation_errors": generation_errors,
        "grounded_answer_rate": (
            correctly_grounded_cases / expected_grounded_cases
            if expected_grounded_cases
            else None
        ),
        "correct_grounded_abstention_rate": (
            correctly_abstained_cases / expected_abstention_cases
            if expected_abstention_cases
            else None
        ),
        "citation_document_precision": (
            expected_document_citations / total_citations
            if total_citations
            else None
        ),
        "answer_constraint_pass_rate": (
            sum(answer_constraint_results)
            / len(answer_constraint_results)
            if answer_constraint_results
            else None
        ),
        "overall_grounding_accuracy": (
            (
                correctly_grounded_cases
                + correctly_abstained_cases
            )
            / total_cases
            if total_cases
            else None
        ),
        "case_results": case_results,
    }
