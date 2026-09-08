from insurance_rag_assistant.generation.answer_generator import (
    GroundedAnswerGenerator,
    normalize_citation_text,
)
from insurance_rag_assistant.llm.fake import FakeStructuredLLMClient
from insurance_rag_assistant.models.retrieval import (
    RetrievedChunk,
    SearchQuery,
    SearchResult,
)


def test_answer_generator_rejects_unknown_citation_chunk() -> None:
    retrieved_chunk = RetrievedChunk(
        chunk_id="commercial_property_policy_v1:flood-exclusion:003",
        document_id="commercial_property_policy_v1",
        document_name="Commercial Property Policy",
        section_title="Flood Exclusion",
        section_path=["Commercial Property Policy", "Flood Exclusion"],
        page_start=None,
        page_end=None,
        text="This policy does not cover loss or damage caused directly or indirectly by flood.",
        score=0.82,
        rank=1,
        language="en",
        coverage="commercial_property",
        jurisdiction="quebec",
        version="1.0",
    )

    search_result = SearchResult(
        query=SearchQuery(query="What exclusions apply to flood damage?"),
        results=[retrieved_chunk],
        retrieval_sufficient=True,
    )

    fake_llm = FakeStructuredLLMClient(
        response={
            "answer": "Flood damage is excluded.",
            "answer_language": "en",
            "grounded": True,
            "confidence": "high",
            "citations": [
                {
                    "chunk_id": "invented_chunk_id",
                    "document_id": "commercial_property_policy_v1",
                    "document_name": "Commercial Property Policy",
                    "section_title": "Flood Exclusion",
                    "page_start": None,
                    "page_end": None,
                    "quote": "This policy does not cover loss or damage caused directly or indirectly by flood.",
                    "retrieval_score": 0.82,
                }
            ],
            "insufficient_context_reason": None,
        }
    )

    generator = GroundedAnswerGenerator(llm_client=fake_llm)

    try:
        generator.generate(
            question="What exclusions apply to flood damage?",
            search_result=search_result,
        )
    except ValueError as error:
        assert "non-retrieved chunk" in str(error)
    else:
        raise AssertionError("Expected an invalid citation to be rejected.")



def test_normalize_citation_text_handles_unicode_variants() -> None:
    source = "Five-year cyber incident and insurance claims history."
    generated = "Five‑year cyber incident and insurance claims history."

    assert (
        normalize_citation_text(generated)
        == normalize_citation_text(source)
    )
