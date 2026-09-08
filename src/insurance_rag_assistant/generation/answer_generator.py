import re
import unicodedata

from insurance_rag_assistant.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from insurance_rag_assistant.llm.base import StructuredLLMClient
from insurance_rag_assistant.models.response import (
    Citation,
    RAGResponse,
)
from insurance_rag_assistant.models.retrieval import RetrievedChunk, SearchResult

CITATION_TRANSLATIONS: dict[str, str | int | None] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2010": "-",
    "\u2011": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}

CITATION_TRANSLATION_TABLE = str.maketrans(
    CITATION_TRANSLATIONS
)


def normalize_citation_text(text: str) -> str:
    """Normalize harmless Unicode differences for verbatim quote matching."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(
        CITATION_TRANSLATION_TABLE
    )

    return re.sub(r"\s+", " ", normalized).strip()



class GroundedAnswerGenerator:
    """Generate and validate answers from retrieved insurance passages only."""

    def __init__(self, llm_client: StructuredLLMClient) -> None:
        self.llm_client = llm_client

    def generate(
        self,
        question: str,
        search_result: SearchResult,
    ) -> RAGResponse:
        """Generate a validated grounded answer or an explicit refusal."""
        if not search_result.retrieval_sufficient:
            return RAGResponse(
                answer=(
                    "I cannot answer this from the retrieved insurance documentation."
                ),
                answer_language="en",
                grounded=False,
                confidence="low",
                citations=[],
                insufficient_context_reason=(
                    "No retrieved passage met the minimum relevance threshold."
                ),
            )

        raw_response = self.llm_client.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(
                question=question,
                passages=search_result.results,
            ),
            response_schema=RAGResponse.model_json_schema(),
        )

        response = RAGResponse.model_validate(raw_response)

        self._validate_citations(
            citations=response.citations,
            retrieved_chunks=search_result.results,
        )

        return response

    @staticmethod
    def _validate_citations(
        citations: list[Citation],
        retrieved_chunks: list[RetrievedChunk],
    ) -> None:
        """Reject citations that do not point to supplied retrieval context."""
        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in retrieved_chunks
        }

        for citation in citations:
            chunk = chunks_by_id.get(citation.chunk_id)

            if chunk is None:
                message = (
                    "Citation references a non-retrieved chunk: "
                    f"{citation.chunk_id}"
                )
                raise ValueError(message)

            normalized_quote: str = normalize_citation_text(
                citation.quote
            )

            normalized_chunk_text: str = normalize_citation_text(
                chunk.text
            )

            if normalized_quote not in normalized_chunk_text:
                message = (
                    "Citation quote is not verbatim in chunk: "
                    f"{citation.chunk_id}\n"
                    f"Generated quote: {citation.quote!r}\n"
                    f"Source excerpt: {chunk.text[:600]!r}"
                )
                raise ValueError(message)

            if citation.document_id != chunk.document_id:
                message = (
                    "Citation document_id does not match chunk: "
                    f"{citation.chunk_id}"
                )
                raise ValueError(message)

            if citation.document_name != chunk.document_name:
                message = (
                    "Citation document_name does not match chunk: "
                    f"{citation.chunk_id}"
                )
                raise ValueError(message)

            if citation.section_title != chunk.section_title:
                message = (
                    "Citation section_title does not match chunk: "
                    f"{citation.chunk_id}"
                )
                raise ValueError(message)
