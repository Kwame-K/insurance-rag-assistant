from insurance_rag_assistant.api.schemas.evidence import (
    EvidenceCitation,
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
    RetrievalStatus,
)
from insurance_rag_assistant.config import QDRANT_COLLECTION_NAME
from insurance_rag_assistant.models.retrieval import SearchQuery, SearchResult
from insurance_rag_assistant.retrieval.search import SemanticSearchService


class EvidenceRetrievalService:
    """Retrieve documentary evidence for underwriting findings."""

    def __init__(self, search_service: SemanticSearchService) -> None:
        self.search_service = search_service

    def retrieve(
        self,
        request: EvidenceRetrievalRequest,
    ) -> EvidenceRetrievalResponse:
        citations: list[EvidenceCitation] = []
        unresolved_finding_ids: list[str] = []

        for evidence_query in request.queries:
            search_query = SearchQuery(
                query=evidence_query.query,
                top_k=request.top_k,
                filters=request.filters,
            )

            search_result = self.search_service.search(search_query)

            if not search_result.retrieval_sufficient:
                unresolved_finding_ids.append(evidence_query.supported_finding_id)
                continue

            citations.extend(
                self._build_citations(
                    supported_finding_id=evidence_query.supported_finding_id,
                    search_result=search_result,
                )
            )

        retrieval_status = self._determine_retrieval_status(
            citations=citations,
            unresolved_finding_ids=unresolved_finding_ids,
        )

        return EvidenceRetrievalResponse(
            request_id=request.request_id,
            retrieval_status=retrieval_status,
            citations=citations,
            unresolved_finding_ids=unresolved_finding_ids,
            knowledge_base_version=QDRANT_COLLECTION_NAME,
        )

    @staticmethod
    def _build_citations(
        supported_finding_id: str,
        search_result: SearchResult,
    ) -> list[EvidenceCitation]:
        retrieved_chunks = search_result.results

        return [
            EvidenceCitation(
                citation_id=(f"{supported_finding_id}:{retrieved_chunk.chunk_id}"),
                source_document_id=retrieved_chunk.document_id,
                source_document_title=retrieved_chunk.document_name,
                section_reference=retrieved_chunk.section_title,
                excerpt=retrieved_chunk.text,
                relevance_score=retrieved_chunk.score,
                supported_finding_id=supported_finding_id,
            )
            for retrieved_chunk in retrieved_chunks
        ]

    @staticmethod
    def _determine_retrieval_status(
        citations: list[EvidenceCitation],
        unresolved_finding_ids: list[str],
    ) -> RetrievalStatus:
        if not citations:
            return "INSUFFICIENT_CONTEXT"

        if unresolved_finding_ids:
            return "PARTIAL"

        return "SUCCESS"
