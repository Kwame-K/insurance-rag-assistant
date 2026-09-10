from functools import lru_cache

from insurance_rag_assistant.application.evidence_retrieval_service import (
    EvidenceRetrievalService,
)
from insurance_rag_assistant.config import MIN_RETRIEVAL_SCORE
from insurance_rag_assistant.retrieval.search import SemanticSearchService


@lru_cache
def get_evidence_retrieval_service() -> EvidenceRetrievalService:
    search_service = SemanticSearchService(
        min_retrieval_score=MIN_RETRIEVAL_SCORE,
    )

    return EvidenceRetrievalService(
        search_service=search_service,
    )
