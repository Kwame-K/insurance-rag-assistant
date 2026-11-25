from functools import lru_cache
from threading import Lock

from insurance_rag_assistant.application.evidence_retrieval_service import (
    EvidenceRetrievalService,
)
from insurance_rag_assistant.config import MIN_RETRIEVAL_SCORE
from insurance_rag_assistant.retrieval.search import SemanticSearchService

_search_init_lock = Lock()


@lru_cache(maxsize=1)
def _build_search_service() -> SemanticSearchService:
    return SemanticSearchService(min_retrieval_score=MIN_RETRIEVAL_SCORE)


def get_search_service() -> SemanticSearchService:
    # Évite une double initialisation si les deux routes reçoivent
    # leur première requête en même temps.
    with _search_init_lock:
        return _build_search_service()


@lru_cache(maxsize=1)
def get_evidence_retrieval_service() -> EvidenceRetrievalService:
    return EvidenceRetrievalService(search_service=get_search_service())
