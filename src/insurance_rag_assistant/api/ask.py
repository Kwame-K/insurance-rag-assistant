import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from groq import APIError as GroqAPIError
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from insurance_rag_assistant.api.dependencies import get_search_service
from insurance_rag_assistant.config import settings
from insurance_rag_assistant.generation.answer_generator import (
    GroundedAnswerGenerator,
)
from insurance_rag_assistant.llm.groq_client import GroqStructuredLLMClient
from insurance_rag_assistant.models.response import RAGResponse
from insurance_rag_assistant.models.retrieval import SearchFilters, SearchQuery
from insurance_rag_assistant.retrieval.search import SemanticSearchService

logger = logging.getLogger(__name__)
router = APIRouter()

Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=500),
]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    top_k: int = Field(default=3, ge=1, le=10)
    filters: SearchFilters = Field(default_factory=SearchFilters)


@router.post("/v1/ask", response_model=RAGResponse)
def ask(
    payload: AskRequest,
    search_service: Annotated[SemanticSearchService, Depends(get_search_service)],
) -> RAGResponse:
    try:
        result = search_service.search(
            SearchQuery(
                query=payload.question,
                top_k=payload.top_k,
                filters=payload.filters,
            )
        )
    except RuntimeError as exc:
        logger.warning("RAG ask retrieval failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="La recherche documentaire est indisponible.",
        ) from exc

    if not result.retrieval_sufficient:
        return RAGResponse(
            answer="Les documents retrouvés ne suffisent pas pour répondre.",
            answer_language="fr",
            grounded=False,
            confidence="low",
            citations=[],
            insufficient_context_reason=(
                "Aucun passage n’a atteint le seuil de pertinence requis."
            ),
        )

    if not settings.groq_api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "La génération n’est pas configurée côté RAG. "
                "Le mode Passages sources reste disponible."
            ),
        )

    try:
        generator = GroundedAnswerGenerator(llm_client=GroqStructuredLLMClient())
        return generator.generate(
            question=payload.question,
            search_result=result,
        )
    except (GroqAPIError, RuntimeError, TypeError, ValueError) as exc:
        # Ne pas renvoyer au navigateur une sortie brute du modèle
        # ou une exception contenant des extraits du corpus.
        logger.warning("RAG ask generation failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail=(
                "Impossible de produire une réponse citée valide. "
                "Essaie le mode Passages sources."
            ),
        ) from exc
