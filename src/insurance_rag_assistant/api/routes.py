from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from insurance_rag_assistant.api.dependencies import (
    get_evidence_retrieval_service,
)
from insurance_rag_assistant.api.schemas.evidence import (
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
)
from insurance_rag_assistant.application.evidence_retrieval_service import (
    EvidenceRetrievalService,
)

router = APIRouter()

EvidenceRetrievalServiceDependency = Annotated[
    EvidenceRetrievalService,
    Depends(get_evidence_retrieval_service),
]


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "service": "insurance-rag-assistant",
        "status": "ok",
    }


@router.post(
    "/v1/retrieve-evidence",
    response_model=EvidenceRetrievalResponse,
    summary="Retrieve documentary evidence for underwriting findings",
)
def retrieve_evidence(
    request: EvidenceRetrievalRequest,
    service: EvidenceRetrievalServiceDependency,
) -> EvidenceRetrievalResponse:
    try:
        return service.retrieve(request)
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
