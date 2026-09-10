from fastapi.testclient import TestClient

from insurance_rag_assistant.api.app import app
from insurance_rag_assistant.api.dependencies import (
    get_evidence_retrieval_service,
)
from insurance_rag_assistant.api.schemas.evidence import (
    EvidenceCitation,
    EvidenceRetrievalResponse,
)


class FakeEvidenceRetrievalService:
    def retrieve(self, request: object) -> EvidenceRetrievalResponse:
        return EvidenceRetrievalResponse(
            request_id="REQ-TEST-001",
            retrieval_status="SUCCESS",
            citations=[
                EvidenceCitation(
                    citation_id="UW-CYB-003:chunk-001",
                    source_document_id="CYBER-UW-GUIDE-001",
                    source_document_title="Cyber Underwriting Guide",
                    section_reference="Section 4.2",
                    excerpt=(
                        "Multi-factor authentication is required for all "
                        "remote and privileged access."
                    ),
                    relevance_score=0.95,
                    supported_finding_id="UW-CYB-003",
                )
            ],
            unresolved_finding_ids=[],
            knowledge_base_version="insurance_documents",
        )


def test_health_check_returns_ok() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "service": "insurance-rag-assistant",
        "status": "ok",
    }


def test_retrieve_evidence_returns_citations() -> None:
    app.dependency_overrides[get_evidence_retrieval_service] = (
        FakeEvidenceRetrievalService
    )

    client = TestClient(app)

    try:
        response = client.post(
            "/v1/retrieve-evidence",
            json={
                "request_id": "REQ-TEST-001",
                "top_k": 3,
                "filters": {
                    "coverage": "cyber",
                    "jurisdiction": "Quebec",
                    "language": "en",
                    "document_type": "underwriting_guide",
                    "version": None,
                },
                "queries": [
                    {
                        "supported_finding_id": "UW-CYB-003",
                        "query": (
                            "What are the MFA requirements for cyber "
                            "insurance underwriting?"
                        ),
                    }
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200

    body = response.json()

    assert body["request_id"] == "REQ-TEST-001"
    assert body["retrieval_status"] == "SUCCESS"
    assert len(body["citations"]) == 1
    assert body["citations"][0]["supported_finding_id"] == "UW-CYB-003"
    assert body["citations"][0]["relevance_score"] == 0.95
