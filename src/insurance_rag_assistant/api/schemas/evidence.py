from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from insurance_rag_assistant.models.retrieval import SearchFilters

RetrievalStatus: TypeAlias = Literal[
    "SUCCESS",
    "PARTIAL",
    "INSUFFICIENT_CONTEXT",
]


class EvidenceQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supported_finding_id: str = Field(min_length=1)
    query: str = Field(min_length=3)


class EvidenceRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    queries: list[EvidenceQuery] = Field(min_length=1, max_length=10)
    top_k: int = Field(default=3, ge=1, le=10)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class EvidenceCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_document_title: str = Field(min_length=1)
    section_reference: str | None = None

    excerpt: str = Field(min_length=1)
    relevance_score: float = Field(ge=-1.0, le=1.0)

    supported_finding_id: str = Field(min_length=1)


class EvidenceRetrievalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    retrieval_status: RetrievalStatus

    citations: list[EvidenceCitation]
    unresolved_finding_ids: list[str]

    knowledge_base_version: str
