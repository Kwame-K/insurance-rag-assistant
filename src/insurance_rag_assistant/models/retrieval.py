from pydantic import BaseModel, ConfigDict, Field

from insurance_rag_assistant.models.documents import (
    DocumentType,
    Language,
)


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coverage: str | None = None
    jurisdiction: str | None = None
    language: Language | None = None
    document_type: DocumentType | None = None
    version: str | None = None


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    document_name: str
    section_title: str
    section_path: list[str]
    page_start: int | None = None
    page_end: int | None = None
    text: str
    score: float = Field(ge=-1.0, le=1.0)
    rank: int = Field(ge=1)
    language: Language
    coverage: str
    jurisdiction: str
    version: str


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: SearchQuery
    results: list[RetrievedChunk]
    retrieval_sufficient: bool
