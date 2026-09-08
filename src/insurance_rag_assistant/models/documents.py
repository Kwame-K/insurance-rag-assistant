from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Language = Literal["en", "fr"]
DocumentType = Literal[
    "policy_wording",
    "underwriting_guide",
    "underwriting_appetite",
    "exclusions_reference",
    "pricing_guide",
    "procedure",
]


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    document_type: DocumentType
    coverage: str = Field(min_length=1)
    jurisdiction: str = Field(min_length=1)
    language: Language
    version: str = Field(min_length=1)
    effective_date: date
    status: Literal["active", "archived", "draft"]


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: DocumentMetadata
    source_path: Path
    raw_text: str = Field(min_length=1)


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        description="Stable unique identifier: document_id:section_slug:chunk_index."
    )
    document_id: str
    document_name: str
    document_type: DocumentType
    coverage: str
    jurisdiction: str
    language: Language
    version: str
    effective_date: date
    section_title: str
    section_path: list[str]
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    content_hash: str = Field(min_length=64, max_length=64)
