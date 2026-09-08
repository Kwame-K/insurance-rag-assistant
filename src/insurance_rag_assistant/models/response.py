from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from insurance_rag_assistant.models.documents import Language


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    document_name: str
    section_title: str
    page_start: int | None = Field(ge=1)
    page_end: int | None = Field(ge=1)
    quote: str = Field(
        min_length=1,
        description="Verbatim excerpt supporting the answer.",
    )
    retrieval_score: float = Field(ge=-1.0, le=1.0)


class RAGResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    answer_language: Language
    grounded: bool
    confidence: Literal["high", "medium", "low"]
    citations: list[Citation]
    insufficient_context_reason: str | None

    @model_validator(mode="after")
    def validate_grounding_state(self) -> "RAGResponse":
        if self.grounded and not self.citations:
            raise ValueError("A grounded response must contain at least one citation.")

        if self.grounded and self.insufficient_context_reason is not None:
            raise ValueError(
                "A grounded response cannot contain an insufficient-context reason."
            )

        if not self.grounded and self.citations:
            raise ValueError("An ungrounded response must not contain citations.")

        if not self.grounded and self.insufficient_context_reason is None:
            raise ValueError(
                "An ungrounded response must explain why context is insufficient."
            )

        return self
