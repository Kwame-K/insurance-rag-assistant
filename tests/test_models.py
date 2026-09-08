from datetime import date

import pytest
from pydantic import ValidationError

from insurance_rag_assistant.models.documents import DocumentMetadata
from insurance_rag_assistant.models.retrieval import SearchQuery


def test_document_metadata_is_valid() -> None:
    metadata = DocumentMetadata(
        document_id="commercial_property_policy_v1",
        title="Commercial Property Policy",
        document_type="policy_wording",
        coverage="commercial_property",
        jurisdiction="quebec",
        language="en",
        version="1.0",
        effective_date=date(2026, 1, 1),
        status="active",
    )

    assert metadata.document_id == "commercial_property_policy_v1"
    assert metadata.language == "en"


def test_document_metadata_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DocumentMetadata(
            document_id="commercial_property_policy_v1",
            title="Commercial Property Policy",
            document_type="policy_wording",
            coverage="commercial_property",
            jurisdiction="quebec",
            language="en",
            version="1.0",
            effective_date=date(2026, 1, 1),
            status="active",
            unknown_field="invalid",
        )


def test_search_query_uses_default_top_k() -> None:
    query = SearchQuery(query="What flood exclusions apply?")

    assert query.top_k == 5
    assert query.filters.coverage is None
