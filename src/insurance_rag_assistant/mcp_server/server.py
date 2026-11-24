"""MCP server exposing read-only documentary evidence retrieval over stdio."""

import logging
import sys
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations


from insurance_rag_assistant.api.dependencies import (
    get_evidence_retrieval_service,
)
from insurance_rag_assistant.api.schemas.evidence import (
    EvidenceQuery,
    EvidenceRetrievalRequest,
)
from insurance_rag_assistant.models.retrieval import SearchFilters, SearchQuery

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = MCPServer("insurance-knowledge")

_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


def _filters(
    coverage: str | None,
    jurisdiction: str | None,
    language: str | None,
    document_type: str | None,
) -> SearchFilters:
    values = {
        "coverage": coverage,
        "jurisdiction": jurisdiction,
        "language": language,
        "document_type": document_type,
    }
    return SearchFilters(**{k: v for k, v in values.items() if v})


@mcp.tool(annotations=_READ_ONLY)
def retrieve_evidence(
    request_id: str,
    queries: list[EvidenceQuery],
    top_k: int = 3,
    coverage: str | None = None,
    jurisdiction: str | None = None,
    language: str | None = None,
    document_type: str | None = None,
) -> dict[str, Any]:
    """Retrieve cited passages from the insurance knowledge base.

    Each query carries a supported_finding_id so evidence can be tied to an
    underwriting finding. Returns retrieval_status (SUCCESS, PARTIAL or
    INSUFFICIENT_CONTEXT), citations with excerpts, and the finding ids with
    no sufficient evidence. No LLM is called; excerpts are verbatim.
    """
    request = EvidenceRetrievalRequest(
        request_id=request_id,
        queries=queries,
        top_k=top_k,
        filters=_filters(coverage, jurisdiction, language, document_type),
    )
    response = get_evidence_retrieval_service().retrieve(request)
    return response.model_dump()


@mcp.tool(annotations=_READ_ONLY)
def search_documents(
    query: str,
    top_k: int = 5,
    coverage: str | None = None,
    jurisdiction: str | None = None,
    language: str | None = None,
    document_type: str | None = None,
) -> dict[str, Any]:
    """Search the knowledge base for a single question.

    Returns retrieval_sufficient (false means the corpus does not support an
    answer: do not answer from general knowledge) and ranked passages with
    document, section, chunk id, text, and scores.
    """
    top_k = max(1, min(top_k, 10))
    service = get_evidence_retrieval_service().search_service
    result = service.search(
        SearchQuery(
            query=query,
            top_k=top_k,
            filters=_filters(coverage, jurisdiction, language, document_type),
        )
    )
    return {
        "retrieval_sufficient": result.retrieval_sufficient,
        "results": [
            {
                "rank": r.rank,
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "document_name": r.document_name,
                "section_title": r.section_title,
                "text": r.text,
                "score": r.score,
                "rerank_score": getattr(r, "rerank_score", None),
            }
            for r in result.results
        ],
    }


def main() -> None:
    get_evidence_retrieval_service()  # load models before the first call
    mcp.run()  # stdio transport


if __name__ == "__main__":
    main()
