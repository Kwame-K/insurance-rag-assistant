from insurance_rag_assistant.config import settings
from insurance_rag_assistant.retrieval.protocols import VectorStore


def create_vector_store() -> VectorStore:
    """Return the backend selected by VECTOR_BACKEND (qdrant | pgvector)."""
    if settings.vector_backend == "pgvector":
        from insurance_rag_assistant.retrieval.pg_store import PgVectorStore

        return PgVectorStore()

    from insurance_rag_assistant.retrieval.vector_store import LocalQdrantVectorStore

    return LocalQdrantVectorStore()
