import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from insurance_rag_assistant.config import (
    EMBEDDING_DIMENSIONS,
    QDRANT_COLLECTION_NAME,
    QDRANT_PATH,
)
from insurance_rag_assistant.models.documents import DocumentChunk
from insurance_rag_assistant.models.retrieval import (
    RetrievedChunk,
    SearchFilters,
)


class LocalQdrantVectorStore:
    """Persistent local Qdrant storage for insurance document chunks."""

    def __init__(
        self,
        path: Path = QDRANT_PATH,
        collection_name: str = QDRANT_COLLECTION_NAME,
        vector_size: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        self.path = path
        self.collection_name = collection_name
        self.vector_size = vector_size

        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))

    def recreate_collection(self) -> None:
        """Delete and recreate the collection with cosine similarity."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def ensure_collection(self) -> None:
        """Create the collection only when it does not already exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunk vectors and all chunk metadata as Qdrant payload."""
        if len(chunks) != len(embeddings):
            message = (
                "Number of chunks and embeddings must match: "
                f"{len(chunks)} chunks, {len(embeddings)} embeddings."
            )
            raise ValueError(message)

        points: list[models.PointStruct] = []

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if len(embedding) != self.vector_size:
                message = (
                    f"Embedding dimension must be {self.vector_size}; "
                    f"received {len(embedding)} for {chunk.chunk_id}."
                )
                raise ValueError(message)

            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=chunk.model_dump(mode="json"),
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: list[float],
        filters: SearchFilters,
        top_k: int,
        min_score: float,
    ) -> list[RetrievedChunk]:
        """Return the top relevant chunks, subject to metadata filters."""
        if len(query_vector) != self.vector_size:
            message = (
                f"Query vector dimension must be {self.vector_size}; "
                f"received {len(query_vector)}."
            )
            raise ValueError(message)

        if not self.client.collection_exists(self.collection_name):
            message = (
                f"Collection '{self.collection_name}' does not exist. "
                "Run 'insurance-rag ingest --recreate' first."
            )
            raise RuntimeError(message)

        filter_conditions = self._build_filter_conditions(filters)

        query_filter = (
            models.Filter(must=filter_conditions) if filter_conditions else None
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=min_score,
            with_payload=True,
            with_vectors=False,
        )

        retrieved_chunks: list[RetrievedChunk] = []

        for rank, point in enumerate(response.points, start=1):
            payload = dict(point.payload or {})

            retrieved_data = {
                "chunk_id": payload.get("chunk_id"),
                "document_id": payload.get("document_id"),
                "document_name": payload.get("document_name"),
                "section_title": payload.get("section_title"),
                "section_path": payload.get("section_path"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "text": payload.get("text"),
                "score": point.score,
                "rank": rank,
                "language": payload.get("language"),
                "coverage": payload.get("coverage"),
                "jurisdiction": payload.get("jurisdiction"),
                "version": payload.get("version"),
            }

            retrieved_chunks.append(RetrievedChunk.model_validate(retrieved_data))

        return retrieved_chunks

    def count(self) -> int:
        """Return the number of indexed vectors."""
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )
        return result.count

    def close(self) -> None:
        """Close local Qdrant resources."""
        self.client.close()

    @staticmethod
    def _build_filter_conditions(
        filters: SearchFilters,
    ) -> list[models.Condition]:
        filter_values: dict[str, Any] = {
            "coverage": filters.coverage,
            "jurisdiction": filters.jurisdiction,
            "language": filters.language,
            "document_type": filters.document_type,
            "version": filters.version,
        }

        conditions: list[models.Condition] = []

        for field_name, field_value in filter_values.items():
            if field_value is not None:
                conditions.append(
                    models.FieldCondition(
                        key=field_name,
                        match=models.MatchValue(value=field_value),
                    )
                )

        return conditions
