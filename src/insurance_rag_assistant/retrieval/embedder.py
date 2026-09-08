from sentence_transformers import SentenceTransformer

from insurance_rag_assistant.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
)


class MultilingualE5Embedder:
    """Generate normalized E5 embeddings for passages and user queries."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, passages: list[str]) -> list[list[float]]:
        """Embed corpus passages using the E5 passage prefix."""
        prefixed_passages = [f"passage: {passage}" for passage in passages]

        embeddings = self.model.encode(
            prefixed_passages,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a user search query using the E5 query prefix."""
        embedding = self.model.encode(
            f"query: {query}",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()
