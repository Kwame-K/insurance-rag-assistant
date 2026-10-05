from typing import Any, Protocol

from insurance_rag_assistant.models.retrieval import RetrievedChunk


class CrossEncoderModel(Protocol):
    """Minimal cross-encoder interface (sentence-transformers CrossEncoder)."""

    def predict(self, sentences: Any, **kwargs: Any) -> Any: ...


class CrossEncoderReranker:
    """Re-score retrieved passages locally with a cross-encoder (no LLM call)."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        max_length: int = 512,
        model: CrossEncoderModel | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        if model is None:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, max_length=max_length)
        self.model = model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return the top_k chunks ordered by cross-encoder relevance."""
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scores = [float(score) for score in raw_scores]

        ordered = sorted(
            zip(chunks, scores, strict=True),
            key=lambda pair: (-pair[1], pair[0].rank),
        )
        return [
            chunk.model_copy(update={"rank": rank, "rerank_score": score})
            for rank, (chunk, score) in enumerate(ordered[:top_k], start=1)
        ]
