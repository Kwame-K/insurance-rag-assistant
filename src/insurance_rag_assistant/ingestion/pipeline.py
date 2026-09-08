from pathlib import Path

from insurance_rag_assistant.config import (
    CHUNK_MAX_CHARACTERS,
    CHUNK_OVERLAP_CHARACTERS,
)
from insurance_rag_assistant.ingestion.chunker import chunk_document
from insurance_rag_assistant.ingestion.loaders import load_markdown_documents
from insurance_rag_assistant.models.documents import DocumentChunk
from insurance_rag_assistant.retrieval.embedder import MultilingualE5Embedder
from insurance_rag_assistant.retrieval.vector_store import LocalQdrantVectorStore


def ingest_markdown_corpus(
    source_dir: Path,
    output_path: Path,
    recreate_collection: bool = False,
    max_characters: int = CHUNK_MAX_CHARACTERS,
    overlap_characters: int = CHUNK_OVERLAP_CHARACTERS,
) -> list[DocumentChunk]:
    """Load, chunk, embed, persist, and index a Markdown corpus."""
    documents = load_markdown_documents(source_dir)

    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document=document,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        )
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(chunk.model_dump_json())
            file.write("\n")

    embedder = MultilingualE5Embedder()
    embeddings = embedder.embed_passages([chunk.text for chunk in chunks])

    vector_store = LocalQdrantVectorStore()

    try:
        if recreate_collection:
            vector_store.recreate_collection()
        else:
            vector_store.ensure_collection()

        vector_store.upsert_chunks(chunks, embeddings)
    finally:
        vector_store.close()

    return chunks
