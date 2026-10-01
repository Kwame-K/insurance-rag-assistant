# Manual patches (files I could not see in full)

## pyproject.toml
dependencies += ["psycopg[binary,pool]>=3.2", "pgvector>=0.4"]

## src/insurance_rag_assistant/config.py  (class Settings)
from typing import Literal
    database_url: str = "postgresql://rag:rag_dev_password@localhost:5433/knowledge"
    vector_backend: Literal["qdrant", "pgvector"] = "qdrant"

## src/insurance_rag_assistant/retrieval/search.py
from insurance_rag_assistant.retrieval.factory import create_vector_store
from insurance_rag_assistant.retrieval.protocols import VectorStore
    vector_store: VectorStore | None = None            # type hint
    self.vector_store = vector_store or create_vector_store()

## ingestion/pipeline.py and cli.py
Replace every direct `LocalQdrantVectorStore()` with `create_vector_store()`.

## .env.example
VECTOR_BACKEND=pgvector
DATABASE_URL=postgresql://rag:rag_dev_password@localhost:5433/knowledge

## .github/workflows/ci.yml  (job that runs pytest)
    services:
      postgres:
        image: pgvector/pgvector:pg17
        env:
          POSTGRES_USER: rag
          POSTGRES_PASSWORD: test
          POSTGRES_DB: knowledge_test
        ports: ["5432:5432"]
        options: >-
          --health-cmd "pg_isready -U rag -d knowledge_test"
          --health-interval 5s --health-timeout 3s --health-retries 10
    env:
      DATABASE_URL_TEST: postgresql://rag:test@localhost:5432/knowledge_test

## Run locally
docker compose -f compose.yaml -f compose.pg.yaml up -d postgres
uv sync
uv run python -m insurance_rag_assistant.db.migrate
DATABASE_URL_TEST=postgresql://rag:rag_dev_password@localhost:5433/knowledge uv run pytest tests/test_pg_store.py -v
VECTOR_BACKEND=pgvector uv run insurance-rag ingest      # adapt to your CLI command name
