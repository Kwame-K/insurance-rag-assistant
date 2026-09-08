from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_FILE = PROCESSED_DATA_DIR / "chunks.jsonl"

VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
QDRANT_PATH = VECTOR_STORE_DIR / "qdrant"
QDRANT_COLLECTION_NAME = "insurance_documents"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIMENSIONS = 768
EMBEDDING_BATCH_SIZE = 32


DEFAULT_TOP_K = 5
MIN_RETRIEVAL_SCORE = 0.55

CHUNK_MAX_CHARACTERS = 1_600
CHUNK_OVERLAP_CHARACTERS = 200

GROQ_MODEL_NAME = "openai/gpt-oss-20b"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

EVALUATION_CASES_FILE = (
    PROJECT_ROOT / "src" / "insurance_rag_assistant" / "evaluation" / "cases.json"
)

EVALUATION_REPORT_FILE = ARTIFACTS_DIR / "retrieval_evaluation_report.json"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str | None = None
    groq_model_name: str = GROQ_MODEL_NAME


settings = Settings()
