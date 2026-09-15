FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev


FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8001 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p \
        /app/data \
        /app/vector_store \
        /app/artifacts \
        /app/.cache/huggingface \
        /app/.cache/sentence-transformers \
    && chown -R appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

EXPOSE 8001

CMD ["uvicorn", "insurance_rag_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
