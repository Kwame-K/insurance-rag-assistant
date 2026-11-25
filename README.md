# Insurance Knowledge Agent (Insurance RAG Assistant)

[![Continuous Integration](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

A bilingual retrieval-augmented generation (RAG) service for insurance and insurance-regulation documents. It indexes documents, retrieves relevant passages, and returns traceable documentary evidence. Optional answer generation uses Groq; the evidence API and MCP retrieval tools do not call an LLM.

This repository is Project 3 in the Insurance Agentic Platform and supplies documentary evidence to the Underwriting Agent.

> Commercial insurance examples are synthetic. Public legal and regulatory texts are used to evaluate retrieval. This repository does not give legal advice, determine coverage, or make underwriting decisions.

## Problem

A plausible answer is not enough when coverage language depends on exclusions, conditions, definitions, and cross-references. The service retrieves source passages first, can abstain when the evidence is insufficient, and checks generated quotations against the retrieved text.

## Key Features

- English and French retrieval with `intfloat/multilingual-e5-base` and 768-dimensional embeddings.
- Local Qdrant by default, with PostgreSQL and pgvector as an alternative.
- Optional hybrid dense and lexical search, fused with reciprocal rank fusion.
- Optional cross-encoder reranking and a score-based abstention gate.
- Metadata filters for coverage, jurisdiction, language, document type, and version.
- Optional Groq generation with verbatim citation validation.
- FastAPI evidence retrieval, CLI search and evaluation, and read-only MCP retrieval.
- Document- and section-level evaluation with abstention metrics.

## Architecture

```text
PDF / HTML / Markdown sources
    ↓
Markdown with YAML metadata
    ↓
Section-aware chunking → E5 embeddings → Qdrant or pgvector
    ↓
Dense retrieval → optional lexical fusion → optional reranking and abstention
    ├→ CLI search
    ├→ FastAPI evidence retrieval / MCP retrieval
    └→ CLI ask / FastAPI POST /v1/ask  → Groq → citation validation
```

See [System Architecture](docs/architecture/system-architecture.md).

## Documents

| Corpus material | Classification |
|---|---|
| Cyber underwriting guide, commercial property policy, Québec appetite and exclusions references | Synthetic insurance documents |
| OSFI B-13 and B-10 | Public guidelines |
| OSFI E-21, 2024 and archived 2016 versions | Public guidelines |
| Code civil du Québec, Des assurances (art. 2389 et suivants) | Public law |

Public sources are registered in `data/sources.yaml`. Raw downloads under `data/external/raw/` are git-ignored. Check publishers' reproduction conditions before redistributing source material.

Documents and chunks carry identifiers, title, document type, coverage, jurisdiction, language, version, effective date, section information, and content hash.

## Project Structure

```text
insurance-rag-assistant/
├── data/
│   ├── sources.yaml
│   ├── raw/
│   ├── external/
│   ├── combined/
│   └── processed/
├── scripts/
│   ├── fetch_corpus.py
│   ├── convert_corpus.py
│   └── calibrate_threshold.py
├── docs/architecture/
├── src/insurance_rag_assistant/
│   ├── api/
│   ├── application/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── llm/
│   ├── mcp_server/
│   ├── models/
│   ├── retrieval/
│   ├── cli.py
│   └── config.py
├── tests/
├── compose.yaml
├── compose.pg.yaml
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Technology Stack

Python 3.11+, uv, FastAPI, Pydantic, Typer, sentence-transformers, Qdrant, PostgreSQL 17 with pgvector, Groq, PyMuPDF and BeautifulSoup for conversion, pytest, Ruff, and mypy.

## Setup

```bash
git clone [https://github.com/Kwame-K/insurance-rag-assistant.git](https://github.com/Kwame-K/insurance-rag-assistant.git)
cd insurance-rag-assistant
uv sync --all-groups
```

Internet access is needed when downloading models for the first time. A Groq key is needed for `ask` and generation evaluation, not for retrieval alone. Settings are read from environment variables or a local `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `VECTOR_BACKEND` | `qdrant` | Select Qdrant or pgvector |
| `DATABASE_URL` | Local PostgreSQL on port `5433` | pgvector connection |
| `HYBRID_SEARCH` | `false` | Combine dense and lexical results |
| `RERANKER` | `false` | Enable cross-encoder reranking |
| `RERANKER_CANDIDATES` | `20` | Candidate count passed to the reranker |
| `RERANK_ABSTENTION_SCORE` | Unset | Optional minimum top rerank score |
| `GROQ_API_KEY` | Unset | Enable generation workflows |

### Default Deployment vs. Evaluation Configuration

The central [Insurance Agentic Platform](https://github.com/Kwame-K/insurance-agentic-platform) starts the service as `knowledge-agent` on port `8001`. It mounts volumes for the local Qdrant store, artifacts, and model cache. It does not set `VECTOR_BACKEND=pgvector` or enable the hybrid and reranking settings used for the evaluation below. The central PostgreSQL service provisions a reserved knowledge database, but that is not the vector store used by the default platform deployment.

To reproduce the configuration used for the recorded pgvector evaluation:

```bash
docker compose -f compose.yaml -f compose.pg.yaml up -d postgres
```

```dotenv
VECTOR_BACKEND=pgvector
HYBRID_SEARCH=true
RERANKER=true
RERANKER_CANDIDATES=20
RERANK_ABSTENTION_SCORE=0
```

Use the same vector backend for ingestion and retrieval.

## Building the Corpus

```bash
uv run python scripts/fetch_corpus.py
uv run python scripts/convert_corpus.py
```

The fetch script records source URL, date, and SHA-256 in a manifest. Sources marked `manual: true` must be placed in `data/external/raw/` manually. Conversion writes Markdown with YAML metadata to `data/external/markdown/`. Review the converted headings and include the Markdown files you want to index in `data/combined/`.

## Ingestion

```bash
uv run insurance-rag ingest --source-dir data/combined --recreate
```

The command validates metadata, creates section-aware chunks, embeds them, rebuilds the selected index, and writes `data/processed/chunks.jsonl`. Use the intended complete corpus when rebuilding the index.

## Usage

### Search

```bash
uv run insurance-rag search \
  --query "What are OSFI's expectations for patching in a timely and controlled manner?" \
  --top-k 5
```

Filters include `--coverage`, `--jurisdiction`, `--language`, `--document-type`, and `--version`. If no passage passes the retrieval gates, search reports insufficient evidence and exits with code 1.

### Grounded Answer

```bash
uv run insurance-rag ask \
  --query "Is water damage caused by a sudden plumbing leak covered under the commercial property policy?" \
  --coverage commercial_property --language en --top-k 5
```

`ask` calls Groq only after retrieval succeeds. Citations must refer to retrieved chunks, and quotations must appear verbatim in those chunks.

### Evaluation Command

```bash
uv run insurance-rag evaluate \
  --cases src/insurance_rag_assistant/evaluation/cases_real.json \
  --output artifacts/evaluation.json
```

Add `--with-generation` to include grounding and citation checks with Groq.

## FastAPI Service


```bash
uv run uvicorn insurance_rag_assistant.api.app:app --reload --port 8001
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Service liveness |
| POST | `/v1/retrieve-evidence` | Return passages and citations without an LLM call |
| GET | `/docs` | Interactive OpenAPI documentation |
| POST | `/v1/ask` | Generate an answer from retrieved passages and validate its citations |

### Ask a question with citations

`POST /v1/ask` retrieves relevant passages and uses Groq to generate an answer.
The service checks that returned citations point to retrieved chunks and that
their quotes appear in the cited passages. `POST /v1/retrieve-evidence` remains
available for retrieval without generation.

```bash
curl -fsS http://127.0.0.1:8001/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "Is water damage caused by a sudden plumbing leak covered under the commercial property policy?",
    "top_k": 5,
    "filters": {
      "coverage": "commercial_property",
      "language": "en"
    }
  }'
```

- `top_k` controls how many passages may be retrieved, not how many citations
  the answer will contain.
- If no passage meets the relevance threshold, the API returns HTTP 200 with
  `grounded: false` and does not call Groq.
- Generation requires `GROQ_API_KEY` on the Knowledge Agent. A missing key
  returns HTTP 503 when generation is needed; a provider or response validation
  failure returns HTTP 502.


```bash
curl -X POST http://127.0.0.1:8001/v1/retrieve-evidence \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id":"REQ-MFA-001",
    "top_k":3,
    "queries":[{
      "supported_finding_id":"UW-CYB-003",
      "query":"What are the multi-factor authentication requirements for cyber insurance underwriting?"
    }]
  }'
```

| Status | Meaning |
|---|---|
| `SUCCESS` | Every requested finding has supporting evidence |
| `PARTIAL` | Only some findings have sufficient evidence |
| `INSUFFICIENT_CONTEXT` | No requested finding has sufficient evidence |

## MCP Server

```bash
uv run insurance-rag-mcp
```

The server exposes `retrieve_evidence` for finding-specific citations and `search_documents` for ranked passages. It uses the same retrieval configuration as the API, requires a populated index, and makes no LLM calls. Logs go to stderr. The central platform does not launch the MCP server; the MCP client starts the stdio process.

## Evaluation Results

Recorded on 2026-10-02 with the pgvector configuration described above. The table reports document-level metrics.

| Dataset | Cases | Recall@5 | MRR | Correct abstention | False abstention |
|---|---:|---:|---:|---:|---:|
| Public documents | 29: 23 answerable, 6 unanswerable | 1.000 | 0.922 | 1.000 | 0.000 |
| Synthetic documents | 35: 21 answerable, 14 unanswerable | 0.929 | 1.000 | 1.000 | 0.095 |

On the public set, section Hit@1 was 0.688, section Hit@5 was 1.000, and section MRR was 0.812.

| Public-set configuration | Recall@5 | MRR |
|---|---:|---:|
| Dense only, 20 rerank candidates | 0.957 | 0.873 |
| Hybrid, 20 rerank candidates | 1.000 | 0.922 |
| Hybrid, 100 rerank candidates | 0.957 | 0.906 |

See the [evaluation baseline](docs/architecture/evaluation-baseline.md) for methodology and caveats. These are not scores for the default platform deployment.

## Known Limitations

- The abstention threshold of 0 was selected on the public-document cases and incorrectly abstained on two of the 21 answerable synthetic cases.
- Both evaluation sets are small. The observed hybrid improvement depended mainly on one public-document case.
- One stale public-set case was removed after its result was seen; that comparison is not an independent measurement.
- Broad queries may rank a general document ahead of the intended section.
- A valid citation is evidence that a passage was retrieved, not a legal opinion or coverage determination.

## Integration with Project 4

The [Underwriting Agent](https://github.com/Kwame-K/underwriting-agent) calls `POST /v1/retrieve-evidence` to attach citations to its findings. In the platform, this uses `http://knowledge-agent:8001`. This service owns retrieval and citation evidence; the Underwriting Agent owns eligibility, risk scoring, pricing, referrals, and human review. Documentary evidence does not override its deterministic rules.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Repeat the evaluation after changes to the corpus, chunking, embeddings, filters, search configuration, reranking, thresholds, prompts, or response schema. Record the configuration with each report.
