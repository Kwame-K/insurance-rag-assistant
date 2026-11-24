# Insurance Knowledge Agent (Insurance RAG Assistant)

[

A grounded, bilingual (English/French) Retrieval-Augmented Generation (RAG) service for insurance and insurance-regulation documentation. It ingests documents, creates 768-dimensional semantic embeddings, retrieves relevant passages with hybrid (dense + lexical) search and cross-encoder reranking, abstains when the evidence is weak, and exposes verified documentary evidence through a CLI and a FastAPI service.

This repository is **Project 3** in a five-project insurance agentic AI portfolio. It is designed to be reused by downstream systems, including the Underwriting Agent (Project 4).

> **Scope note:** this is a portfolio and learning project. The commercial-insurance documents (cyber underwriting guide, commercial property policy, appetite and exclusions references) are synthetic. The regulatory documents (OSFI guidelines, Civil Code of Québec) are public texts used to test retrieval on real material. Nothing here is insurance advice, a policy-administration system, or a coverage determination engine.

## Problem

Insurance and regulatory wording is long, technical, and spread across grants, exclusions, definitions, conditions, and cross-references. A generic LLM can give a plausible answer that omits a condition, cites the wrong section, or invents wording. This project retrieves the relevant passages first, refuses when the corpus does not support an answer, and validates every citation against the retrieved text.

## Key Features

- Bilingual retrieval in English and French (`intfloat/multilingual-e5-base`, 768 dimensions)
- Two vector backends: local Qdrant (default) and PostgreSQL with pgvector
- Optional hybrid search (dense + lexical, fused with RRF)
- Optional cross-encoder reranking with a score-based abstention gate
- Metadata filtering by coverage, jurisdiction, language, document type, and version
- Optional grounded LLM answers through Groq structured output
- Deterministic verbatim citation validation against retrieved passages
- FastAPI evidence-retrieval endpoint that returns evidence without calling an LLM
- Read-only MCP server (stdio) exposing evidence retrieval and document search to MCP clients
- Document-level and section-level retrieval evaluation, with abstention metrics
- Corpus scripts: fetch public documents, convert PDF/HTML to Markdown, calibrate thresholds
- Pydantic contracts, Ruff, mypy, pytest, and GitHub Actions CI

## Architecture

```text
Source documents (PDF / HTML / Markdown)
        |  scripts/fetch_corpus.py, scripts/convert_corpus.py
        v
Markdown with YAML front matter (data/external/markdown, data/combined)
        |
        v
Ingestion: loaders -> section-aware chunker -> E5 embeddings
        |
        v
Vector store: Qdrant (default) or PostgreSQL + pgvector
        |
        v
SemanticSearchService
   dense search  (+ lexical search if HYBRID_SEARCH=true, fused with RRF)
   -> cross-encoder rerank of the top candidates (if RERANKER=true)
   -> abstention gate (rerank score threshold)
        |
        +-------------------+-------------------+
        v                   v                   v
   CLI search      CLI ask + Groq        FastAPI evidence
                    grounded generation   retrieval (api/)
                                                 |
                                                 v
                                   Project 4 Underwriting Agent
```

| Layer | Responsibility |
|---|---|
| Ingestion | Load Markdown, validate metadata, create section-aware chunks, index vectors |
| Embeddings | Normalized E5 passage and query vectors |
| Vector store | Persist chunks and metadata; dense and lexical search (Qdrant or pgvector) |
| Retrieval | Filters, hybrid fusion, reranking, relevance and abstention gates |
| Generation | Optional grounded answers through a structured LLM client |
| Citation validation | Every quote must be verbatim in a retrieved chunk |
| API | Return documentary evidence to downstream services without an LLM call |

See [System Architecture](docs/architecture/system-architecture.md) for component details and sequence diagrams.

## Documents

| Document | Type | Language |
|---|---|---|
| Commercial property policy, cyber underwriting guide, underwriting appetite (Québec), exclusions reference | Synthetic insurance documents | EN/FR |
| OSFI Guideline B-13, Technology and Cyber Risk Management | Public regulation | EN |
| OSFI Guideline B-10, Third-Party Risk Management | Public regulation | EN |
| OSFI Guideline E-21 (2024), Operational Risk Management and Resilience | Public regulation | EN |
| OSFI Guideline E-21 (2016), Operational Risk Management | Public regulation (archived) | EN |
| Code civil du Québec, Des assurances (art. 2389 et suivants) | Public law | FR |

Public documents are listed in `data/sources.yaml` and downloaded into `data/external/raw/`, which is git-ignored. Check each publisher's reproduction conditions before redistributing any text.

Each document and chunk carries structured metadata: `document_id`, `title`, `document_type`, `coverage`, `jurisdiction`, `language`, `version`, `effective_date`, `section_title`, `section_path`, `chunk_id`, and `content_hash`.

## Project Structure

```text
insurance-rag-assistant/
├── .github/workflows/ci.yml
├── data/
│   ├── sources.yaml                  # Public source registry
│   ├── raw/                          # Synthetic Markdown documents
│   ├── external/{raw,markdown}/      # Downloaded and converted public documents
│   ├── combined/                     # Corpus indexed by ingest
│   └── processed/                    # Generated chunks.jsonl
├── scripts/
│   ├── fetch_corpus.py               # Download public documents
│   ├── convert_corpus.py             # PDF/HTML -> Markdown with front matter
│   └── calibrate_threshold.py        # Abstention threshold calibration
├── artifacts/                        # Evaluation reports
├── docs/architecture/
│   ├── evaluation-baseline.md
│   └── system-architecture.md
├── src/insurance_rag_assistant/
│   ├── api/                          # FastAPI app, routes, schemas
│   ├── application/                  # Evidence retrieval service
│   ├── evaluation/                   # Evaluator, cases.json, cases_real.json
│   ├── generation/                   # Answer generator, prompts
│   ├── ingestion/                    # Loaders, chunker, converters, pipeline
│   ├── llm/                          # Base, fake, Groq clients
│   ├── mcp_server/                   # Read-only MCP server (stdio)
│   ├── models/                       # Pydantic models
│   ├── retrieval/                    # Embedder, search, reranker, stores
│   ├── cli.py
│   └── config.py
├── tests/
├── compose.yaml, compose.pg.yaml     # Docker (app, PostgreSQL + pgvector)
├── Dockerfile, pyproject.toml, uv.lock
```

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.11+ |
| Environment | uv |
| Web framework | FastAPI + uvicorn |
| LLM integration | Groq (structured output), provider-agnostic interface |
| Embeddings | `intfloat/multilingual-e5-base` via `sentence-transformers` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Vector store | Qdrant (local) or PostgreSQL 17 + pgvector |
| Document conversion | PyMuPDF (`pymupdf4llm`), BeautifulSoup |
| CLI | Typer |
| Quality | pytest, mypy, Ruff, GitHub Actions |

## Setup

### Prerequisites

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/)
- Internet access on first run to download the embedding and reranker models
- Docker, only for the pgvector backend
- A Groq API key, only for `insurance-rag ask` and generation evaluation

### Installation

```bash
git clone https://github.com/Kwame-K/insurance-rag-assistant.git
cd insurance-rag-assistant
uv sync --all-groups
```

### Configuration

Settings are read from environment variables or a `.env` file (never commit it).

| Variable | Default | Purpose |
|---|---|---|
| `VECTOR_BACKEND` | `qdrant` | `qdrant` or `pgvector` |
| `DATABASE_URL` | local PostgreSQL on port 5433 | pgvector connection string |
| `HYBRID_SEARCH` | `false` | Add lexical search fused with dense search |
| `RERANKER` | `false` | Rerank candidates with the cross-encoder |
| `RERANKER_CANDIDATES` | `20` | Candidates passed to the reranker |
| `RERANK_ABSTENTION_SCORE` | unset | Abstain when the top rerank score is below this value |
| `GROQ_API_KEY` | unset | Needed only for generation |

Recommended configuration (the one evaluated below):

```dotenv
VECTOR_BACKEND=pgvector
HYBRID_SEARCH=true
RERANKER=true
RERANKER_CANDIDATES=20
RERANK_ABSTENTION_SCORE=0
```

### PostgreSQL with pgvector

```bash
docker compose -f compose.yaml -f compose.pg.yaml up -d postgres
```

## Building the Corpus

Public documents are described in `data/sources.yaml`.

```bash
uv run python scripts/fetch_corpus.py                    # download missing documents
uv run python scripts/convert_corpus.py                  # convert all
uv run python scripts/convert_corpus.py --only osfi_b10  # convert one
```

`fetch_corpus.py` writes `data/external/raw/` and a download manifest with URL, date, and SHA-256. Sources marked `manual: true` must be downloaded by hand into `data/external/raw/<id>.<format>`. `convert_corpus.py` writes `data/external/markdown/<id>.md` with YAML front matter and validates each file with the project's own loader and chunker.

To add a document: register it in `data/sources.yaml`, place or download the source file, convert it, check the headings, then copy the Markdown into the indexed corpus directory.

## Ingestion

```bash
uv run insurance-rag ingest --source-dir data/combined --recreate
```

This loads Markdown documents, validates metadata, creates section-aware chunks, embeds them, rebuilds the collection, and writes `data/processed/chunks.jsonl`. Always ingest the whole corpus with `--recreate`, otherwise documents missing from the source directory are removed from the index.

## Usage

### Search

```bash
uv run insurance-rag search \
  --query "What are OSFI's expectations for patching in a timely and controlled manner?" \
  --top-k 5
```

Filters: `--coverage`, `--jurisdiction`, `--language`, `--document-type`, `--version`. When no passage passes the relevance and abstention gates, the command reports that and exits with code 1.

### Grounded Answer

```bash
uv run insurance-rag ask \
  --query "Is water damage caused by a sudden plumbing leak covered under the commercial property policy?" \
  --coverage commercial_property --language en --top-k 5
```

`ask` calls Groq only after retrieval succeeds. Every citation must refer to a retrieved chunk and every quote must appear verbatim in that chunk.

### Evaluation

```bash
uv run insurance-rag evaluate \
  --cases src/insurance_rag_assistant/evaluation/cases_real.json \
  --output artifacts/evaluation.json
```

Add `--with-generation` to also evaluate grounding, citations, and answer constraints (one Groq request per case).

## FastAPI Service

```bash
uv run uvicorn insurance_rag_assistant.api.app:app --reload --port 8001
```

```text
GET  /health
POST /v1/retrieve-evidence
GET  /docs
```

The service returns relevant source passages and metadata without calling an LLM.

```bash
curl -X POST "http://127.0.0.1:8001/v1/retrieve-evidence" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "REQ-MFA-001",
    "top_k": 3,
    "queries": [{
      "supported_finding_id": "UW-CYB-003",
      "query": "What are the multi-factor authentication requirements for cyber insurance underwriting?"
    }]
  }'
```

| Status | Meaning |
|---|---|
| `SUCCESS` | Every requested finding has at least one relevant citation |
| `PARTIAL` | Some findings have no sufficient evidence |
| `INSUFFICIENT_CONTEXT` | No finding produced passages above the threshold |

## MCP Server

The same retrieval service is also available as a [Model Context Protocol](https://modelcontextprotocol.io) server, so MCP-compatible clients can query the knowledge base directly. It uses the stdio transport, runs the same retrieval pipeline as the FastAPI service, and never calls an LLM.

```bash
uv run insurance-rag-mcp
```

The server loads the embedding model before serving its first request, writes logs to stderr (stdout is reserved for the protocol), and reads the same configuration as the API. The vector store (Qdrant or pgvector) must already be populated by the ingestion step.

| Tool | Purpose |
|---|---|
| `retrieve_evidence` | Takes a `request_id` and a list of queries, each tied to a `supported_finding_id`. Returns `retrieval_status` (`SUCCESS`, `PARTIAL`, or `INSUFFICIENT_CONTEXT`), verbatim citations, and the finding ids without sufficient evidence |
| `search_documents` | Takes a single question. Returns `retrieval_sufficient` and ranked passages with document, section, chunk id, text, and scores. `top_k` is clamped to 1-10 |

Both tools accept optional `coverage`, `jurisdiction`, `language`, and `document_type` filters. Both are annotated as read-only and idempotent. When `retrieval_sufficient` is false, the corpus does not support an answer and the client should not answer from general knowledge.

Example client configuration (adjust the absolute path):

```json
{
  "mcpServers": {
    "insurance-knowledge": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/insurance-rag-assistant", "insurance-rag-mcp"]
    }
  }
}
```

The MCP server is not part of the Docker Compose platform, because stdio servers are launched by the client process rather than exposed on a network port.

## Evaluation Results

Measured on 2026-10-02 with the recommended configuration. Metrics are document-level unless noted. Details and caveats are in [evaluation-baseline.md](docs/architecture/evaluation-baseline.md).

| Set | Cases | Recall@5 | MRR | Correct abstention | False abstention |
|---|---:|---:|---:|---:|---:|
| Public documents (`cases_real.json`) | 29 (23 answerable, 6 unanswerable) | 1.000 | 0.922 | 1.000 | 0.000 |
| Synthetic documents (`cases.json`) | 35 (21 answerable, 14 unanswerable) | 0.929 | 1.000 | 1.000 | 0.095 |

On the public-document set, section-level Hit@1 is 0.688, Hit@5 is 1.000, and Section MRR is 0.812.

Retrieval choices measured on the public-document set:

| Configuration | Recall@5 | MRR |
|---|---:|---:|
| Dense only, 20 rerank candidates | 0.957 | 0.873 |
| Hybrid, 20 rerank candidates | 1.000 | 0.922 |
| Hybrid, 100 rerank candidates | 0.957 | 0.906 |

## Known Limitations

- **Abstention threshold.** The threshold of 0 was chosen on the public-document cases. On the synthetic set it wrongly refuses 2 of 21 answerable questions (`holdout_burst_pipe_en`, `holdout_cyber_mfa_loose_en`), although the correct document is ranked first. Short or informal questions can score below 0 and trigger an abstention.
- **Small evaluation sets.** 29 and 35 cases, with only 6 unanswerable cases on the public documents. Differences of a few points are not statistically meaningful.
- **Hybrid gain is narrow.** It depends mostly on one case (`b13_patch_management_en`).
- **Cases adjusted after results.** One stale case was removed after B-10 was added to the corpus; that was decided after seeing its result and is not an independent measurement.
- **Broad single-word queries.** Short queries such as "patch management" can retrieve a larger, more general document before the specific section.
- This project is not a legal or coverage-determination tool.

## Integration with Project 4

Project 4 is the **Underwriting Agent**. It owns deterministic underwriting decisions, risk scoring, pricing indications, conditions, and human-review routing. This repository owns document retrieval and citations.

```text
Project 4: POST /underwrite, port 8002
        |
        v
HTTPInsuranceKnowledgeAgent
        |
        v
Project 3: POST /v1/retrieve-evidence, port 8001
        |
        v
Hybrid retrieval + reranking
        |
        v
Verbatim citations returned to UnderwritingDecision.evidence
```

| Project 3 owns | Project 4 owns |
|---|---|
| Documents and metadata | Submission validation |
| Chunking and embeddings | Completeness checks |
| Vector store and retrieval | Appetite and eligibility rules |
| Reranking and abstention | Risk score and pricing indication |
| Citation evidence | Referral, decline, and review workflow |

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Tests cover answer generation, the API, conversion, evaluation normalisation and section matching, false abstention, ingestion, models, the pgvector store, and the reranker. CI runs on GitHub Actions.

Repeat the evaluation after any change to the corpus, chunking, embeddings, filters, `top-k`, hybrid weights, reranker, thresholds, prompts, or response schema. Store each report with the configuration used.
