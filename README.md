# Insurance Knowledge Agent (Insurance RAG Assistant)

[![Continuous Integration](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

A grounded, bilingual Retrieval-Augmented Generation (RAG) service for insurance documentation. The project ingests insurance documents, creates 768-dimensional semantic embeddings, stores them in a local Qdrant vector store, retrieves relevant passages, and exposes verified documentary evidence through both a CLI and a FastAPI service.

This repository is **Project 3** in a five-project insurance agentic AI portfolio. It is designed to be reusable by downstream systems, including the Underwriting Agent (Project 4).

> **Scope note:** this is a portfolio and learning project. Its documents, rules, outputs, and examples are synthetic or demonstrative. It is not production insurance advice, a policy-administration system, or a coverage determination engine.

## Problem

Insurance policy wording is long, technical, and distributed across coverage grants, exclusions, deductibles, definitions, conditions, and endorsements. A question that appears simple — for example, whether a plumbing leak is covered — may require evidence from several different sections of a policy.

A generic LLM may provide an answer that sounds plausible but omits a deductible, fails to mention a material condition, cites the wrong section, or invents policy language. This project addresses that problem by retrieving relevant policy passages before generation and validating each citation against the retrieved source text.

The application can answer questions about covered perils and coverage grants, policy exclusions and limitations, deductibles and sublimits, definitions and insured-property terminology, conditions (including vacancy and reporting conditions), policy-specific questions requiring evidence from multiple sections, and questions that cannot be supported by the indexed corpus.

## Key Features

- Bilingual insurance-document retrieval in English and French
- Semantic retrieval with `intfloat/multilingual-e5-base` embeddings (768 dimensions)
- Persistent local Qdrant vector store (`vector_store/qdrant`)
- Metadata filtering by coverage, jurisdiction, language, document type, and version
- Retrieval relevance gate to support abstention when evidence is insufficient
- Optional grounded LLM answers through Groq structured output
- Deterministic verbatim citation validation against retrieved source passages
- FastAPI evidence-retrieval endpoint that returns evidence without calling an LLM
- CLI workflows for ingestion, search, question answering, and evaluation
- Pydantic contracts, Ruff, mypy, pytest, and GitHub Actions CI

## Architecture

```text
Markdown insurance documents (data/raw/)
        |
        v
Ingestion pipeline (ingestion/loaders.py, chunker.py, pipeline.py)
        |
        v
Chunking + metadata (data/processed/chunks.jsonl)
        |
        v
Multilingual E5 embeddings, 768 dimensions (retrieval/embedder.py)
        |
        v
Local Qdrant collection: insurance_documents (retrieval/vector_store.py)
        |
        v
SemanticSearchService (retrieval/search.py)
        |
        +-------------------+-------------------+
        v                   v                   v
   CLI search      CLI ask + Groq        FastAPI evidence
   (cli.py)         grounded generation   retrieval (api/)
                    (generation/)                |
                                                  v
                                    Project 4 Underwriting Agent
```

The repository deliberately separates retrieval from generation:

| Layer | Responsibility |
|---|---|
| Ingestion | Load Markdown documents, validate metadata, create chunks, and index vectors |
| Embeddings | Generate normalized E5 passage and query vectors |
| Vector store | Persist chunks and metadata in local Qdrant; execute semantic search |
| Retrieval | Apply metadata filters, rank chunks, and enforce the minimum relevance threshold |
| Generation | Optionally generate grounded answers through a structured LLM client |
| Citation validation | Verify that every generated quote is verbatim in a retrieved chunk |
| API | Return documentary evidence to downstream services without making an LLM call |

For a detailed architecture description, component responsibilities, sequence diagrams, and citation-validation flow, see [System Architecture](docs/architecture/system-architecture_updated.md).

## Documents and Metadata

The initial knowledge base contains synthetic insurance documents in `data/raw/`: a cyber underwriting guide, commercial property policy wording, an underwriting appetite reference, and an exclusions reference.

Each document and chunk includes structured metadata: `document_id`, `name`/`title`, `document_type`, `coverage`, `jurisdiction`, `language`, `version`, `effective_date`, `section_title`, `section_path`, `page_start`/`page_end`, `chunk_id`, and `content_hash`. This metadata enables focused retrieval — for example, a cyber underwriting query for Quebec can be restricted to a specific coverage, jurisdiction, language, document type, or document version.

## Project Structure

```text
insurance-rag-assistant/
├── .github/workflows/ci.yml
├── data/
│   ├── raw/                          # Source insurance Markdown documents
│   └── processed/                    # Generated chunks.jsonl
├── vector_store/
│   └── qdrant/                       # Local persistent Qdrant data
├── artifacts/
│   └── retrieval_evaluation_report.json
├── docs/
│   └── architecture/
│       ├── evaluation-baseline.md
│       └── system-architecture.md
├── src/
│   └── insurance_rag_assistant/
│       ├── api/
│       │   ├── app.py                # FastAPI application
│       │   ├── dependencies.py
│       │   ├── routes.py
│       │   └── schemas/
│       │       └── evidence.py       # HTTP request/response contracts
│       ├── application/
│       │   └── evidence_retrieval_service.py
│       ├── evaluation/
│       │   └── evaluator.py
│       ├── generation/
│       │   ├── answer_generator.py
│       │   └── prompts.py
│       ├── ingestion/
│       │   ├── chunker.py
│       │   ├── loaders.py
│       │   └── pipeline.py
│       ├── llm/
│       │   ├── base.py
│       │   ├── fake.py
│       │   └── groq_client.py
│       ├── models/
│       │   ├── documents.py
│       │   ├── response.py
│       │   └── retrieval.py
│       ├── retrieval/
│       │   ├── embedder.py
│       │   ├── search.py
│       │   └── vector_store.py
│       ├── cli.py
│       └── config.py
├── tests/
│   ├── test_answer_generator.py
│   ├── test_api.py
│   ├── test_evaluation_normalisation.py
│   ├── test_ingestion.py
│   └── test_models.py
├── .env.example
├── compose.yaml
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.11+ |
| Environment and dependencies | uv |
| Web framework | FastAPI + uvicorn[standard] |
| LLM integration | Groq (structured output), provider-agnostic client interface |
| Structured data | Pydantic and pydantic-settings |
| Embeddings | `intfloat/multilingual-e5-base` (768 dimensions) via `sentence-transformers` |
| Vector store | Qdrant (`qdrant-client`), persisted locally |
| CLI framework | Typer |
| Testing | pytest and pytest-cov |
| Static typing | mypy |
| Linting and formatting | Ruff |
| Containerization | Docker and Docker Compose |
| CI | GitHub Actions |

## Setup

### Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- Internet access on first run to download `intfloat/multilingual-e5-base`
- A Groq API key only if using grounded generation through `insurance-rag ask`

### Installation

```bash
git clone https://github.com/Kwame-K/insurance-rag-assistant.git
cd insurance-rag-assistant
uv sync --all-groups
```

### Environment Configuration

```bash
cp .env.example .env
```

Add your Groq key if you plan to use LLM generation:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` files, API keys, private policy documents, or production claims data.

Inspect available CLI commands:

```bash
uv run insurance-rag --help
```

## Ingestion

Index the Markdown corpus into local Qdrant:

```bash
uv run insurance-rag ingest --recreate
```

This workflow loads Markdown documents from `data/raw/`, validates document metadata, creates overlapping chunks, generates normalized 768-dimensional embeddings, recreates the local `insurance_documents` Qdrant collection, stores vectors and metadata, and writes processed chunks to `data/processed/chunks.jsonl`.

## CLI Usage

### Search Source Passages

```bash
uv run insurance-rag search \
  --query "What are the multi-factor authentication requirements for cyber insurance underwriting?" \
  --top-k 5
```

With filters:

```bash
uv run insurance-rag search \
  --query "What are the referral criteria for cyber insurance?" \
  --coverage cyber \
  --jurisdiction Quebec \
  --language en \
  --document-type underwriting_guide \
  --top-k 5
```

### Generate a Grounded Answer

```bash
uv run insurance-rag ask \
  --query "Is water damage caused by a sudden plumbing leak covered under the commercial property policy?" \
  --coverage commercial_property \
  --language en \
  --top-k 5
```

Example output:

```text
Answer:
Yes. Water damage caused by a sudden and accidental escape of water
from plumbing, heating, air conditioning, or fire-protection systems
is covered under the commercial property policy, subject to the
applicable deductible.

Grounded: True
Confidence: high

Citations:

- [commercial_property_policy_v1:commercial-property-policy-3-water-damage-coverage:002]
  Document: Commercial Property Policy – Specimen Wording
  Section: 3. Water Damage Coverage
  Supporting quote: "Water damage caused by the sudden and accidental escape of water from plumbing, heating, air conditioning, or fire-protection systems is covered, subject to the applicable deductible."
```

The `ask` command uses Groq only after retrieval succeeds; its response is validated so that every citation must refer to a retrieved chunk and every quote must be found verbatim in that chunk.

| Argument | Purpose |
|---|---|
| `--query` | The insurance-policy question to answer |
| `--coverage` | Limits retrieval to the relevant insurance product or coverage corpus |
| `--language` | Selects the language context for the query and response |
| `--top-k` | Number of candidate chunks passed from retrieval to generation |

## FastAPI Service

The FastAPI service exposes evidence retrieval for the Underwriting Agent. Unlike the `ask` command, it does **not** call Groq or generate an answer — it only returns relevant source passages and metadata.

### Start the Service

```bash
uv run uvicorn insurance_rag_assistant.api.app:app --reload --port 8001
```

```text
GET  http://127.0.0.1:8001/health
POST http://127.0.0.1:8001/v1/retrieve-evidence
GET  http://127.0.0.1:8001/docs
```

### Health Check

```bash
curl "http://127.0.0.1:8001/health"
```

```json
{
  "service": "insurance-rag-assistant",
  "status": "ok"
}
```

### Retrieve Underwriting Evidence

```bash
curl -X POST "http://127.0.0.1:8001/v1/retrieve-evidence" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "REQ-MFA-001",
    "top_k": 3,
    "queries": [
      {
        "supported_finding_id": "UW-CYB-003",
        "query": "What are the multi-factor authentication requirements for cyber insurance underwriting?"
      }
    ]
  }'
```

```json
{
  "request_id": "REQ-MFA-001",
  "retrieval_status": "SUCCESS",
  "citations": [
    {
      "citation_id": "UW-CYB-003:cyber_underwriting_guide_v1:...",
      "source_document_id": "cyber_underwriting_guide_v1",
      "source_document_title": "Cyber Insurance Underwriting Guide",
      "section_reference": "3. Minimum Security Controls",
      "excerpt": "...multi-factor authentication must be enabled...",
      "relevance_score": 0.889,
      "supported_finding_id": "UW-CYB-003"
    }
  ],
  "unresolved_finding_ids": [],
  "knowledge_base_version": "insurance_documents"
}
```

| Status | Meaning |
|---|---|
| `SUCCESS` | At least one relevant citation was found for every requested finding |
| `PARTIAL` | Citations were found, but one or more findings have insufficient evidence |
| `INSUFFICIENT_CONTEXT` | No requested finding produced passages above the retrieval threshold |

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
Qdrant retrieval
        |
        v
Verbatim citations returned to UnderwritingDecision.evidence
```

| Project 3 owns | Project 4 owns |
|---|---|
| Documents and metadata | Submission validation |
| Chunking and embeddings | Completeness checks |
| Qdrant collection | Appetite and eligibility rules |
| Retrieval relevance | Risk score and pricing indication |
| Citation evidence | Referral, decline, and review workflow |
| Grounded answer generation | Final underwriting decision |

The RAG service does not accept, decline, price, or bind insurance. It provides documentary evidence only.

## Citation Validation Strategy

```text
Retrieved policy passages
    +
Structured LLM answer
    +
Pydantic schema validation
    +
Deterministic citation validation
    =
Auditable source-grounded response
```

### Structured Output Validation

Pydantic validates that the LLM response conforms to the application contract, including answer text, groundedness status, confidence level, citation list, citation chunk ID, document and section metadata, and supporting quotation. Malformed or incomplete LLM JSON is rejected before the answer reaches the caller.

### Verbatim Quote Validation

For every citation, the application verifies that the `chunk_id` belongs to the retrieved chunks for the current question, the citation `document_id` matches the cited chunk, the document name and section title match the cited chunk metadata, and the supporting quote is present as a contiguous substring of the source chunk.

The validation layer normalizes only harmless formatting differences (non-breaking spaces, typographic apostrophes, quotation marks, and dash variants). It does **not** lowercase text, remove words, use semantic similarity, or accept a paraphrased quote as evidence:

```text
Accepted after typography normalization
Source:    fire-protection systems
Generated: fire‑protection systems

Rejected as a paraphrase
Source:    sudden and accidental escape of water
Generated: sudden plumbing leak damage
```

### Groundedness

`Grounded: True` means the response contains citations that passed the project's provenance, metadata, and verbatim-quote checks. It does not mean the system has made a legally binding coverage decision. If the available documents do not contain enough evidence, the correct behavior is to state that the answer cannot be verified rather than infer or invent one.

## Evaluation

The project includes retrieval and generation evaluation assets in `src/insurance_rag_assistant/evaluation/` and `artifacts/retrieval_evaluation_report.json`. Evaluation focuses on retrieval relevance, correct document selection, grounding of generated responses, citation validity, and explicit abstention when retrieval is insufficient.

```bash
uv run insurance-rag evaluate-retrieval
```

Useful evaluation signals:

| Signal | Question answered |
|---|---|
| Recall@k | Was the necessary policy passage retrieved? |
| MRR | How highly was the necessary passage ranked? |
| Grounded-answer rate | How often does the system produce source-supported answers? |
| Citation validity | Are supporting quotes present in the cited chunks? |
| Citation completeness | Do citations support all material claims in the answer? |
| Refusal correctness | Does the system decline to answer when evidence is insufficient? |

See `docs/architecture/evaluation-baseline.md` for the latest recorded results.

## Quality Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
```

Format the project locally:

```bash
uv run ruff format .
```

The GitHub Actions workflow runs formatting checks, linting, static typing, and tests on pushes and pull requests to `main`.

## Docker

```bash
docker compose up --build
docker compose up --build -d
docker compose down
```

This service can also be orchestrated centrally by the `insurance-agentic-platform` Docker Compose project.

## Limitations

- Answer quality depends on document quality, chunking, metadata, embeddings, retrieval configuration, and the LLM.
- A validated quote proves that the quoted text comes from the cited chunk; it does not alone prove that no other policy provision changes the outcome.
- The assistant does not replace review of the complete policy, endorsements, schedules, declarations, or facts of a specific claim.
- The system must not be used to make binding coverage, underwriting, pricing, or claims decisions.
- Real policy documents and customer information require appropriate access controls, retention rules, and privacy protections.

## Design Principles

- Retrieval before generation: no LLM response is generated without retrieved context.
- Verbatim citations: generated quotes must be found in retrieved chunks.
- Explicit abstention: insufficient evidence returns an explicit insufficient-context state.
- Metadata-aware retrieval: filters prevent mixing products, jurisdictions, or document versions.
- Service boundaries: downstream agents consume an API rather than accessing Qdrant directly.
- LLM separation: evidence retrieval remains usable without Groq or another generation provider.
- Testability: retrieval, generation, embeddings, vector storage, and API contracts remain independently testable.

## Roadmap

- [x] Markdown ingestion and chunking
- [x] 768-dimensional multilingual E5 embeddings
- [x] Local Qdrant vector storage
- [x] Metadata-aware semantic search
- [x] Grounded Groq generation with citation validation
- [x] Retrieval and generation evaluation framework
- [x] FastAPI evidence-retrieval endpoint
- [x] Integration with the Project 4 Underwriting Agent
- [ ] Add an API endpoint for grounded question answering (not just evidence retrieval)
- [ ] Add document-version selection and effective-date routing
- [ ] Add asynchronous retrieval and production deployment configuration
- [ ] Add retrieval observability, latency metrics, and audit logs
- [ ] Support a remotely deployed Qdrant service with authenticated access

## Position in the Insurance Agentic Platform

| Project | Repository | Role |
|---|---|---|
| 1 | insurance-submission-extractor | Structured submission intake and validation |
| 2 | insurance-data-analyst-agent | Controlled portfolio analytics (loss ratio, deterministic SQL) |
| 3 | insurance-rag-assistant (this repo) | Insurance Knowledge Agent with grounded, cited retrieval |
| 4 | underwriting-agent | Deterministic underwriting rules, risk scoring, pricing, and evidence retrieval |
| — | insurance-agentic-platform | Docker Compose orchestration layer for the running services |

## Recommended Technology Upgrades

| Area | Current | Recommended | Benefit |
|---|---|---|---|
| Observability | No tracing | Langfuse or OpenTelemetry around retrieval and generation | Visibility into retrieval latency, embedding cost, and citation-rejection rate |
| Retrieval quality | Single-stage semantic search | Add a reranker (e.g. cross-encoder) after Qdrant retrieval | Improves precision before the relevance gate, reducing false abstentions |
| Vector store deployment | Local persisted Qdrant only | Managed or remote Qdrant with authenticated access (already on roadmap) | Enables multi-instance and production deployment |
| Evaluation | Retrieval-only evaluation report | Add LLM-as-a-judge for answer faithfulness beyond exact-quote matching | Detects subtle grounding issues not caught by substring matching |
| API surface | Evidence retrieval only | Add a grounded question-answering endpoint (already on roadmap) | Lets the Underwriting Agent request synthesized answers, not just raw citations |
| Async processing | Synchronous ingestion and retrieval | Async FastAPI endpoints with async Qdrant client | Better throughput under concurrent underwriting requests |

## Improvements and Next Steps

1. Add a reranking stage (cross-encoder) between Qdrant retrieval and the relevance gate to improve precision on ambiguous multi-section questions.
2. Deliver the roadmap item for a grounded question-answering API endpoint, extending `application/evidence_retrieval_service.py` beyond raw evidence retrieval.
3. Instrument retrieval and generation with tracing (Langfuse or OpenTelemetry) to monitor citation-rejection rate and retrieval latency in production.
4. Add document-version selection and effective-date routing so that outdated policy wordings are excluded from retrieval automatically.
5. Move from local Qdrant storage to a remotely deployed, authenticated Qdrant instance to support the platform's Docker Compose orchestration at scale.
6. Extend the evaluation framework with LLM-as-a-judge scoring for answer faithfulness, complementing the existing verbatim-quote validation.

## Agentic AI Best Practices Applied Here

- **Retrieval before generation**: the answer generator can only use retrieved chunks as context, which is the core guardrail against hallucinated policy language.
- **Deterministic verification of generative output**: verbatim citation validation treats every LLM claim as unverified until it is matched, character-for-character, against retrieved source text — a strong example of not trusting LLM output by default.
- **Explicit abstention over fabrication**: `INSUFFICIENT_CONTEXT` and `grounded=false` states let the system say "I don't know" instead of inventing plausible-sounding coverage language, which is critical in a regulated domain.
- **Clean service boundary for downstream agents**: the Underwriting Agent consumes a versioned HTTP API (`/v1/retrieve-evidence`) rather than querying Qdrant directly, keeping retrieval internals swappable.
- **LLM-independent core capability**: evidence retrieval works without Groq, so the RAG service degrades gracefully rather than failing completely if the LLM provider is unavailable.
- **Next practice to adopt**: expose the evidence-retrieval and grounded-answer capabilities as MCP tools so the Underwriting Agent (or any future orchestrator) can call them through a standardized tool contract instead of a bespoke HTTP client.
