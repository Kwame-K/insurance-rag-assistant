# Insurance Knowledge Agent

[![Continuous Integration](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

A provider-agnostic Retrieval-Augmented Generation (RAG) application that answers insurance-policy questions from indexed policy wordings with validated, source-grounded citations.

The project combines document chunking, vector retrieval, structured LLM outputs, Pydantic data contracts, deterministic citation validation, and a command-line interface. It is designed as the insurance knowledge layer of a future agentic underwriting workflow.

> **Important:** This project is an educational and portfolio demonstrator. It does not provide legal advice, underwriting decisions, claims decisions, binding coverage confirmation, or policy interpretation for a real claim. The applicable policy wording and the authorized insurer remain controlling.

## Problem

Insurance policy wording is long, technical, and distributed across coverage grants, exclusions, deductibles, definitions, conditions, and endorsements. A question that appears simple — for example, whether a plumbing leak is covered — may require evidence from several different sections of a policy.

A generic LLM may provide an answer that sounds plausible but omits a deductible, fails to mention a material condition, cites the wrong section, or invents policy language.

This project addresses that problem by retrieving relevant policy passages before generation and validating each citation against the retrieved source text.

The application can answer questions about:

- Covered perils and coverage grants
- Policy exclusions and limitations
- Deductibles and sublimits
- Definitions and insured-property terminology
- Conditions, including vacancy and reporting conditions
- Policy-specific questions that require evidence from multiple sections
- Questions that cannot be supported by the indexed corpus

The application does not decide whether a real claim is covered. It provides a traceable answer based only on the policy passages available to the system.

## Key Features

- Provider-agnostic LLM architecture
- Structured JSON answer generation with Pydantic validation
- Retrieval over chunked insurance-policy documents
- 768-dimensional embedding model and vector-based similarity search
- Coverage, language, and `top-k` retrieval filters
- Chunk-level document metadata: document ID, document name, section title, and chunk ID
- Source-grounded answers with supporting quotations
- Deterministic validation that every citation refers to a retrieved chunk
- Deterministic verification that each supporting quote is verbatim in its cited source chunk
- Limited Unicode normalization for harmless typographic differences without accepting paraphrases
- Rejection of citations with inconsistent document or section metadata
- Groundedness and confidence fields in the structured response
- CLI output designed for inspection and auditability
- Automated formatting, linting, static typing, and tests

## Architecture

```text
                            User Question
                                  |
                                  v
                         Insurance RAG CLI
                                  |
                                  v
                      Query Filters and Retrieval
            coverage / language / top-k / vector similarity
                                  |
                                  v
                  Retrieved Policy Chunks + Metadata
                                  |
                                  v
                    Provider-Agnostic LLM Client
                                  |
                                  v
              Structured Answer + Citation JSON Output
                                  |
                                  v
                    Pydantic Schema Validation
                                  |
                                  v
                Deterministic Citation Validation
           chunk ID + metadata + verbatim quote validation
                                  |
                  +---------------+---------------+
                  |                               |
                  v                               v
          Invalid citation                    Valid response
          Reject response                         |
                                                  v
                                Grounded CLI Answer + Evidence
```

For a detailed architecture description, component responsibilities, retrieval flow, citation-validation flow, and future underwriting workflow, see [System Architecture](docs/architecture/system-architecture.md).

## Project Structure

```text
insurance-rag-assistant/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── documents/
│   ├── evaluations/
│   └── indexes/
├── docs/
│   └── architecture/
│       └── system-architecture.md
├── src/
│   └── insurance_rag_assistant/
│       ├── generation/
│       │   ├── answer_generator.py
│       │   └── prompts.py
│       ├── llm/
│       │   ├── base.py
│       │   └── factory.py
│       ├── retrieval/
│       │   ├── chunking.py
│       │   ├── indexing.py
│       │   └── search.py
│       ├── models/
│       │   ├── response.py
│       │   └── retrieval.py
│       ├── cli.py
│       └── config.py
├── tests/
├── .env.example
├── pyproject.toml
└── uv.lock
```

> Adjust directory names in this diagram if your repository uses a different layout. The important separation is retrieval, generation, LLM clients, data contracts, and CLI orchestration.

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.11+ |
| Environment and dependencies | uv |
| LLM integration | Provider-agnostic client interface |
| Structured data | Pydantic |
| Retrieval | Chunked policy documents and vector similarity search |
| Embeddings | 768-dimensional embedding model |
| Testing | pytest |
| Static typing | mypy |
| Linting and formatting | Ruff |
| CI | GitHub Actions |

## Setup

### Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- At least one configured LLM provider supported by the project
- An indexed insurance-document corpus before running policy queries

### Installation

```bash
git clone https://github.com/Kwame-K/insurance-rag-assistant.git
cd insurance-rag-assistant
uv sync --all-groups
```

### Environment Configuration

Create a local environment file:

```bash
cp .env.example .env
```

Configure the LLM provider, API key, model, and storage/index settings using the variables defined in `.env.example`.

Do not commit `.env` files, API keys, private policy documents, or production claims data.

Inspect the commands available in the installed CLI:

```bash
uv run insurance-rag --help
```

## Usage

### Ask a Policy Question

Run the CLI against the indexed commercial-property corpus:

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

### Use Retrieval Filters

The CLI accepts filters that help constrain the retrieval context:

```bash
uv run insurance-rag ask \
  --query "What deductible applies to water damage claims?" \
  --coverage commercial_property \
  --language en \
  --top-k 5
```

| Argument | Purpose |
|---|---|
| `--query` | The insurance-policy question to answer |
| `--coverage` | Limits retrieval to the relevant insurance product or coverage corpus |
| `--language` | Selects the language context for the query and response |
| `--top-k` | Number of candidate chunks passed from retrieval to generation |

## Citation Validation Strategy

The project deliberately separates probabilistic LLM behavior from deterministic evidence checks.

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

Pydantic validates that the LLM response conforms to the application contract, including fields such as:

- Answer text
- Groundedness status
- Confidence level
- Citation list
- Citation chunk ID
- Document and section metadata
- Supporting quotation

Malformed or incomplete LLM JSON is rejected before the answer reaches the CLI.

### Verbatim Quote Validation

For every citation, the application verifies that:

1. The `chunk_id` belongs to the retrieved chunks for the current question.
2. The citation `document_id` matches the cited chunk.
3. The document name and section title match the cited chunk metadata.
4. The supporting quote is present as a contiguous substring of the source chunk.

The validation layer normalizes only harmless formatting differences, such as non-breaking spaces, typographic apostrophes, quotation marks, and dash variants. It does **not** lowercase text, remove words, use semantic similarity, or accept a paraphrased quote as evidence.

For example, this is accepted after limited typography normalization:

```text
Source:    fire-protection systems
Generated: fire‑protection systems
```

This is rejected because it is a paraphrase rather than a source quotation:

```text
Source:    sudden and accidental escape of water
Generated: sudden plumbing leak damage
```

### Groundedness

`Grounded: True` means the response contains citations that passed the project's provenance, metadata, and verbatim-quote checks. It does not mean the system has made a legally binding coverage decision.

If the available documents do not contain enough evidence to support an answer, the correct system behavior is to state that the answer cannot be verified from the retrieved policy text rather than infer or invent an answer.

## Evaluation

The evaluation strategy should test retrieval and generation separately. A response may fail because the necessary passage was not retrieved, because the model misunderstood a retrieved passage, or because a citation was not sufficiently complete for the claim made.

A policy-question evaluation dataset should include:

- Coverage questions
- Exclusions and limitations
- Deductibles, limits, and sublimits
- Definitions and policy conditions
- Multi-section questions
- Contradictory or ambiguous policy language
- Questions without supporting evidence in the corpus
- English and French questions where the corpus supports them
- Citation-validity failure cases

Useful evaluation signals include:

| Signal | Question answered |
|---|---|
| Recall@k | Was the necessary policy passage retrieved? |
| MRR | How highly was the necessary passage ranked? |
| Grounded-answer rate | How often does the system produce source-supported answers? |
| Citation validity | Are supporting quotes present in the cited chunks? |
| Citation completeness | Do citations support all material claims in the answer? |
| Refusal correctness | Does the system decline to answer when evidence is insufficient? |

## Quality Checks

Run all local checks before committing:

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

The GitHub Actions workflow should run formatting checks, linting, static typing, and tests on pushes and pull requests to `main`.

## Limitations

- Answer quality depends on document quality, chunking, metadata, embeddings, retrieval configuration, and the LLM.
- A validated quote proves that the quoted text comes from the cited chunk; it does not alone prove that no other policy provision changes the outcome.
- Retrieval scores should be displayed only when they are correctly propagated from the retriever and remain meaningful after any reranking step.
- The assistant does not replace review of the complete policy, endorsements, schedules, declarations, or facts of a specific claim.
- The system must not be used to make binding coverage, underwriting, pricing, or claims decisions.
- Real policy documents and customer information require appropriate access controls, retention rules, and privacy protections.

## Roadmap

This project is the insurance knowledge layer of a broader agentic insurance and risk platform.

### Project 1: Insurance Submission Extractor

- Extract structured submission data from unstructured broker materials
- Normalize insurance-domain fields and validate business rules
- Route incomplete or inconsistent submissions for human review

### Project 2: Data Analyst Agent

- Controlled SQL and Python tool calling
- Portfolio metrics, claims summaries, and chart generation
- Auditable tool execution

### Project 3: Insurance Knowledge Agent

- Insurance-document ingestion and versioning
- Retrieval, citations, and grounded policy answers
- Hybrid retrieval, reranking, evaluation, and observability

### Project 4: Underwriting Agent

- Workflow orchestration
- Submission extraction, policy retrieval, analytics, pricing, and recommendation
- Human-in-the-loop review gates
- Underwriting report generation

### Project 5: Multi-Agent Risk Committee

- Orchestrator, data, actuary, risk, underwriting, and reviewer roles
- Evidence sharing, verification, escalation, and disagreement handling

### Project 6: Production Agent Platform

- FastAPI backend
- Persistent storage and caching
- Containerized deployment
- Observability, evaluation, access control, and CI/CD

## Development Principles

- Do not trust unvalidated LLM output.
- Do not treat a paraphrase as a verified quotation.
- Do not answer beyond the evidence available in retrieved policy passages.
- Keep retrieval, generation, validation, and decision logic separate.
- Prefer deterministic controls for high-impact insurance rules.
- Version documents, chunking settings, embeddings, prompts, schemas, and evaluation results.
- Use synthetic, anonymized, or properly authorized data only.
- Preserve human review for ambiguous, incomplete, inconsistent, or high-impact cases.

## License

This project is intended for educational and portfolio purposes. Add a license appropriate for your intended use before sharing or deploying it publicly.

## Author

Built by Kristian Laban.

- GitHub: [@Kwame-K](https://github.com/Kwame-K)
- LinkedIn: [Kwame Kristian LABAN](https://www.linkedin.com/in/kwame-kristian-laban/)
