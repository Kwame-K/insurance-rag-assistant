# RAG Evaluation Baseline

## Purpose

This document records the initial evaluation baseline for the **Insurance RAG Assistant**.

The baseline focuses on whether the system retrieves relevant policy evidence, generates answers that remain grounded in retrieved context, and produces citations that can be deterministically verified against the source chunks.

This is not a benchmark of legal accuracy, insurance coverage determination, or general LLM capability. It is a project-specific quality baseline for the current corpus, chunking strategy, embedding configuration, prompt, and response-validation rules.

## Evaluation Principles

A RAG answer can fail at several layers. Evaluation must not treat all incorrect answers as a single failure class.

| Layer | Evaluation question |
|---|---|
| Corpus | Does the indexed document set contain the needed policy language? |
| Chunking | Is the needed information preserved in one or more useful chunks? |
| Retrieval | Is the needed chunk retrieved for the question? |
| Ranking | Does the needed chunk appear high enough in the result set? |
| Generation | Does the answer accurately express the retrieved policy information? |
| Citation validity | Is each quote actually present in its cited retrieved chunk? |
| Citation completeness | Do citations support every material claim in the answer? |
| Refusal behavior | Does the system decline to make unsupported claims? |

A schema-valid answer is not necessarily grounded. A citation that exists is not necessarily complete. A high retrieval score does not establish a policy conclusion.

## Evaluation Criteria

| Criterion | Description |
|---|---|
| Response schema validity | The response conforms to the `RAGResponse` Pydantic contract. |
| Retrieval relevance | The required policy chunk appears in the retrieved `top-k` results. |
| Answer correctness | The response accurately states the policy language relevant to the test case. |
| Groundedness | The answer makes no material claims beyond retrieved evidence. |
| Citation provenance | Every citation references a chunk retrieved for the current query. |
| Metadata consistency | Citation document ID, document name, and section title match the cited chunk. |
| Verbatim quote validity | Each supporting quote appears as a contiguous substring of its source chunk after narrow Unicode normalization. |
| Citation completeness | Material answer claims, including deductibles and conditions, have supporting evidence. |
| Refusal correctness | The system clearly identifies insufficient evidence instead of inventing an answer. |
| Retrieval-score integrity | Displayed scores are non-default values correctly propagated from retrieval or reranking. |

## Current Baseline Case

The first manually verified end-to-end baseline is the commercial-property water-damage scenario.

| Case ID | Scenario | Expected outcome | Status |
|---|---|---|---|
| `water_damage_plumbing_leak` | Question about water damage caused by a sudden plumbing leak | Covered subject to deductible; vacancy condition is surfaced; all citations pass validation | Passed |

### Query

```text
Is water damage caused by a sudden plumbing leak covered under the commercial property policy?
```

### Expected Evidence

| Policy topic | Expected source section | Expected answer requirement |
|---|---|---|
| Coverage grant | `3. Water Damage Coverage` | State that sudden and accidental escape of water from the listed systems is covered, subject to deductible |
| Deductible | `2. Deductible` | State the CAD 5,000 deductible per occurrence |
| Vacancy condition | `6. Vacancy Condition` | State that liability for water damage does not apply after more than 60 consecutive days of vacancy |

### Observed Result

The system returned:

- `Grounded: True`
- `Confidence: high`
- One citation for the water-damage coverage grant
- One citation for the water-damage deductible
- One citation for the vacancy condition
- Supporting quotations that were accepted by deterministic verbatim validation

This case demonstrates that the current pipeline can synthesize a response from multiple policy sections while preserving separate evidence for the coverage grant, deductible, and material condition.

## Citation-Validation Finding

During initial development, the model generated a citation quote that was not present verbatim in the cited water-damage chunk. The application correctly raised:

```text
ValueError: Citation quote is not verbatim in chunk
```

The validation was updated to normalize only harmless Unicode and whitespace variants before matching. This resolves formatting differences such as a non-breaking hyphen in `fire‑protection` without allowing semantic weakening of the evidence check.

The policy remains:

- Accept narrow Unicode typography differences.
- Reject added, removed, or reworded words.
- Reject ellipsis-based shortened quotations unless the remaining quote is still a contiguous source substring.
- Reject paraphrases even when their meaning is similar to the source text.

This distinction is necessary because a citation is evidence, not merely an indication that a policy section may be relevant.

## Retrieval-Score Finding

The initial successful water-damage output displayed a retrieval score of `0.000` for each cited chunk.

The cited chunks were relevant and passed all citation checks, so this observation does not establish a retrieval-quality failure. It does indicate that score propagation or CLI rendering must be reviewed before retrieval scores are presented as diagnostic information.

Possible causes include:

- A default `0.0` value on the retrieval result model.
- A score not copied from the vector-search result into the `RetrievedChunk` model.
- A CLI formatter reading a different score field than the retriever populates.
- Scores intentionally omitted or unavailable after retrieval/reranking.

Until this is resolved, retrieval rank is preferable to a displayed numeric score, or the CLI should omit the score field entirely.

## Baseline Evaluation Dataset

The next evaluation milestone is a versioned, manually reviewed dataset of insurance-policy questions. Each case should define the question, retrieval filters, required source chunks, expected groundedness, and answer/citation assertions.

Suggested fixture format:

```json
{
  "id": "water_damage_plumbing_leak",
  "question": "Is water damage caused by a sudden plumbing leak covered under the commercial property policy?",
  "coverage": "commercial_property",
  "language": "en",
  "top_k": 5,
  "expected_grounded": true,
  "expected_answer_contains": [
    "sudden and accidental",
    "deductible",
    "CAD 5,000",
    "60 consecutive days"
  ],
  "expected_chunk_ids": [
    "commercial_property_policy_v1:commercial-property-policy-3-water-damage-coverage:002",
    "commercial_property_policy_v1:commercial-property-policy-2-deductible:001",
    "commercial_property_policy_v1:commercial-property-policy-6-vacancy-condition:005"
  ]
}
```

Expected test categories:

| Category | Example evaluation scenario |
|---|---|
| Coverage | Sudden and accidental escape of water |
| Deductible | Applicable water-damage deductible |
| Exclusion | A peril explicitly excluded by the policy |
| Condition | Vacancy longer than the permitted period |
| Definition | Meaning of a defined policy term |
| Multi-section reasoning | Coverage subject to deductible and condition |
| Unanswerable query | Peril or product not described in the indexed corpus |
| Ambiguous query | Question requiring policy version, jurisdiction, or additional facts |
| Language behavior | French question over an English policy corpus, with exact source quotes preserved |
| Citation failure | Model outputs a paraphrased quote or an unknown chunk ID |

The initial target should be 20–30 curated cases before comparing prompts, models, embedding configurations, retrieval strategies, or rerankers.

## Retrieval Metrics

Evaluation should measure retrieval independently from answer quality.

| Metric | Definition | Why it matters |
|---|---|---|
| Recall@k | Proportion of cases where a required source chunk appears among the first `k` retrieved results | The generator cannot reliably cite evidence it never receives |
| MRR | Reciprocal rank of the first required chunk, averaged across cases | Measures whether key evidence appears early enough in the result list |
| Precision@k | Share of the first `k` chunks that are relevant | Helps identify noisy context that can distract generation |
| Metadata accuracy | Share of retrieved chunks with correct document, version, section, and language metadata | Prevents mixing incompatible policy wording |

A retrieval experiment should be evaluated with the same dataset, corpus version, chunking configuration, embedding model, filters, and `top-k` value.

## Answer and Citation Metrics

The answer layer should be assessed separately after confirming that required evidence was retrieved.

| Metric | Definition | Passing expectation |
|---|---|---|
| Schema-valid response rate | Responses that pass Pydantic validation | 100% for accepted responses |
| Citation provenance rate | Citations pointing only to retrieved chunks | 100% |
| Metadata-consistent citation rate | Citations whose source metadata matches the cited chunk | 100% |
| Verbatim citation-validity rate | Citations whose quotes are present in source chunks | 100% |
| Grounded-answer rate | Answers making only source-supported material claims | Target established from reviewed cases |
| Citation completeness | Material answer claims supported by one or more citations | Target established from reviewed cases |
| Correct-refusal rate | Unsupported questions that result in a clear refusal or insufficient-evidence response | 100% for reviewed unanswerable cases |

For this project, provenance, metadata consistency, and verbatim quote validity are hard safety requirements. A response that fails one of these checks must not be counted as an accepted grounded answer.

## Regression Policy

Repeat the evaluation suite after any material change to:

- Policy corpus or document versions.
- Parsing and chunking configuration.
- Embedding model or vector dimension.
- Retrieval filters, `top-k`, or hybrid-search weights.
- Reranking model or reranking policy.
- LLM provider, model, or generation parameters.
- Prompt wording.
- Pydantic response schema.
- Citation-normalization or validation logic.

Store each evaluation report with a timestamp and record the versions of the corpus, index, embedding model, prompt, and application code used for the run.

## Current Limitations

- The current documented baseline contains one manually verified end-to-end policy question.
- The baseline does not yet establish Recall@k, MRR, precision, cost, latency, or model-comparison metrics.
- The initial corpus and examples must not be treated as a complete representation of any production insurance policy portfolio.
- Successful quote validation confirms source presence, not legal interpretation or applicability to a real claim.
- The system has not yet been evaluated on a broad multilingual, multi-jurisdictional, multi-version policy corpus.
- Retrieval scores are currently not reliable for CLI interpretation until score propagation is verified.

## Next Evaluation Steps

1. Fix or suppress the displayed `0.000` retrieval scores.
2. Build and version the first 20–30 reviewed evaluation cases.
3. Add automated assertions for retrieved chunk IDs, groundedness, required answer content, and citation validity.
4. Establish baseline Recall@k, MRR, citation validity, citation completeness, and refusal correctness.
5. Compare retrieval configurations using the fixed dataset before changing the embedding model or adding a reranker.
6. Add one controlled regeneration attempt for citation-validation failures, then fail safely if validation still fails.
7. Run the complete evaluation suite in GitHub Actions using mocked provider calls for unit tests and isolated integration tests for live providers.
