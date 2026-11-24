# RAG Evaluation Baseline

## Purpose

This document records the evaluation baseline for the **Insurance RAG Assistant**.

The baseline covers three questions: does the system retrieve the relevant evidence, does it abstain when the corpus does not support an answer, and do generated answers stay grounded with citations that can be verified against the source chunks.

This is not a benchmark of legal accuracy, insurance coverage determination, or general LLM capability. It is a project-specific quality baseline for the current corpus, chunking strategy, embedding configuration, retrieval configuration, prompt, and response-validation rules.

Last updated: 2026-10-02.

## Evaluation Principles

A RAG answer can fail at several layers. Evaluation must not treat all incorrect answers as a single failure class.

| Layer | Evaluation question |
|---|---|
| Corpus | Does the indexed document set contain the needed language? |
| Chunking | Is the needed information preserved in one or more useful chunks? |
| Retrieval | Is the needed chunk retrieved for the question? |
| Ranking | Does the needed chunk appear high enough in the result set? |
| Abstention | Does the system decline when the corpus cannot answer, and answer when it can? |
| Generation | Does the answer accurately express the retrieved information? |
| Citation validity | Is each quote actually present in its cited retrieved chunk? |
| Citation completeness | Do citations support every material claim in the answer? |

A schema-valid answer is not necessarily grounded. A citation that exists is not necessarily complete. A high retrieval score does not establish a policy conclusion.

## Evaluation Sets

| Set | File | Documents | Cases | Answerable | Unanswerable |
|---|---|---|---:|---:|---:|
| Public documents | `evaluation/cases_real.json` | OSFI B-10, B-13, E-21; Code civil du Québec | 29 | 23 | 6 |
| Synthetic documents | `evaluation/cases.json` | Commercial property, cyber underwriting, appetite, exclusions | 35 | 21 | 14 |

The synthetic set includes three `holdout_*` cases that were not used to choose the retrieval configuration. They are the only independent measurement in the current evaluation, and must not be used for tuning.

16 of the public-document cases also define expected section titles, enabling section-level metrics.

## Retrieval Metrics

All retrieval metrics are computed at `top-k` = 5.

| Metric | Definition | Why it matters |
|---|---|---|
| Macro Recall@5 | For each answerable case, the share of expected documents found in the top 5; averaged across cases | The generator cannot cite evidence it never receives |
| MRR | Reciprocal rank of the first expected document, averaged across answerable cases | Whether key evidence appears early |
| Section Hit@1 / Hit@5 | Share of section-labelled cases whose expected section is first / in the top 5 | Citation precision |
| Section MRR | Reciprocal rank of the first expected section | Section ranking quality |
| Correct abstention rate | Share of unanswerable cases where the system abstains | Avoiding unsupported answers |
| Answerable retrieval rate | Share of answerable cases the system accepts | Avoiding unnecessary refusals |
| False abstention rate | Share of answerable cases the system refuses | The cost of the abstention gate |

Recall and MRR are document-level: they do not check that the right passage inside the document was found. Section metrics address this for the 16 labelled cases.

## Current Results

Configuration: pgvector backend, hybrid search on, cross-encoder reranker on, 20 rerank candidates, abstention threshold 0, `top-k` 5.

| Set | Cases | Recall@5 | MRR | Correct abstention | Answerable retrieval | False abstention |
|---|---:|---:|---:|---:|---:|---:|
| Public documents | 29 | 1.000 | 0.922 | 1.000 | 1.000 | 0.000 |
| Synthetic documents | 35 | 0.929 | 1.000 | 1.000 | 0.905 | 0.095 |

Section-level results on the public-document set (16 cases): Hit@1 0.688, Hit@5 1.000, Section MRR 0.812.

On the synthetic set, Recall@5 is 0.929 because three cases expect two documents and retrieve one (`property_flood_exclusion_en`, `property_gradual_leak_en`, `cyber_mfa_referral_en`). This is unrelated to abstention.

## Configuration Decisions

Measured on the public-document set (29 cases).

| Configuration | Recall@5 | MRR | Section MRR |
|---|---:|---:|---:|
| Dense only, 20 rerank candidates | 0.957 | 0.873 | 0.802 |
| Hybrid, 20 rerank candidates | 1.000 | 0.922 | 0.812 |
| Hybrid, 100 rerank candidates | 0.957 | 0.906 | 0.802 |

- **Hybrid search is kept.** It adds 0.043 Recall@5 and 0.049 MRR over dense-only retrieval. The gain depends mostly on one case, `b13_patch_management_en`: with dense search alone, its B-13 patch-management sections rank beyond the reranker's 20-candidate pool, so the top 5 are all from B-10. With hybrid search the correct document is ranked first.
- **A pool of 100 candidates is worse than 20.** A larger pool gave the reranker more general passages to over-score, and one B-13 case fell out of the top 5.
- **The result was reproduced** with explicit settings and gave identical metrics.

These conclusions rest on small sets. A change of a few points corresponds to one or two cases.

### Observed reranker behaviour

For the short question "What are OSFI's expectations on patch management?", the top 5 contained general passages from B-10 (for example the overview and the "Management of third-party risk" section) and from B-13's scope sections, not the specific "2.6 Patch management" section. The more specific question "What are OSFI's expectations for applying patches in a timely and controlled manner?" returned B-13 2.6.1 and 2.6 at ranks 1 and 2. Short or generic queries tend to favour broad passages that repeat "OSFI expects".

## Known Limitation: Reranker Abstention Threshold

The threshold of 0 was chosen on the public-document cases. On those cases the lowest top rerank score among answerable questions is 1.30 and the highest among unanswerable questions is -0.64, so 0 separates them.

On the synthetic set the same threshold wrongly refuses 2 of 21 answerable cases. In both, the expected document is ranked first (MRR 1.000), so the failure is the gate, not retrieval.

| Case | Question | Top rerank score |
|---|---|---:|
| `holdout_burst_pipe_en` | Is a burst pipe covered under the commercial property policy? | -2.17 |
| `holdout_cyber_mfa_loose_en` | Do cyber applicants need MFA before we quote? | -0.46 |

Why the threshold stays at 0:

- A threshold low enough to accept both cases (between -2.87 and -2.17 on the synthetic set) would accept the closest public-document unanswerable case (-0.64).
- The holdout cases are the only measurement not used to choose the configuration; tuning the threshold on them would remove it.
- The public-document set has only 6 unanswerable cases, so a lower threshold is weakly supported.

Practical consequence: short or informally phrased questions can score below 0 even when the correct passage is first, and the system will abstain. Re-evaluate the threshold when the unanswerable set grows or the corpus is extended.

## Adjustments Made After Seeing Results

Recorded so the evaluation is not read as fully independent.

- `b13_third_party_en` (third-party risk answered from B-13/E-21) was removed after B-10 was added, because B-10 answers that question directly. The decision was made after seeing that its expected documents no longer ranked in the top 5. `b10_third_party_providers_en` replaces it.
- The configuration (hybrid on, 20 candidates, threshold 0) was selected using the public-document cases. Only the three synthetic holdout cases were kept out of tuning.

## Evaluation Criteria (Generation)

| Criterion | Description |
|---|---|
| Response schema validity | The response conforms to the `RAGResponse` Pydantic contract. |
| Retrieval relevance | The required chunk appears in the retrieved `top-k` results. |
| Answer correctness | The response accurately states the language relevant to the test case. |
| Groundedness | The answer makes no material claims beyond retrieved evidence. |
| Citation provenance | Every citation references a chunk retrieved for the current query. |
| Metadata consistency | Citation document ID, document name, and section title match the cited chunk. |
| Verbatim quote validity | Each quote appears as a contiguous substring of its source chunk after narrow Unicode normalization. |
| Citation completeness | Material claims, including deductibles and conditions, have supporting evidence. |
| Refusal correctness | The system identifies insufficient evidence instead of inventing an answer. |

Generation evaluation runs with `insurance-rag evaluate --with-generation` and makes one Groq request per case. The results above are retrieval-only.

## End-to-End Baseline Case

The first manually verified end-to-end case is the commercial-property water-damage scenario.

| Case ID | Scenario | Expected outcome | Status |
|---|---|---|---|
| `water_damage_plumbing_leak` | Water damage caused by a sudden plumbing le
